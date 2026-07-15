"""Task-guided query vectors (LatentBriefing).

Vanilla AM scores keys with random probe queries. LatentBriefing instead uses the
worker's actual query vectors -- the per-(layer, head) Q states the model produces
when it reads the task prompt -- so compaction keeps the KV entries most relevant
to *this* task, not to a generic probe distribution.

extract_task_queries() runs the task prompt through the model, captures each
attention layer's post-projection / post-q_norm / post-RoPE query states via
forward hooks, and groups them by KV head (accounting for GQA). The result plugs
straight into compact_cache(cache, ..., queries=...).
"""

import torch

from transformers.models.qwen3.modeling_qwen3 import rotate_half


def extract_task_queries(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Per-layer task query vectors, grouped by KV head.

    Runs `input_ids` (1, S) through `model` once and captures each layer's query
    states (q_proj -> q_norm -> RoPE). Under GQA each KV head is shared by
    `n_heads // n_kv_heads` query heads, so those query heads' vectors are pooled
    into one probe set per KV head.

    Returns a list of length num_layers; entry `l` has shape
    (n_kv_heads, S * group_size, head_dim) -- ready for compact_cache(queries=...).
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    group = n_heads // n_kv

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            q = module.q_proj(hidden)                             # (1, S, n_heads*head_dim)
            q = q.view(hidden.shape[0], hidden.shape[1], n_heads, head_dim)
            if hasattr(module, "q_norm"):
                q = module.q_norm(q)
            q = q.transpose(1, 2)                                 # (1, n_heads, S, head_dim)
            q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))  # RoPE
            captured[layer_idx] = q[0]                            # (n_heads, S, head_dim)
        return hook

    handles = [
        model.model.layers[i].self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i in range(cfg.num_hidden_layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Group query heads into their shared KV head: (n_kv, S*group, head_dim).
    out = []
    for i in range(cfg.num_hidden_layers):
        q = captured[i]                                          # (n_heads, S, head_dim)
        grouped = q.reshape(n_kv, group, q.shape[1], head_dim)  # (n_kv, group, S, d)
        out.append(grouped.reshape(n_kv, group * q.shape[1], head_dim))
    return out
