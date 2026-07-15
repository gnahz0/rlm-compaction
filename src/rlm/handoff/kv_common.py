"""Shared KV-handoff plumbing as free functions.

The KV handoffs (FullKV, LatentBriefing) are peers -- none subclasses another --
so the machinery they share (prefill a context, build the query suffix,
generate on a possibly-compacted cache) lives here as plain functions they call,
rather than in a common base class.

Generation handles the compaction rope_base offset: a compacted cache keeps its
kept keys' original RoPE rotations, so the worker's query must be positioned at
the ORIGINAL context length, not the (smaller) cache length. offset = orig_len -
cache_len; for an uncompacted cache offset is 0.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PreparedContext:
    """A prefilled context handed to workers.

    cache : the KV cache (possibly compacted by a compaction handoff).
    orig_len : the ORIGINAL context length, used to position the worker's query
        for RoPE (offset = orig_len - cache_len; 0 when uncompacted).
    """

    cache: Any
    orig_len: int


def format_context_prefix(system_prompt: str, context: str) -> str:
    """Shared cached prefix: [system turn] + an OPEN user turn holding context."""
    prefix = ""
    if system_prompt:
        prefix += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    return prefix + f"<|im_start|>user\n{context}"


def build_query_ids(tokenizer, model, query: str, thinking: bool):
    """Per-worker suffix: the query, the close of the user turn, and the assistant
    generation prompt (empty think block when not in thinking mode)."""
    gen_prompt = "" if thinking else "<think>\n\n</think>\n\n"
    suffix = f"\n\n{query}<|im_end|>\n<|im_start|>assistant\n{gen_prompt}"
    return tokenizer(suffix, return_tensors="pt").to(model.device)["input_ids"]


def prefill(model, tokenizer, text: str):
    """Prefill `text` and return (full KV cache, token count)."""
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
    return out.past_key_values, enc["input_ids"].shape[1]


def clone_cache(cache: Any) -> Any:
    """Deep-copy a cache so per-worker generation can't mutate a shared prefix."""
    return copy.deepcopy(cache)


def _sample_next(logits: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Sample one token id from (1, vocab) logits per cfg (do_sample/temperature/
    top_k/top_p). Greedy if do_sample is False."""
    if not cfg.get("do_sample", True):
        return logits.argmax(-1, keepdim=True)
    logits = logits / (cfg.get("temperature") or 1.0)
    top_k = cfg.get("top_k")
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    top_p = cfg.get("top_p")
    if top_p and top_p < 1.0:
        sp, idx = torch.sort(probs, descending=True, dim=-1)
        drop = (torch.cumsum(sp, dim=-1) - sp) > top_p
        probs = torch.zeros_like(probs).scatter(-1, idx, sp.masked_fill(drop, 0.0))
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _make_beta_hook(beta_layer: torch.Tensor, group: int):
    """Forward pre-hook that adds the NNLS beta to a layer's attention logits over
    the compacted-key columns. beta_layer (B, n_kv, t) -> per query head (GQA)."""
    beta_q = beta_layer.repeat_interleave(group, dim=1)   # (B, n_heads, t)
    t = beta_q.shape[-1]

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.dim() != 4:
            return None
        b, _, ql, kv = mask.shape
        mask = mask.expand(b, beta_q.shape[1], ql, kv).clone().to(torch.float32)
        mask[:, :, :, :t] = mask[:, :, :, :t] + beta_q[:, :, None, :].to(torch.float32)
        kwargs["attention_mask"] = mask
        return args, kwargs

    return hook


def generate_on_cache(
    model, tokenizer, cache: Any, orig_len: int, query_ids, max_new_tokens: int,
    sampling_args: dict, betas: "list[torch.Tensor] | None" = None,
) -> str:
    """Decode the worker's answer on `cache` as the KV prefix.

    orig_len positions the query for RoPE (offset = orig_len - cache_len), so a
    compacted cache's kept keys line up. If `betas` is given (from a compaction
    that produced NNLS biases), they are injected into each layer's attention over
    the compacted keys via hooks (eager attention). Sampling defaults to the
    model's generation_config, overridden by sampling_args.
    """
    gc = model.generation_config
    cfg = {
        "do_sample": gc.do_sample,
        "temperature": gc.temperature,
        "top_k": gc.top_k,
        "top_p": gc.top_p,
        **sampling_args,
    }
    cache_len = cache.get_seq_length()
    offset = orig_len - cache_len

    # beta injection needs an explicit additive mask -> force eager attention and
    # add beta to each layer's attention logits via a forward pre-hook.
    handles, prev_impl = [], None
    if betas is not None:
        prev_impl = model.config._attn_implementation
        model.config._attn_implementation = "eager"
        group = model.config.num_attention_heads // model.config.num_key_value_heads
        handles = [
            model.model.layers[i].self_attn.register_forward_pre_hook(
                _make_beta_hook(betas[i], group), with_kwargs=True
            )
            for i in range(len(betas))
        ]

    cur = query_ids
    generated = []
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                q_len = cur.shape[1]
                cache_position = torch.arange(cache_len, cache_len + q_len, device=model.device)
                position_ids = (cache_position + offset).unsqueeze(0)
                attn = torch.ones((1, cache_len + q_len), device=model.device, dtype=torch.long)
                out = model(
                    input_ids=cur,
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    attention_mask=attn,
                )
                cache = out.past_key_values
                cache_len += q_len
                nxt = _sample_next(out.logits[:, -1, :], cfg)
                generated.append(nxt)
                if nxt.item() == tokenizer.eos_token_id:
                    break
                cur = nxt
    finally:
        for h in handles:
            h.remove()
        if prev_impl is not None:
            model.config._attn_implementation = prev_impl

    text = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
    return text.split("</think>", 1)[-1].strip()
