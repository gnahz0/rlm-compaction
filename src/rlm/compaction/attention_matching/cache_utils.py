"""Apply AM compaction across a full transformers KV cache.

Walks every (layer, kv-head) of a prefilled cache, compacts its keys/values with
the selected attention-matching algorithm (see `algorithms/`), and returns a new
DynamicCache plus the per-key NNLS betas.

RoPE caveat: the kept keys retain the rotations they got at prefill (from their
*original* positions), so a worker generating on the compacted cache must position
its query at the ORIGINAL sequence length, not the compacted length -- i.e.
rope_base = original_seq_len - compacted_len. This function returns original_seq_len
so the caller can handle that; it does not itself patch generation positions.
"""

import torch
from transformers import DynamicCache

from rlm.compaction.attention_matching.algorithms import (
    HighestAttentionKeysCompaction,
    OMPCompaction,
)

# Preconfigured AM algorithms (nnls_iters = beta PGD refinement steps; see README).
ALGORITHMS = {
    "highest_attention_keys": HighestAttentionKeysCompaction(nnls_iters=2),
    "omp": OMPCompaction(nnls_iters=0),
}

# Algorithm compact_cache runs per (layer, head); swap to any key above.
_ALGO = ALGORITHMS["highest_attention_keys"]


def sample_probe_queries(
    n: int, head_dim: int, device, dtype, generator: torch.Generator | None = None
) -> torch.Tensor:
    """n random N(0,1) probe query vectors of size head_dim (the simple default)."""
    return torch.randn(n, head_dim, device=device, dtype=dtype, generator=generator)


def _resolve_t(target_size: int | float, seq_len: int) -> int:
    """target_size as an int count or a fraction in (0, 1] of seq_len -> int t."""
    if isinstance(target_size, float) and 0.0 < target_size <= 1.0:
        return max(1, round(target_size * seq_len))
    return min(int(target_size), seq_len)


def compact_cache(
    cache: DynamicCache,
    target_size: int | float,
    queries: list[torch.Tensor] | None = None,
    n_queries: int = 128,
    generator: torch.Generator | None = None,
) -> tuple[DynamicCache, int, list[torch.Tensor]]:
    """Top-k AM-compact every (layer, kv-head) of `cache`.

    target_size : kept entries per head -- an int count, or a fraction in (0, 1].
    queries : optional per-layer probe queries, entry `l` of shape
        (n_kv_heads, N, head_dim). Random N(0,1) probes are used where omitted;
        pass task-derived queries for task-guided compaction / LatentBriefing
        (see task_scoring.extract_task_queries).
    n_queries : number of random probes per head, when `queries` is None.

    Returns (compacted_cache, original_seq_len, betas), where betas[l] has shape
    (B, H, t): the NNLS per-key bias to add to attention over the compacted keys
    (the cache is only valid together with these betas -- see generate_on_cache).
    """
    original_seq_len = cache.get_seq_length()
    out = DynamicCache()
    betas = []

    for layer_idx, layer in enumerate(cache.layers):
        keys, values = layer.keys, layer.values          # (B, H, T, d)
        B, H, T, d = keys.shape
        t = _resolve_t(target_size, T)

        new_keys = keys.new_empty((B, H, t, d))
        new_values = values.new_empty((B, H, t, d))
        beta = keys.new_zeros((B, H, t))
        for b in range(B):
            for h in range(H):
                if queries is not None:
                    q = queries[layer_idx][h].to(device=keys.device, dtype=keys.dtype)
                else:
                    q = sample_probe_queries(n_queries, d, keys.device, keys.dtype, generator)
                c1, bta, c2, _idx = _ALGO.compute_compacted_cache(keys[b, h], values[b, h], q, t)
                new_keys[b, h], beta[b, h], new_values[b, h] = c1, bta, c2

        out.update(new_keys, new_values, layer_idx)
        betas.append(beta)

    return out, original_seq_len, betas
