# attention_matching

Attention Matching (AM) KV-cache compaction, used by the KV handoffs
(`rlm/handoff/latent_briefing.py`). AM builds a smaller set of keys `C1`, per-key
biases `beta`, and values `C2` that reproduce a head's attention output
`softmax(qKᵀ/√d)V` over a set of probe queries.

Reference: **Fast KV Compaction via Attention Matching**, Zweiger, Fu, Guo, Kim,
2026 ([arXiv:2602.16284](https://arxiv.org/abs/2602.16284),
[code](https://github.com/adamzweiger/compaction)).

## Layout

| Path | Source |
| --- | --- |
| `algorithms/` | Verbatim from upstream `compaction/algorithms/` — every AM method. |
| `cache_utils.py` | Ours. Compacts a transformers `DynamicCache` per (layer, head). |
| `task_scoring.py` | Ours. Task-guided probe queries for LatentBriefing. |

See `UPSTREAM.md` for exact provenance.

## Usage

```python
from rlm.compaction.attention_matching import compact_cache, extract_task_queries

queries = extract_task_queries(model, task_ids)          # per-layer task probes
cache, orig_len, betas = compact_cache(full_cache, target_size=0.1, queries=queries)
```

`compact_cache` returns the compacted cache, the original sequence length (for the
RoPE offset at generation), and per-layer NNLS `betas` (the cache is only valid
together with these — see `handoff/kv_common.generate_on_cache`).

## Configuration

The active algorithm is `_ALGO` in `cache_utils.py`; preconfigured methods live in
the `ALGORITHMS` dict, and all variants are in `algorithms.ALGORITHM_REGISTRY`.
Default is Highest-Attention-Keys, matching the paper's defaults:

| Knob | Value |
| --- | --- |
| key selection | top-`t` by peak attention (`score_method='max'`) |
| `beta` | NNLS via projected-gradient descent (`nnls_iters=2`) |
| `C2` | least squares, no ridge (`c2_solver='lstsq'`, `ridge_lambda=0`) |

`lstsq` follows the paper's finding that it gives the best `C2` quality; ridge/pinv/
cholesky remain as numerical fallbacks. To change method or hyperparameters, edit
`_ALGO` (e.g. `HighestAttentionKeysCompaction(nnls_iters=5)` or
`ALGORITHM_REGISTRY['omp']()`).
