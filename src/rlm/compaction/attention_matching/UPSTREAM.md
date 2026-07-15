# Provenance

Source: https://github.com/adamzweiger/compaction (vendored at
`external/attention-matching/`; no upstream commit hash recorded).

**Verbatim** (byte-for-byte, relative imports intact):

- `algorithms/` ← `compaction/algorithms/` — the whole directory (base, batched,
  highest_attention_keys, omp*, optim*, random_*, truncate, kvmerger). Self-contained:
  every file imports only `torch` and its siblings.

**Ours:**

- `cache_utils.py` — compacts a transformers `DynamicCache` per (layer, head).
- `task_scoring.py` — task-guided probe queries (upstream uses `query_generation/`).

**Not vendored:** `compaction_methods/` orchestration, `query_generation/`, `models/`
(`CompactedPrefixCache`, generation), eval harness, datasets, plotting, head-budget
optimization.
