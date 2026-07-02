"""Batched AM solves across layers/heads once they share a global mask.

TODO (Phase 7): replace sequential per-head/layer solves with a batched
[num_layers * num_kv_heads, kept_tokens, head_dim] solve.
"""
