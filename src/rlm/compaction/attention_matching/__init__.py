"""Attention Matching KV-cache compaction. See README.md and UPSTREAM.md."""

from rlm.compaction.attention_matching.algorithms import (
    ALGORITHM_REGISTRY,
    CompactionAlgorithm,
    HighestAttentionKeysCompaction,
    evaluate_compaction,
)
from rlm.compaction.attention_matching.cache_utils import compact_cache, sample_probe_queries
from rlm.compaction.attention_matching.task_scoring import extract_task_queries

__all__ = [
    "ALGORITHM_REGISTRY",
    "CompactionAlgorithm",
    "HighestAttentionKeysCompaction",
    "evaluate_compaction",
    "compact_cache",
    "sample_probe_queries",
    "extract_task_queries",
]
