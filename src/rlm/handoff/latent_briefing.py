"""LatentBriefingHandoff: task-guided Attention Matching KV compaction.

Compacts the context's KV cache with top-k Attention Matching, scoring keys by the
query vectors the model produces when reading the worker's task prompt (rather than
generic/random probes). This keeps the KV entries most relevant to *this* task.
Because the compaction depends on the query, it happens PER WORKER (in run_worker),
not once in prepare_context: the context is prefilled once and shared, and each
worker compacts its own view.

TODO (Phases 5-8): shared global mask, MAD thresholding, batched solves.
"""

from rlm.compaction.attention_matching import compact_cache, extract_task_queries
from rlm.handoff.base import KVHandoff
from rlm.handoff.kv_common import (
    PreparedContext,
    build_query_ids,
    format_context_prefix,
    generate_on_cache,
    prefill,
)


class LatentBriefingHandoff(KVHandoff):
    def __init__(self, model, tokenizer, target_size: float | int = 0.1, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_size = target_size  # kept fraction (0,1] or int count per head

    def prepare_context(self, context: str) -> PreparedContext:
        """Prefill the FULL cache once; compaction is deferred to run_worker
        because it depends on the (per-worker) task query."""
        cache, n = prefill(self.model, self.tokenizer, format_context_prefix(self.system_prompt, context))
        return PreparedContext(cache=cache, orig_len=n)

    def run_worker(self, prepared: PreparedContext, query: str) -> str:
        thinking = self.chat_template_kwargs.get("enable_thinking", False)
        query_ids = build_query_ids(self.tokenizer, self.model, query, thinking)
        # Task-guided compaction: probe with the worker's own query vectors.
        task_queries = extract_task_queries(self.model, query_ids)
        cache, orig_len, betas = compact_cache(prepared.cache, self.target_size, queries=task_queries)
        return generate_on_cache(
            self.model, self.tokenizer, cache, orig_len,
            query_ids, self.max_new_tokens, self.sampling_args, betas=betas,
        )

    def run_workers(self, prepared: PreparedContext, queries: list[str]) -> list[str]:
        return [self.run_worker(prepared, q) for q in queries]
