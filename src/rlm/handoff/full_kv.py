"""FullKVHandoff: no-compaction KV handoff -- prefill the context through the
model ONCE, capture its full KV cache, and hand that cache to each worker as a
precomputed prefix. The worker's query tokens are appended fresh and attend back
over the cached context.

This is the plumbing + equivalence baseline for all KV handoffs: with no
compaction, a worker attending over the full context KV is numerically identical
to one that received the context as text and prefilled it itself. Compaction
methods (LatentBriefing) later shrink the cache; the shared prefill / query /
generation plumbing lives in kv_common.
"""

from rlm.handoff.base import KVHandoff
from rlm.handoff.kv_common import (
    PreparedContext,
    build_query_ids,
    clone_cache,
    format_context_prefix,
    generate_on_cache,
    prefill,
)


class FullKVHandoff(KVHandoff):
    def prepare_context(self, context: str) -> PreparedContext:
        """Prefill `context` and capture its full KV cache."""
        cache, n = prefill(self.model, self.tokenizer, format_context_prefix(self.system_prompt, context))
        return PreparedContext(cache=cache, orig_len=n)

    def run_worker(self, prepared: PreparedContext, query: str) -> str:
        """Generate an answer to `query` using `prepared` as the KV prefix."""
        thinking = self.chat_template_kwargs.get("enable_thinking", False)
        query_ids = build_query_ids(self.tokenizer, self.model, query, thinking)
        # Clone: generation appends to the cache in place, so a shared prepared
        # cache must not be mutated across workers.
        cache = clone_cache(prepared.cache)
        return generate_on_cache(
            self.model, self.tokenizer, cache, prepared.orig_len,
            query_ids, self.max_new_tokens, self.sampling_args,
        )

    def run_workers(self, prepared: PreparedContext, queries: list[str]) -> list[str]:
        """Fan-out: many workers share one prepared context (cloned per worker)."""
        return [self.run_worker(prepared, q) for q in queries]
