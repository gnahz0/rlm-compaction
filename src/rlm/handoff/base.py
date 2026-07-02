"""Common interface all handoff methods must implement.

The RLM core only ever talks to this interface, never to compaction/
handoff internals directly -- that's what makes methods swappable.
"""


class HandoffMethod:
    def prepare(self, orchestrator_trace, worker_query, model, tokenizer):
        """Prepare whatever the worker needs: raw text, summary text, compressed
        KV cache, selected tokens, or a hybrid text + KV representation.
        """
        raise NotImplementedError

    def call_worker(self, prepared_context, worker_query, model, tokenizer):
        """Call the worker using the prepared context."""
        raise NotImplementedError
