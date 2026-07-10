"""FullKVHandoff: no-compaction KV handoff -- prefill the context through the
model ONCE, capture its full KV cache, and hand that cache to each worker as a
precomputed prefix. The worker's query tokens are appended fresh and attend
back over the cached context.

What this is (and isn't):
- It is the *plumbing + equivalence baseline* for all KV handoffs. With no
  compaction, a worker attending over the full context KV is numerically
  identical to a worker that received the full context as text and prefilled it
  itself -- same KV tensors, same memory, same output. So this does NOT save
  worker memory/tokens; it only skips re-prefill compute and gives us a
  correctness anchor: its greedy output should MATCH TextHandoff on the same
  input. Compaction (VanillaAM / LatentBriefing) is what later shrinks the
  cache; the only thing that changes is the KV gets smaller.

Hard constraints (why this can't use the text `send` seam):
- Same model, in-process. A KV cache is tied to specific weights (layers, head
  dims, RoPE positions, dtype, device); producer and worker must be the SAME
  loaded model object. It cannot cross the LM-handler socket -- these are GPU
  tensors, not JSON. So this handoff needs a direct model/tokenizer handle
  (the `hf` backend), which is why the text-seam methods below raise.

NOT WIRED YET: this class is not registered in `handoff.get_handoff` and the
REPL seam (`local_repl._llm_query`) still passes a single text string. Wiring
requires exposing a (context, query) split to the seam -- a separate change.
This file is a template to build against.
"""

from dataclasses import dataclass
from typing import Any

from rlm.handoff.base import HandoffMethod


@dataclass
class PreparedContext:
    """The handoff object produced from a context and consumed by workers.

    Attributes:
        cache: the captured KV cache (transformers `Cache`/`DynamicCache`),
            representing `context_len` prefilled tokens.
        context_len: number of tokens the cache covers (needed to position the
            worker's query tokens after the prefix and to size the attn mask).
        prefix_ids: the context token ids that produced `cache`. Kept for
            debugging / equivalence checks (compare prefill vs full forward).
    """

    cache: Any
    context_len: int
    prefix_ids: Any


class FullKVHandoff(HandoffMethod):
    """Args:
    model: the loaded HF causal LM (e.g. HuggingFaceClient.model).
    tokenizer: its tokenizer.
    chat_template_kwargs: passed to apply_chat_template (e.g.
        {"enable_thinking": False} for Qwen3).
    max_new_tokens / sampling_args: worker generation config.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 4096,
        sampling_args: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_kwargs = (
            {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        )
        self.max_new_tokens = max_new_tokens
        self.sampling_args = sampling_args or {}

    # ------------------------------------------------------------------ #
    # KV interface: the real entry points.                               #
    # prepare_context() runs once; run_worker(s)() runs per worker.      #
    # ------------------------------------------------------------------ #

    def prepare_context(self, context: str) -> PreparedContext:
        """Prefill `context` through the model and capture its full KV cache."""
        import torch

        # TODO(template): chat-template boundary.
        # The cache must end at a point where a worker query + generation prompt
        # can legally follow. Simplest split: cache the user turn's context
        # prefix, append "{query}<|im_end|><assistant>" per worker in
        # _build_query_ids(). Verify the invariant:
        #     tokenize(prefix) ++ tokenize(suffix) == tokenize(prefix+suffix)
        # (no merge across the boundary). Safest check: compare this cache's
        # generation against a single full forward on prefix+suffix.
        prefix_text = self._format_context_prefix(context)
        enc = self.tokenizer(prefix_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=True,
            )

        return PreparedContext(
            cache=out.past_key_values,
            context_len=enc["input_ids"].shape[1],
            prefix_ids=enc["input_ids"],
        )

    def run_worker(self, prepared: PreparedContext, query: str) -> str:
        """Generate an answer to `query` using `prepared` as the KV prefix."""
        # IMPORTANT: clone the cache. generate() appends the query's + generated
        # tokens' KV into the cache object in place; reusing one PreparedContext
        # across workers without cloning corrupts it. See _clone_cache.
        cache = self._clone_cache(prepared.cache)
        return self._generate_with_prefix(cache, prepared.context_len, query)

    def run_workers(self, prepared: PreparedContext, queries: list[str]) -> list[str]:
        """Fan-out: many workers share ONE cached context (clone per worker).

        TODO(template): parallelize. model.generate isn't thread-safe (see the
        HuggingFaceClient _generate_lock), so true concurrency needs batched
        generation (pad queries, one generate call over a batch that shares the
        expanded prefix) rather than threads. For the baseline, sequential is
        fine and correct:
        """
        
        return [self.run_worker(prepared, q) for q in queries]

    # ------------------------------------------------------------------ #
    # Helpers -- the mechanically tricky bits.                           #
    # ------------------------------------------------------------------ #

    def _format_context_prefix(self, context: str) -> str:
        """Render the context as the cached prefix string.

        TODO(template): decide the exact template split. A starting point that
        keeps the context inside the user turn so a query can follow:
        """
        # Placeholder: treat context as the start of a user turn, no generation
        # prompt yet (the query + generation prompt are appended per worker).
        # Replace with a template-correct split for your model.
        return context

    def _build_query_ids(self, query: str):
        """Tokenize the worker's query + generation prompt (the suffix that
        follows the cached context).

        TODO(template): must match the template used in _format_context_prefix
        so prefix+suffix forms one valid conversation with add_generation_prompt.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs,
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)["input_ids"]

    def _clone_cache(self, cache: Any) -> Any:
        """Deep-copy a KV cache so per-worker generation can't mutate the shared
        prefix.

        TODO(template): implement for your transformers version. Modern
        DynamicCache exposes key_cache/value_cache lists of tensors; build a new
        cache with per-tensor .clone(). copy.deepcopy(cache) also works but is
        slower. Verify: generating twice from the same PreparedContext yields
        identical output (proves no in-place corruption).
        """
        raise NotImplementedError("_clone_cache: implement per transformers version")

    def _generate_with_prefix(self, cache: Any, context_len: int, query: str) -> str:
        """Generate the worker's answer with `cache` as the KV prefix."""
        import torch

        query_ids = self._build_query_ids(query)
        query_len = query_ids.shape[1]

        # Attention mask must span the cached prefix + the new query tokens, so
        # the query attends back over the whole context.
        attention_mask = torch.ones(
            (1, context_len + query_len), device=self.model.device, dtype=torch.long
        )

        # TODO(template): positions. Passing past_key_values + input_ids +
        # a full-length attention_mask lets transformers infer cache_position
        # for most models; if Qwen3 mispositions (garbage output), pass
        # cache_position=arange(context_len, context_len+query_len) explicitly.
        sampling = dict(self.sampling_args)
        max_new_tokens = sampling.pop("max_tokens", None) or self.max_new_tokens

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=query_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
            )

        # generate returns [query_ids ++ generated]; the cached context is not
        # in input_ids, so slice off the query portion to get only new tokens.
        completion_ids = output_ids[0, query_len:]
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # Qwen3 may emit a think block before the answer.
        return response.split("</think>", 1)[-1].strip()

    # ------------------------------------------------------------------ #
    # Text-seam methods: not applicable to KV handoff.                   #
    # ------------------------------------------------------------------ #

    def run(self, prompt, model, send):
        raise NotImplementedError(
            "FullKVHandoff uses the in-process KV interface (prepare_context / "
            "run_worker), not the text `send` seam. It needs a direct "
            "model/tokenizer handle and can't route over the LM-handler socket."
        )

    def run_batched(self, prompts, model, send_batched):
        raise NotImplementedError(
            "FullKVHandoff uses run_workers(prepared, queries), not the text "
            "`send_batched` seam."
        )
