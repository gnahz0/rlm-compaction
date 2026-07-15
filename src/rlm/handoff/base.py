"""Handoff interfaces: how the orchestrator's context reaches a worker.

The RLM core only talks to Handoff, never to a method's internals -- that's what
makes them swappable. Two seams under the Handoff root, dispatched on
``requires_model``:

- TextHandoff (requires_model=False): run() / run_batched(). The environment
  calls these with a ``send`` primitive -- a plain-text LM call over the socket.
  There's only one text method, so TextHandoff (text.py) is the seam itself.
- KVHandoff (requires_model=True): prepare_context() / run_worker() /
  run_workers(). Abstract base for the KV methods (FullKV, LatentBriefing) --
  they implement it independently, as peers. Needs an in-process model/tokenizer,
  since a KV cache can't cross the text socket.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# A worker call the environment hands to a text method: (prompt, model) -> response.
Send = Callable[[str, "str | None"], str]
SendBatched = Callable[["list[str]", "str | None"], "list[str]"]


class Handoff(ABC):
    """Base for every handoff -- a swappable strategy for passing the
    orchestrator's context to a worker. The environment dispatches on
    ``requires_model``; KV-seam methods set it True (see KVHandoff)."""

    requires_model: bool = False


class KVHandoff(Handoff):
    """KV seam: the worker gets the context as a precomputed (and, for compaction
    methods, shrunk) KV-cache prefix, then attends over it while processing its
    own query tokens. Two-phase interface:

        prepare_context(context) -> prepared   # ONCE: prefill -> KV cache
        run_worker(prepared, query) -> str      # PER worker: append + generate

    Concretes (FullKVHandoff, LatentBriefingHandoff) implement these
    independently -- they are peers, not subclasses of one another. The
    shared worker config lives in __init__ here; the shared plumbing (prefill,
    query building, generation) lives in kv_common as free functions.
    """

    requires_model = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        system_prompt: str = "",
        chat_template_kwargs: "dict[str, Any] | None" = None,
        max_new_tokens: int = 4096,
        sampling_args: "dict[str, Any] | None" = None,
    ):
        """model / tokenizer: in-process HF model and tokenizer (hf backend).
        system_prompt: baked into the shared cached prefix. chat_template_kwargs:
        apply_chat_template kwargs (default enable_thinking=True, matching the
        text worker). max_new_tokens / sampling_args: worker generation config.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.chat_template_kwargs = (
            {"enable_thinking": True} if chat_template_kwargs is None else chat_template_kwargs
        )
        self.max_new_tokens = max_new_tokens
        self.sampling_args = sampling_args or {}

    @abstractmethod
    def prepare_context(self, context: str) -> Any:
        """Prefill ``context`` once and return a prepared object holding its KV
        cache (opaque to callers; consumed by run_worker)."""
        raise NotImplementedError

    @abstractmethod
    def run_worker(self, prepared: Any, query: str) -> str:
        """Answer ``query`` using ``prepared`` as the cached context prefix."""
        raise NotImplementedError

    @abstractmethod
    def run_workers(self, prepared: Any, queries: list[str]) -> list[str]:
        """Fan-out: many workers share one prepared context, responses in order."""
        raise NotImplementedError
