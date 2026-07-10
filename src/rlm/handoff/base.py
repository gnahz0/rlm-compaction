"""Common interface all handoff methods must implement.

The RLM core only ever talks to this interface, never to compaction/
handoff internals directly -- that's what makes methods swappable.

A handoff decides *how* the orchestrator's query for a worker is delivered:
verbatim (TextHandoff), text-compressed (SummaryHandoff), or -- in future --
as a compacted KV cache (VanillaAM/LatentBriefing).

Text-space methods operate purely on strings and reach the worker through a
``send`` primitive the environment provides: a plain-text LM call over the LM
handler socket. KV-space methods will additionally need in-process
model/tokenizer access and will extend this interface (a compacted KV cache
cannot cross the text socket); see the reproduction plan.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

# A worker call the environment hands to the method: (prompt, model) -> response.
Send = Callable[[str, "str | None"], str]
SendBatched = Callable[["list[str]", "str | None"], "list[str]"]


class HandoffMethod(ABC):
    @abstractmethod
    def run(self, prompt: str, model: str | None, send: Send) -> str:
        """Deliver a single worker query and return the worker's response.

        Args:
            prompt: The query the orchestrator assembled for the worker.
            model: Optional worker model override (None = handler default).
            send: Primitive that runs one plain-text worker completion.
        """
        raise NotImplementedError

    @abstractmethod
    def run_batched(
        self, prompts: list[str], model: str | None, send_batched: SendBatched
    ) -> list[str]:
        """Deliver many worker queries, returning responses in input order."""
        raise NotImplementedError
