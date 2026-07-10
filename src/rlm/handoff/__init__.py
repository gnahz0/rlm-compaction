"""handoff: swappable methods for passing context from orchestrator to worker.

Methods (see reproduction plan for phase ordering):
- text.TextHandoff             -- baseline RLM (raw text context)
- summary.SummaryHandoff       -- text-summarization baseline
- vanilla_am.VanillaAMHandoff  -- unmodified Attention Matching KV compaction
- latent_briefing.LatentBriefingHandoff -- Ramp-style task-guided KV briefing

``get_handoff`` maps the string selectors used in RLM settings/configs to
instances. Only the text-space methods (text, summary) are implemented; the KV
methods are still stubs.
"""

from typing import Any, Literal

from rlm.handoff.base import HandoffMethod
from rlm.handoff.summary import SummaryHandoff
from rlm.handoff.text import TextHandoff

HandoffType = Literal["text", "summary"]

_HANDOFFS: dict[str, type[HandoffMethod]] = {
    "text": TextHandoff,
    "summary": SummaryHandoff,
}

__all__ = [
    "HandoffMethod",
    "HandoffType",
    "TextHandoff",
    "SummaryHandoff",
    "get_handoff",
]


def get_handoff(
    handoff: "HandoffType | HandoffMethod",
    handoff_kwargs: dict[str, Any] | None = None,
) -> HandoffMethod:
    """Resolve a handoff selector to an instance.

    Accepts a string ("text" | "summary") or an already-built HandoffMethod
    (returned as-is, ignoring handoff_kwargs).
    """
    if isinstance(handoff, HandoffMethod):
        return handoff
    if handoff not in _HANDOFFS:
        raise ValueError(
            f"Unknown handoff method: {handoff!r}. Supported: {sorted(_HANDOFFS)}."
        )
    return _HANDOFFS[handoff](**(handoff_kwargs or {}))
