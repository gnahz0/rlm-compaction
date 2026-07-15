"""handoff: swappable methods for passing context from orchestrator to worker.

Methods (all subclass Handoff; see base.py for the text vs KV seams):
- text.TextHandoff             -- baseline RLM: raw text context, verbatim
- full_kv.FullKVHandoff        -- no-compaction KV handoff (equivalence baseline)
- latent_briefing.LatentBriefingHandoff -- task-guided Attention Matching KV briefing
"""

from typing import Any, Literal

from rlm.handoff.base import Handoff, KVHandoff
from rlm.handoff.full_kv import FullKVHandoff
from rlm.handoff.latent_briefing import LatentBriefingHandoff
from rlm.handoff.text import TextHandoff

HandoffType = Literal["text", "full_kv", "latent_briefing"]

_HANDOFFS: dict[str, type[Handoff]] = {
    "text": TextHandoff,
    "full_kv": FullKVHandoff,
    "latent_briefing": LatentBriefingHandoff,
}

__all__ = [
    "Handoff",
    "KVHandoff",
    "HandoffType",
    "TextHandoff",
    "FullKVHandoff",
    "LatentBriefingHandoff",
    "get_handoff",
    "is_kv_handoff",
]


def is_kv_handoff(handoff: "HandoffType | Handoff") -> bool:
    """True if the selector names / is a KV-seam handoff (needs model access)."""
    if isinstance(handoff, Handoff):
        return handoff.requires_model
    return isinstance(handoff, str) and handoff in _HANDOFFS and _HANDOFFS[handoff].requires_model


def get_handoff(
    handoff: "HandoffType | Handoff",
    handoff_kwargs: dict[str, Any] | None = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
) -> Handoff:
    """Resolve a handoff selector to an instance.

    Accepts an already-built Handoff (returned as-is) or a name ("text",
    "full_kv"). KV-seam methods (requires_model) additionally need ``model`` and
    ``tokenizer`` -- an in-process handle from the hf backend, since a KV cache
    can't cross the text socket.
    """
    if isinstance(handoff, Handoff):
        return handoff
    if handoff not in _HANDOFFS:
        raise ValueError(f"Unknown handoff method: {handoff!r}. Supported: {sorted(_HANDOFFS)}.")

    cls = _HANDOFFS[handoff]
    kwargs = dict(handoff_kwargs or {})
    if cls.requires_model:
        if model is None or tokenizer is None:
            raise ValueError(
                f"{handoff!r} is a KV handoff: it needs an in-process model/tokenizer "
                "(the 'hf' backend). Pass model= and tokenizer= to get_handoff."
            )
        kwargs.update(model=model, tokenizer=tokenizer)
    return cls(**kwargs)
