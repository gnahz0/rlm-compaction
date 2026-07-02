"""VanillaAMHandoff: prefill -> full KV cache -> unmodified Attention Matching
compaction -> worker. Proves KV plumbing works before adding Ramp-style
modifications (task-guided scoring, shared global mask, batched solves, ...).

TODO (Phase 3): implement using rlm.compaction.attention_matching.
"""

from rlm.handoff.base import HandoffMethod


class VanillaAMHandoff(HandoffMethod):
    pass
