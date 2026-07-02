"""LatentBriefingHandoff: Ramp-style handoff -- task-guided token scoring +
shared global mask + AM compaction -> compacted KV briefing -> worker.

TODO (Phases 4-8): task-guided scoring, shared global mask, MAD thresholding,
batched solves, prefix-cache reuse.
"""

from rlm.handoff.base import HandoffMethod


class LatentBriefingHandoff(HandoffMethod):
    pass
