"""handoff: swappable methods for passing context from orchestrator to worker.

Methods (see reproduction plan for phase ordering):
- text.TextHandoff             -- baseline RLM (raw text context)
- summary.SummaryHandoff       -- text-summarization baseline
- vanilla_am.VanillaAMHandoff  -- unmodified Attention Matching KV compaction
- latent_briefing.LatentBriefingHandoff -- Ramp-style task-guided KV briefing
"""
