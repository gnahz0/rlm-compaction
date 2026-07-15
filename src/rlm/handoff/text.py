"""TextHandoff: THIS IS WHAT BASIC RLM DOES -- pass the orchestrator's query to
the worker verbatim over the text socket. It is exactly the upstream ``llm_query``
behavior, expressed through the swappable Handoff interface, and it's the
fidelity ceiling (nothing dropped) + token-cost baseline the KV methods are
measured against.
"""

from rlm.handoff.base import Handoff, Send, SendBatched


class TextHandoff(Handoff):
    """Text seam (requires_model=False by default): pass the query verbatim."""

    def run(self, prompt: str, model: str | None, send: Send) -> str:
        return send(prompt, model)

    def run_batched(
        self, prompts: list[str], model: str | None, send_batched: SendBatched
    ) -> list[str]:
        return send_batched(prompts, model)
