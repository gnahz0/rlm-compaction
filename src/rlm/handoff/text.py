"""TextHandoff: baseline RLM handoff -- passes the orchestrator's query to the
worker verbatim. This is exactly the upstream ``llm_query`` behavior, now
expressed through the swappable HandoffMethod interface. It is the fidelity
ceiling (nothing is dropped) and the token-cost baseline the compression
methods are measured against.
"""

from rlm.handoff.base import HandoffMethod, Send, SendBatched


class TextHandoff(HandoffMethod):
    def run(self, prompt: str, model: str | None, send: Send) -> str:
        return send(prompt, model)

    def run_batched(
        self, prompts: list[str], model: str | None, send_batched: SendBatched
    ) -> list[str]:
        return send_batched(prompts, model)
