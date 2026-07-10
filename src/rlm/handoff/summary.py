"""SummaryHandoff: text-space compression baseline. Compresses the
orchestrator's query with one LM summarization pass, then hands the *summary*
to the worker. This is the naive alternative to KV compaction -- it saves
worker tokens by discarding information at the word level.

Semantics note: at the ``llm_query`` seam the orchestrator's query is a single
string that mixes context and instruction, so the whole thing is summarized.
That is the intended baseline behavior -- it answers "if you just want fewer
tokens, why not summarize?" and is the control the latent (KV) methods must
beat on quality-per-token.
"""

from rlm.handoff.base import HandoffMethod, Send, SendBatched

DEFAULT_INSTRUCTION = (
    "Summarize the following as concisely as possible while preserving every "
    "detail, name, number, and instruction that could be needed to act on it. "
    "Output only the summary.\n\n"
)


class SummaryHandoff(HandoffMethod):
    """Args:
    instruction: Prefix prepended to the text being summarized.
    summary_model: Optional model for the summarization pass (None = same
        model as the worker call / handler default).
    """

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION, summary_model: str | None = None):
        self.instruction = instruction
        self.summary_model = summary_model

    def run(self, prompt: str, model: str | None, send: Send) -> str:
        summary = send(self.instruction + prompt, self.summary_model or model)
        # If summarization itself failed, don't send a garbage prompt onward.
        if summary.startswith("Error:"):
            return summary
        return send(summary, model)

    def run_batched(
        self, prompts: list[str], model: str | None, send_batched: SendBatched
    ) -> list[str]:
        summaries = send_batched(
            [self.instruction + p for p in prompts], self.summary_model or model
        )
        # Failed summaries are already "Error: ..." strings; sending them back
        # as prompts would waste a call, so pass them straight through.
        passthrough = {i for i, s in enumerate(summaries) if s.startswith("Error:")}
        to_send = [s for i, s in enumerate(summaries) if i not in passthrough]
        answers = send_batched(to_send, model) if to_send else []
        out, it = [], iter(answers)
        for i, s in enumerate(summaries):
            out.append(s if i in passthrough else next(it))
        return out
