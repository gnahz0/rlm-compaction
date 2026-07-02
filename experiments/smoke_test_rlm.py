"""Smoke test: run a tiny text-only RLM task end-to-end (Phase 1).

A magic number is buried in a haystack of filler sentences; the RLM has to find
it via its REPL. Uses OPENAI_API_KEY from .env.

    python experiments/smoke_test_rlm.py
    RLM_MODEL=gpt-5 python experiments/smoke_test_rlm.py
"""

import os

from rlm import RLM

MAGIC_NUMBER = 748291


def build_context() -> str:
    filler = [f"Document {i}: nothing interesting happens in section {i % 7}." for i in range(500)]
    filler.insert(311, f"By the way, the magic number is {MAGIC_NUMBER}.")
    return "\n".join(filler)


def main() -> None:
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": os.getenv("RLM_MODEL", "gpt-5-mini")},
        environment="local",
        max_iterations=8,
        verbose=True,
    )
    result = rlm.completion(
        build_context(),
        root_prompt="What is the magic number hidden in the context? Answer with just the number.",
    )

    print(f"\nFinal answer: {result.response}")
    print(f"Usage: {result.usage_summary.to_dict()}")
    print(f"Time: {result.execution_time:.2f}s")

    ok = str(MAGIC_NUMBER) in result.response
    print("PASS" if ok else "FAIL", f"(expected {MAGIC_NUMBER})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
