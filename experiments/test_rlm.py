"""Smoke test: run a tiny text-only RLM task end-to-end (Phase 1).

A magic number is buried in a haystack of filler sentences; the RLM has to find
it via its REPL. Uses OPENAI_API_KEY from .env.

    # default: local Qwen3-4B in-process via transformers (no server, no API key)
    python experiments/test_rlm.py
    # your own prompt instead of the built-in haystack task:
    python experiments/test_rlm.py "Print the first 20 powers of 2."
    # API models (backend inferred from the model name):
    python experiments/test_rlm.py --model gpt-5-mini
    # local model behind an OpenAI-compatible server (e.g. `transformers serve`):
    python experiments/test_rlm.py --backend vllm --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B

Every run writes a JSONL trajectory (per-iteration prompt, response incl. the
model's reasoning, executed code + output, final answer) under results/logs/.
"""

import argparse

from rlm import RLM
from rlm.logger import RLMLogger

MAGIC_NUMBER = 748291


def build_context() -> str:
    filler = [f"Document {i}: nothing interesting happens in section {i % 7}." for i in range(500)]
    filler.insert(311, f"By the way, the magic number is {MAGIC_NUMBER}.")
    return "\n".join(filler)


def infer_backend(model: str) -> str:
    """HF hub ids look like 'org/model' (run in-process); bare names go to OpenAI."""
    return "hf" if "/" in model else "openai"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="?", help="run this prompt instead of the built-in magic-number task")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--backend",
        choices=["openai", "vllm", "openrouter", "anthropic", "hf"],
        help="default: inferred from --model ('org/model' -> hf, otherwise openai)",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible server URL (required for --backend vllm)")
    parser.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode (hf backend only; slower, emits <think> CoT)")
    parser.add_argument(
        "--handoff",
        default="text",
        choices=["text", "full_kv", "latent_briefing"],
        help="how the orchestrator hands context to workers ('full_kv' needs --backend hf)",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1, help="recursion depth; at max depth calls fall back to plain LM completion")
    parser.add_argument("--log-dir", default="results/logs", help="trajectory JSONL output dir ('' to disable)")
    args = parser.parse_args()

    backend = args.backend or infer_backend(args.model)
    backend_kwargs = {"model_name": args.model}
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.thinking:
        backend_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

    logger = RLMLogger(log_dir=args.log_dir) if args.log_dir else None
    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        handoff=args.handoff,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        logger=logger,
        verbose=True,
    )

    if args.prompt:
        # Pass the prompt as root_prompt too: the root LM then sees the actual
        # question directly instead of just "explore the context variable"
        # (small models describe the context instead of following it otherwise).
        result = rlm.completion(args.prompt, root_prompt=args.prompt)
    else:
        result = rlm.completion(
            build_context(),
            root_prompt="What is the magic number hidden in the context? Answer with just the number.",
        )

    print(f"\nFinal answer: {result.response}")
    print(f"Usage: {result.usage_summary.to_dict()}")
    print(f"Time: {result.execution_time:.2f}s")
    if logger and logger.log_file_path:
        print(f"Trajectory log: {logger.log_file_path}")

    if args.prompt:
        return

    ok = str(MAGIC_NUMBER) in result.response
    print("PASS" if ok else "FAIL", f"(expected {MAGIC_NUMBER})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
