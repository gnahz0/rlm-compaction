"""Run lightweight LongBench-v2 multiple-choice evals.

This is intentionally small and repo-native. It supports:
- direct LM calls through rlm.clients
- optional full RLM calls through rlm.RLM
- safe context truncation for local smoke tests
- JSONL predictions for later analysis
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rlm import RLM
from rlm.clients import get_client


DEFAULT_DATASET = "THUDM/LongBench-v2"
DEFAULT_SCRATCH_ROOT = Path("/orcd/scratch/orcd/010/alecz/rlm-compaction")
DEFAULT_CACHE_DIR = DEFAULT_SCRATCH_ROOT / "hf-datasets"
CHOICE_RE = re.compile(r"(?:^|\b)(?:answer\s*[:\-]?\s*)?([ABCD])(?:\b|$)", re.IGNORECASE)


def infer_backend(model: str) -> str:
    """HF hub ids look like 'org/model'; bare names go to OpenAI."""
    return "hf" if "/" in model else "openai"


def parse_filter(value: str | None, valid: set[str], name: str) -> set[str] | None:
    """Parse a comma-separated filter (e.g. 'short,medium') into a set.

    None/empty or a list containing 'all' means no filter (returns None).
    """
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if "all" in parts:
        return None
    bad = [p for p in parts if p not in valid]
    if bad:
        raise SystemExit(f"Invalid --{name} value(s): {bad}; choose from {sorted(valid)} or 'all'")
    return set(parts)


def truncate_context(context: str, max_chars: int | None, strategy: str) -> str:
    if max_chars is None or max_chars <= 0 or len(context) <= max_chars:
        return context

    marker = "\n\n[... context truncated ...]\n\n"
    if max_chars <= len(marker):
        return context[:max_chars]

    budget = max_chars - len(marker)
    if strategy == "head":
        return context[:max_chars]
    if strategy == "tail":
        return context[-max_chars:]
    if strategy == "middle":
        start = max(0, (len(context) - max_chars) // 2)
        return context[start : start + max_chars]
    if strategy == "head-tail":
        head = budget // 2
        tail = budget - head
        return context[:head] + marker + context[-tail:]
    raise ValueError(f"Unknown truncation strategy: {strategy}")


def format_prompt(example: dict[str, Any], max_context_chars: int | None, context_strategy: str) -> str:
    context = truncate_context(str(example["context"]), max_context_chars, context_strategy)
    return (
        "Read the context and answer the multiple-choice question.\n"
        "Return only one letter: A, B, C, or D.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{example['question']}\n\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}\n\n"
        "Answer:"
    )


def extract_choice(text: str) -> str | None:
    stripped = text.strip()
    if stripped[:1].upper() in {"A", "B", "C", "D"}:
        return stripped[:1].upper()

    matches = CHOICE_RE.findall(stripped)
    if matches:
        return matches[-1].upper()
    return None


def make_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    backend_kwargs: dict[str, Any] = {
        "model_name": args.model,
        "sampling_args": {
            "temperature": args.temperature,
            "max_tokens": args.max_new_tokens,
        },
    }
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.backend == "hf":
        backend_kwargs["max_new_tokens"] = args.max_new_tokens
        backend_kwargs["chat_template_kwargs"] = {"enable_thinking": args.thinking}
    return backend_kwargs


def iter_examples(dataset, args: argparse.Namespace):
    seen = 0
    skipped = 0
    for idx, example in enumerate(dataset):
        if idx < args.start:
            continue
        if args.domain and example.get("domain") != args.domain:
            skipped += 1
            continue
        if args.sub_domain and example.get("sub_domain") != args.sub_domain:
            skipped += 1
            continue
        if args.difficulty and example.get("difficulty") not in args.difficulty:
            skipped += 1
            continue
        if args.length and example.get("length") not in args.length:
            skipped += 1
            continue
        yield idx, example
        seen += 1
        if args.limit and seen >= args.limit:
            break


def run_direct(args: argparse.Namespace, prompt: str) -> str:
    if not hasattr(run_direct, "_client"):
        run_direct._client = get_client(args.backend, make_backend_kwargs(args))  # type: ignore[attr-defined]
    return run_direct._client.completion(prompt)  # type: ignore[attr-defined]


def run_rlm(args: argparse.Namespace, example: dict[str, Any]) -> str:
    root_prompt = (
        "Answer the multiple-choice question from the context. "
        "Return only one letter: A, B, C, or D.\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}"
    )
    context = truncate_context(str(example["context"]), args.max_context_chars, args.context_strategy)
    rlm = RLM(
        backend=args.backend,
        backend_kwargs=make_backend_kwargs(args),
        environment="local",
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        verbose=args.verbose,
    )
    return rlm.completion(context, root_prompt=root_prompt).response


def output_path(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = args.model.replace("/", "__")
    mode = "rlm" if args.use_rlm else "direct"
    return out_dir / f"longbench_v2_{mode}_{model}_{timestamp}.jsonl"


def _bucketed(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Accuracy broken down by a categorical field (difficulty/length/domain)."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[str(r.get(key))].append(r)
    out: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(buckets.items()):
        n = len(rows)
        c = sum(int(r["correct"]) for r in rows)
        out[name] = {"correct": c, "total": n, "accuracy": (c / n if n else 0.0)}
    return out


def build_summary(
    args: argparse.Namespace, records: list[dict[str, Any]], out_path: Path, total_elapsed: float
) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(r["correct"]) for r in records)
    null_preds = sum(1 for r in records if r["pred"] is None)
    elapsed = [r["elapsed_s"] for r in records]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predictions_file": out_path.name,
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "backend": args.backend,
            "mode": "rlm" if args.use_rlm else "direct",
            "thinking": args.thinking,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "max_iterations": args.max_iterations,
            "max_depth": args.max_depth,
            "max_context_chars": args.max_context_chars,
            "filters": {
                "domain": args.domain,
                "sub_domain": args.sub_domain,
                "difficulty": sorted(args.difficulty) if args.difficulty else None,
                "length": sorted(args.length) if args.length else None,
                "limit": args.limit,
                "start": args.start,
            },
        },
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total if total else 0.0),
            "null_predictions": null_preds,
        },
        "by_difficulty": _bucketed(records, "difficulty"),
        "by_length": _bucketed(records, "length"),
        "by_domain": _bucketed(records, "domain"),
        "timing_s": {
            "total_wall": total_elapsed,
            "mean_per_example": (statistics.mean(elapsed) if elapsed else 0.0),
            "median_per_example": (statistics.median(elapsed) if elapsed else 0.0),
            "max_per_example": (max(elapsed) if elapsed else 0.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--backend", choices=["openai", "vllm", "openrouter", "anthropic", "hf"])
    parser.add_argument("--base-url")
    parser.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode for HF backend")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=1, help="0 means all matching examples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--domain")
    parser.add_argument("--sub-domain")
    parser.add_argument("--difficulty", help="comma-separated: easy,hard ('all' or omit = no filter)")
    parser.add_argument("--length", help="comma-separated: short,medium,long ('all' or omit = no filter)")
    parser.add_argument("--max-context-chars", type=int, default=30000, help="0 disables truncation")
    parser.add_argument("--context-strategy", choices=["head", "tail", "middle", "head-tail"], default="head-tail")
    parser.add_argument("--use-rlm", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=30, help="upstream RLM default is 30")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="HF datasets cache directory")
    parser.add_argument("--output-dir", default="evals/longbench_v2/results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.backend = args.backend or infer_backend(args.model)
    if args.max_context_chars == 0:
        args.max_context_chars = None
    # Normalize comma-separated filters to sets (None = no filter, incl. "all").
    args.difficulty = parse_filter(args.difficulty, {"easy", "hard"}, "difficulty")
    args.length = parse_filter(args.length, {"short", "medium", "long"}, "length")

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, split=args.split, cache_dir=str(cache_dir))
    print(f"Loaded {args.dataset}:{args.split} with {len(dataset)} examples")

    first = next(iter_examples(dataset, args), None)
    if first is None:
        raise SystemExit("No examples matched the requested filters.")

    if args.dry_run:
        idx, example = first
        prompt = format_prompt(example, args.max_context_chars, args.context_strategy)
        preview = prompt[:2000] + ("..." if len(prompt) > 2000 else "")
        print(json.dumps({k: example.get(k) for k in ("_id", "domain", "sub_domain", "difficulty", "length", "answer")}, indent=2))
        print(f"\nPrompt chars: {len(prompt)}")
        print(preview)
        return

    out_path = output_path(args)
    total = 0
    correct = 0
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    with out_path.open("w") as f:
        for idx, example in iter_examples(dataset, args):
            prompt = format_prompt(example, args.max_context_chars, args.context_strategy)
            t0 = time.perf_counter()
            raw = run_rlm(args, example) if args.use_rlm else run_direct(args, prompt)
            elapsed = time.perf_counter() - t0
            pred = extract_choice(raw)
            gold = str(example["answer"]).strip().upper()
            is_correct = pred == gold
            total += 1
            correct += int(is_correct)

            record = {
                "idx": idx,
                "_id": example.get("_id"),
                "domain": example.get("domain"),
                "sub_domain": example.get("sub_domain"),
                "difficulty": example.get("difficulty"),
                "length": example.get("length"),
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_response": raw,
                "prompt_chars": len(prompt),
                "context_chars": len(str(example["context"])),
                "elapsed_s": elapsed,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(f"[{total}] idx={idx} pred={pred} gold={gold} correct={is_correct} elapsed={elapsed:.1f}s")

    total_elapsed = time.perf_counter() - started
    accuracy = correct / total if total else 0.0
    summary = build_summary(args, records, out_path, total_elapsed)
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nAccuracy: {correct}/{total} = {accuracy:.3f}")
    if summary["overall"]["null_predictions"]:
        print(f"Null predictions: {summary['overall']['null_predictions']}")
    for label, key in (("difficulty", "by_difficulty"), ("length", "by_length")):
        parts = [f"{name}={s['correct']}/{s['total']}" for name, s in summary[key].items()]
        if len(parts) > 1:
            print(f"By {label}: " + "  ".join(parts))
    print(f"Elapsed: {total_elapsed:.1f}s (mean {summary['timing_s']['mean_per_example']:.1f}s/ex)")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
