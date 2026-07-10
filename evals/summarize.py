"""Summarize LongBench-v2 prediction JSONL into readable tables (+ optional plot).

Pure stdlib by default so it runs anywhere on the cluster. matplotlib is only
imported when you pass --plot, and its absence is reported, not fatal.

Examples:
    # summarize the most recent result file
    python -m evals.summarize

    # a specific file, plus a PNG bar chart of accuracy by breakdown
    python -m evals.summarize evals/longbench_v2/results/longbench_v2_rlm_*.jsonl --plot

    # compare several runs side by side (e.g. direct vs rlm)
    python -m evals.summarize evals/longbench_v2/results/*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("evals/longbench_v2/results")
# Fields we bucket accuracy by, in display order.
BREAKDOWNS = ("domain", "difficulty", "length", "sub_domain")


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — honest error bars for small-n accuracy."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def resolve_paths(patterns: list[str]) -> list[Path]:
    if not patterns:
        files = sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit(f"No result files in {RESULTS_DIR}/ — run an eval first.")
        return [files[-1]]  # most recent
    paths: list[Path] = []
    for pat in patterns:
        matched = [Path(p) for p in glob.glob(pat)]
        paths.extend(matched or ([Path(pat)] if Path(pat).exists() else []))
    if not paths:
        raise SystemExit(f"No files matched: {patterns}")
    return sorted(dict.fromkeys(paths), key=lambda p: p.stat().st_mtime)


def summarize_one(path: Path) -> dict[str, Any]:
    records = load_records(path)
    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    invalid = sum(1 for r in records if r.get("pred") is None)
    latencies = [r["elapsed_s"] for r in records if r.get("elapsed_s") is not None]
    ctx = [r["context_chars"] for r in records if r.get("context_chars") is not None]

    buckets: dict[str, dict[str, list[int]]] = {}
    for field in BREAKDOWNS:
        b: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # key -> [correct, total]
        for r in records:
            key = r.get(field) or "—"
            b[key][0] += int(bool(r.get("correct")))
            b[key][1] += 1
        buckets[field] = dict(b)

    return {
        "path": path,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "latencies": latencies,
        "ctx": ctx,
        "buckets": buckets,
    }


def fmt_pct(correct: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * correct / total:5.1f}%"


def bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def print_summary(s: dict[str, Any]) -> None:
    total, correct = s["total"], s["correct"]
    lo, hi = wilson_interval(correct, total)
    print(f"\n\033[1m{s['path'].name}\033[0m")
    print("─" * 60)
    acc = correct / total if total else 0.0
    print(f"  Accuracy   {correct}/{total} = \033[1m{100*acc:.1f}%\033[0m  "
          f"[95% CI {100*lo:.1f}–{100*hi:.1f}]  {bar(acc)}")
    if s["invalid"]:
        print(f"  \033[33mUnparsed predictions: {s['invalid']} "
              f"({100*s['invalid']/total:.0f}%)\033[0m  (counted as wrong)")
    if s["latencies"]:
        lat = sorted(s["latencies"])
        p50 = lat[len(lat) // 2]
        print(f"  Latency    mean {sum(lat)/len(lat):.1f}s  "
              f"median {p50:.1f}s  max {lat[-1]:.1f}s  total {sum(lat)/60:.1f}min")
    if s["ctx"]:
        print(f"  Context    mean {sum(s['ctx'])/len(s['ctx'])/1000:.0f}k chars  "
              f"max {max(s['ctx'])/1000:.0f}k")

    for field in BREAKDOWNS:
        b = s["buckets"][field]
        if len(b) <= 1 and "—" in b:
            continue  # field absent from this run
        print(f"\n  by {field}:")
        for key in sorted(b, key=lambda k: (-b[k][1], k)):
            c, t = b[key]
            frac = c / t if t else 0.0
            print(f"    {str(key)[:22]:22} {fmt_pct(c, t)}  ({c}/{t})  {bar(frac, 12)}")


def maybe_plot(summaries: list[dict[str, Any]], out: Path, field: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display on compute nodes
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n\033[33m[--plot] matplotlib not installed; skipping graph.\033[0m")
        print("  pip install matplotlib  (or run without --plot)")
        return

    # Collect the union of bucket keys for the chosen field across runs.
    keys: list[str] = []
    for s in summaries:
        for k in s["buckets"].get(field, {}):
            if k not in keys:
                keys.append(k)
    if not keys:
        print(f"\n[--plot] no '{field}' breakdown to plot.")
        return

    n_runs = len(summaries)
    width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(keys)), 4.5))
    for i, s in enumerate(summaries):
        b = s["buckets"].get(field, {})
        vals = [100 * b[k][0] / b[k][1] if k in b and b[k][1] else 0 for k in keys]
        xs = [j + i * width for j in range(len(keys))]
        ax.bar(xs, vals, width=width, label=s["path"].stem[:28])
        for x, v in zip(xs, vals):
            if v:
                ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks([j + width * (n_runs - 1) / 2 for j in range(len(keys))])
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"LongBench-v2 accuracy by {field}")
    ax.axhline(25, ls="--", lw=1, color="gray", alpha=0.7)  # random-chance baseline (4-way MC)
    ax.text(len(keys) - 0.5, 26, "chance", color="gray", fontsize=8, va="bottom", ha="right")
    if n_runs > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\n\033[32mSaved plot → {out}\033[0m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="result JSONL path(s)/glob(s); default = latest in evals/longbench_v2/results/")
    ap.add_argument("--plot", action="store_true", help="render a PNG accuracy bar chart")
    ap.add_argument("--plot-by", default="length", choices=BREAKDOWNS, help="breakdown field for the plot")
    ap.add_argument("--plot-out", default="evals/longbench_v2/results/accuracy.png")
    args = ap.parse_args()

    paths = resolve_paths(args.files)
    summaries = [summarize_one(p) for p in paths]
    for s in summaries:
        print_summary(s)

    if len(summaries) > 1:
        print("\n" + "═" * 60)
        print("  \033[1mComparison\033[0m")
        for s in summaries:
            print(f"    {fmt_pct(s['correct'], s['total'])}  {s['path'].name}")

    if args.plot:
        maybe_plot(summaries, Path(args.plot_out), args.plot_by)


if __name__ == "__main__":
    main()
