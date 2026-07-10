#!/bin/bash
set -euo pipefail

# LongBench-v2 with a Qwen3 model through the RLM loop.
#
# Usage:
#   bash run_qwen3.sh <size> [extra flags...]
#
#   <size> is one of: 4b | 8b | 14b   (default: 8b)
#
# Defaults to the easiest subset (easy + short); widen by overriding on the
# command line (forwarded flags come last, so they win), e.g.:
#   bash run_qwen3.sh 4b --limit 3
#   bash run_qwen3.sh 14b --difficulty easy --length medium
#   bash run_qwen3.sh 8b --limit 0 --length medium

# Pick the model from the first positional arg (default 8b), then drop it so the
# rest of the arguments pass straight through to the eval.
SIZE="${1:-8b}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SIZE" in
  4b)  MODEL="Qwen/Qwen3-4B" ;;
  8b)  MODEL="Qwen/Qwen3-8B" ;;
  14b) MODEL="Qwen/Qwen3-14B" ;;
  *)
    echo "ERROR: unknown size '$SIZE' (expected: 4b | 8b | 14b)" >&2
    exit 2
    ;;
esac

# Run from the repo root so `python -m evals.longbench_v2` resolves.
cd "$(dirname "$0")/../.."

# Fail loudly if no GPU is visible (the eval silently falls back to CPU otherwise).
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no GPU available. Request a GPU session (e.g. mit_normal_gpu) before running." >&2
  exit 1
fi
nvidia-smi -L

python -m evals.longbench_v2 \
  --backend hf \
  --model "$MODEL" \
  --use-rlm \
  --max-iterations 30 \
  --difficulty all \
  --length short,medium \
  --limit 0 \
  --max-context-chars 0 \
  --thinking \
  --temperature 0.6 \
  --max-new-tokens 4096 \
  "$@"
