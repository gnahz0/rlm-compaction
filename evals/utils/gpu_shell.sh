#!/bin/bash
# Open an interactive GPU shell on ORCD for running the LongBench-v2 evals.
# Usage:
#   ./gpu_shell.sh                 # 1x H200, 6h (partition max)
#   ./gpu_shell.sh h100            # 1x H100, 6h
#   ./gpu_shell.sh h200 2 04:00:00 # 2x H200, 4h
#
# After it drops you into the compute node:
#   conda activate rlm-comp
#   nvidia-smi -L
#   bash evals/longbench_v2/run_qwen3_8b.sh --limit 3
#
# Note: run this from your login shell (it launches an interactive srun);
# it cannot be launched from inside another non-interactive job.

GPU="${1:-h200}"          # h200 | h100 | l40s
COUNT="${2:-1}"           # number of GPUs
TIME="${3:-06:00:00}"     # mit_normal_gpu max is 06:00:00

exec srun \
  --partition=mit_normal_gpu \
  --gres="gpu:${GPU}:${COUNT}" \
  --cpus-per-task=8 \
  --mem=96G \
  --time="${TIME}" \
  --job-name="lb2-${GPU}" \
  --pty bash
