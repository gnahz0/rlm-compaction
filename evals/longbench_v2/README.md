# LongBench-v2

LongBench-v2 is a multiple-choice long-context benchmark (`THUDM/LongBench-v2`,
503 examples). The runner loads the dataset, truncates each context with a
head/tail strategy, caches dataset files under scratch, and writes JSONL
predictions under `evals/longbench_v2/results/`.

The runner lives in this package (`__main__.py`), so it is still invoked as:

```bash
python -m evals.longbench_v2 [flags]
```

Default data cache path:

```bash
/orcd/scratch/orcd/010/alecz/rlm-compaction/hf-datasets
```

## Ready-made run script

One script, pick the model size with the first argument (`4b` | `8b` | `14b`,
default `8b`). It defaults to **all difficulties, short length** (`--difficulty
all --length short`, 180 examples), runs in Qwen3 **thinking mode** (`--thinking
--temperature 0.6 --max-new-tokens 4096`) on full untruncated contexts, and
forwards any extra flags. It `cd`s to the repo root and aborts if no GPU is
visible, so run it from inside a GPU session.

`--difficulty` and `--length` each accept `all` (or omit them) to disable that
filter; the concrete values are `easy|hard` and `short|medium|long`.

Thinking-mode notes: Qwen3 emits a `<think>…</think>` block that the client
strips before parsing the answer, and those tokens count against
`--max-new-tokens` — hence 4096. Do **not** run thinking mode with
`--temperature 0` (greedy): the client forces greedy decoding at temp 0, which
Qwen3 warns causes repetition loops that never close `</think>`, yielding `null`
predictions. `--temperature 0.6` selects the recommended thinking sampling
(top_p 0.95, top_k 20). Sampling makes runs non-deterministic; pass a wider
subset and/or repeat to average.

```bash
bash evals/longbench_v2/run_qwen3.sh               # Qwen3-8B (default), easy+short subset
bash evals/longbench_v2/run_qwen3.sh 4b            # Qwen3-4B
bash evals/longbench_v2/run_qwen3.sh 14b           # Qwen3-14B
bash evals/longbench_v2/run_qwen3.sh 4b --limit 3  # quick smoke test
```

Subset sizes: easy=192, hard=311; short=180, medium=215, long=108;
easy AND short=59; **all difficulties AND short=180** (the script default).
Widen with `--length medium`/`--length long`, or `--length all` for every
document length (= all 503, includes the very slow long tail).

## Manual invocations

Dry-run the loader and prompt formatting (no model):

```bash
python -m evals.longbench_v2 --dry-run
```

Direct local-HF evaluation (no RLM loop):

```bash
python -m evals.longbench_v2 \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --limit 1 \
  --max-context-chars 20000
```

Through the RLM loop:

```bash
python -m evals.longbench_v2 \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --use-rlm \
  --max-iterations 30 \
  --limit 0 \
  --difficulty easy \
  --length short \
  --max-context-chars 20000
```

Notes:
- `--limit 0` means "all matching examples"; the default is 1.
- `--max-new-tokens` defaults to 1024 — the RLM loop needs room to emit its REPL
  blocks plus a final answer; too small a value truncates the answer to `null`.
- `--max-context-chars` truncates each example's context string *before* the
  model sees it (head/tail strategy). The scripts set it to `0` (no truncation),
  matching the upstream RLM, which loads the full context as a REPL variable and
  lets the model chunk it via `llm_query`. Set a positive value only as a
  guardrail if the full 1M-char contexts are too slow or the model mis-chunks.
- HF weights load once per process and are cached in scratch via the
  `~/.cache/huggingface` → scratch symlink; no `HF_HOME` override is needed.
- Use `--cache-dir` to override the dataset cache, or `--output-dir` to change
  where result JSONL is written.

## Outputs

Each run writes two files to `results/` sharing the same stem:

- `longbench_v2_<mode>_<model>_<timestamp>.jsonl` — one JSON record per example
  (gold, pred, correct, raw response, char counts, elapsed).
- `longbench_v2_<mode>_<model>_<timestamp>.summary.json` — aggregate statistics:
  the run config, overall accuracy and null-prediction count, accuracy broken
  down by difficulty / length / domain, and per-example timing (mean, median,
  max, total wall). A condensed version is also printed at the end of the run.

Summarize across runs with:

```bash
python -m evals.summarize "evals/longbench_v2/results/longbench_v2_rlm_*.jsonl" --plot
```
