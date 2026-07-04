# rlm-compaction

Exploring how KV cache compaction can be used within the Recursive Language Models (RLM) framework, and whether it can improve RLMs in practice. RLMs recursively call themselves/sub-models over long contexts, which tends to blow up KV cache size — this repo investigates compaction strategies (e.g. eviction, merging, low-rank/quantized summaries) to keep that cache manageable across recursive calls, and measures the effect on quality, memory, and speed.

This is an early-stage, exploratory research repo.

## Structure

```text
src/rlm/
  # verbatim copy of upstream alexzhang13/rlm (see src/rlm/UPSTREAM.md):
  core/         # RLM orchestration loop (rlm.py), LM handler, types
  clients/      # LM API clients (openai, anthropic, gemini, ...)
  environments/ # REPL sandboxes (local, ipython, docker, modal, ...)
  logger/       # trajectory logging + rich verbose output
  utils/        # prompts, parsing, exceptions, token utils
  # ours:
  handoff/      # swappable ways to pass context from orchestrator to worker
                # (text, summary, vanilla_am, latent_briefing)
  compaction/   # low-level KV/text compression methods (e.g. attention_matching)

experiments/    # smoke tests + handoff comparison scripts
configs/        # one config per handoff method
results/        # experiment outputs
```

The upstream code is kept as close to verbatim as possible (any local edits get noted in `src/rlm/UPSTREAM.md`). Our handoff/compaction work lives in the two extra subpackages: the RLM loop will talk to the generic `HandoffMethod` interface (`handoff/base.py`), never to compaction internals directly — that's what makes compaction/handoff methods swappable without rewriting the orchestration loop.

## External Code

`external/` is a **gitignored, local-only** scratch copy of prior work we're extracting from — not part of the committed project:

- `external/attention-matching` — clone of [adamzweiger/compaction](https://github.com/adamzweiger/compaction) (Fast KV Compaction via Attention Matching, MIT licensed). Kept around temporarily as a reference/demo to run unmodified while we identify the core files/functions we actually need.
- `external/rlm` — clone of [alexzhang13/rlm](https://github.com/alexzhang13/rlm) (official Recursive Language Models inference library, MIT OASYS lab). The `rlm/` package from this repo is copied verbatim into `src/rlm/` (provenance in `src/rlm/UPSTREAM.md`); the clone stays around for diffing and for its tests/docs.

The real implementation lives in `src/rlm/compaction/attention_matching/` — only the minimum extracted pieces, copied over and modified freely, with provenance recorded in that folder's `UPSTREAM.md`. To get set up locally:

```bash
git clone https://github.com/adamzweiger/compaction.git external/attention-matching
git clone https://github.com/alexzhang13/rlm.git external/rlm
```

## Setup

```bash
conda create -n rlm-comp python=3.12
conda activate rlm-comp
pip install -e .
```

Create a `.env` at the repo root with `OPENAI_API_KEY=...` (and optionally `ANTHROPIC_API_KEY=...`). Keys are loaded from `.env` via `python-dotenv` — `.env` is gitignored, never committed. Don't export real keys inside the conda env itself; it gets deleted/recreated and isn't meant to hold secrets.

Quick check that the RLM loop works (defaults to local Qwen3-4B in-process; no API key needed):

```bash
python experiments/test_rlm.py                        # built-in magic-number test
python experiments/test_rlm.py "your own prompt"      # custom prompt
python experiments/test_rlm.py --model gpt-5-mini     # API model (~1 cent)
```

## Status

🚧 Phase 1: upstream RLM loop vendored verbatim into `src/rlm/`, smoke test passes end-to-end. Handoff methods and compaction are still stubs.
