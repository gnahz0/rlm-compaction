# Upstream provenance

The subpackages `clients/`, `core/`, `environments/`, `logger/`, `utils/` and
`__init__.py` are a **verbatim copy** of the `rlm/` package from
[alexzhang13/rlm](https://github.com/alexzhang13/rlm) (MIT license), copied
2026-07-02 at upstream commit `72d6940142ddfb84ee6be573dc999a37e633e671`
(local clone: `external/rlm`). Upstream's `tests/`, `training/`, `visualizer/`,
and `docs/` were not copied.

Our additions on top (not upstream):

- `handoff/` — swappable orchestrator→worker context-handoff methods
- `compaction/` — low-level KV/text compression (attention_matching, see its
  own `UPSTREAM.md`)

When we modify an upstream file, note it here so we can diff against
`external/rlm` later.

Modifications so far (trimming only — kept code is verbatim):

- Removed clients we won't use: `clients/gemini.py`, `clients/portkey.py`,
  `clients/azure_openai.py` (kept: openai, anthropic; `vllm`/`openrouter`
  backends route through `OpenAIClient`, which is also the path for local
  open-source models via `base_url`).
- Removed non-local sandboxes: `environments/{ipython,docker,modal,daytona,prime,e2b}_repl.py`
  and the now-orphaned `environments/constants.py` (kept: `local_repl.py`, `base_env.py`).
- Trimmed the corresponding routing branches in `clients/__init__.py`,
  `environments/__init__.py`, and the `ClientBackend`/`EnvironmentType`
  literals in `core/types.py`.

Additions (ours, not upstream):

- `clients/huggingface.py` — in-process transformers client (backend `"hf"`),
  for local models with direct KV-cache access; registered in
  `clients/__init__.py` and added to the `ClientBackend` literal.
