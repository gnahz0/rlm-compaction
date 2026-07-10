"""kv_scratch.py -- learning the KV cache mechanics (throwaway scratch script).

Run:  conda activate rlm-comp && python experiments/kv_scratch.py
This is NOT a test -- just edit, rerun, eyeball the prints. Delete when done.

RUNG 1 -- "does a KV cache even do what I think?"
Prove the core invariant to yourself before writing any handoff logic:
a cached prefill + one more token == a single full forward. If this passes,
you understand KV caching and the rest of FullKVHandoff is just packaging.

Use a TINY model (Qwen/Qwen3-0.6B) so reruns take seconds -- same family as
Qwen3-4B, so everything transfers to full_kv.py.

Steps:
  1. Load model + tokenizer (Qwen/Qwen3-0.6B). model.eval().
  2. Forward a short prompt with use_cache=True. Inspect out.past_key_values:
     print type(...), cache.get_seq_length(), and one layer's key .shape.
     GOAL: see what the cache actually IS.
  3. The invariant. Take token ids split into `prefix` + `next_token`. Compute
     the logits for `next_token`'s position TWO ways:
       (a) one full forward on prefix+next_token
       (b) prefill `prefix` (use_cache=True), then forward just `next_token`
           with past_key_values=that cache
     SUCCESS: the last-position logits match -> torch.allclose(a, b, atol=1e-3)

API you'll touch:
  model(input_ids, attention_mask, use_cache=True).past_key_values
  .past_key_values -> transformers DynamicCache
  cache.get_seq_length() ; cache.key_cache / cache.value_cache (lists of tensors)
  torch.no_grad() ; torch.allclose

Traps (recognize, don't pre-solve):
  - logits differ slightly -> check you're comparing the SAME position, and use
    atol ~1e-3 (fp math isn't bit-exact).
  - shape mismatch feeding next_token -> it needs a batch dim: shape [1, 1].
"""
# %%
# ===== CELL 1: load once (run a single time; model stays in the kernel) =====
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# --- 1. load model + tokenizer ---
# TODO
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# --- 2. one forward, inspect the cache ---
# TODO
prompt = "Can you prove that a continuous function on a closed interval is Riemann integrable?"

model.eval()
with torch.no_grad():
  #first have to tokenize
  inputs = tokenizer(prompt, return_tensors='pt')
  output = model(**inputs, use_cache=True)
  cache = output.past_key_values
  print(cache, type(cache))



# --- 3. the invariant: cached-prefill+token vs full forward ---
# TODO
