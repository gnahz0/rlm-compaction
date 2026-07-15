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
  cache = output.past_key_values #kv cache
  print(cache.get_seq_length())
  print(cache.layers[0].keys.shape)

# --- 3a. cache invariant: prove cached prefill+token == one full forward ---
# This is the REAL Rung-1 proof and it compares LOGITS, so it's deterministic
# regardless of how we later decode. If this passes, the cache is correct.
with torch.no_grad():
  full_ids = inputs.input_ids
  prefix, last = full_ids[:, :-1], full_ids[:, -1:]        # split off final token

  logits_full = model(full_ids).logits[:, -1, :]           # (a) one full forward

  pref = model(prefix, use_cache=True)                     # (b) prefill prefix...
  logits_cached = model(last, past_key_values=pref.past_key_values).logits[:, -1, :]

  print("invariant holds:", torch.allclose(logits_full, logits_cached, atol=1e-3))


# --- 3b. decode after prefill, Qwen3 non-thinking sampling (NOT greedy) ---
# Qwen3 recommends temp=0.7, top_p=0.8, top_k=20, min_p=0 and explicitly warns
# against greedy decoding (endless repetition). We implement the sampler by hand
# so the cache mechanics stay visible.
def sample_next(logits, temperature=0.7, top_k=20, top_p=0.8, min_p=0.0):
  logits = logits / temperature
  if top_k > 0:                                            # keep only top_k logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < kth, float("-inf"))
  probs = torch.softmax(logits, dim=-1)
  if top_p < 1.0:                                          # nucleus: smallest set with cumprob >= top_p
    sp, idx = torch.sort(probs, descending=True, dim=-1)
    drop = (torch.cumsum(sp, dim=-1) - sp) > top_p
    probs = torch.zeros_like(probs).scatter(-1, idx, sp.masked_fill(drop, 0.0))
  if min_p > 0.0:                                          # drop tokens below min_p * peak
    probs = probs.masked_fill(probs < min_p * probs.max(dim=-1, keepdim=True).values, 0.0)
  probs = probs / probs.sum(dim=-1, keepdim=True)
  return torch.multinomial(probs, num_samples=1)          # shape [1, 1]

torch.manual_seed(0)                                       # reproducible sampling
with torch.no_grad():
  cache = output.past_key_values                           # reuse the prompt prefill
  next_id = sample_next(output.logits[:, -1, :])
  generated = [next_id]

  for _ in range(50):                                      # feed ONE token per step
    out = model(next_id, past_key_values=cache, use_cache=True)
    cache = out.past_key_values                            # grew by 1 position
    next_id = sample_next(out.logits[:, -1, :])
    generated.append(next_id)
    if next_id.item() == tokenizer.eos_token_id:
      break

  gen_ids = torch.cat(generated, dim=-1)
  print("\nsampled decode:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))
