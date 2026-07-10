"""In-process HuggingFace transformers client (ours, not upstream).

Runs a local model (e.g. Qwen/Qwen3-4B) directly via transformers instead of
an API server. This is the backend that gives us in-process access to the
model's KV cache, which the compaction/handoff work needs.
"""

import asyncio
import threading
from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

# Qwen3 model-card sampling recommendations, keyed by thinking mode.
# Greedy decoding degrades Qwen3 into endless repetition, so both modes sample.
# (presence_penalty is an API-server knob; the HF-generate analogue for taming
# repetition is repetition_penalty, accepted via sampling_args.)
SAMPLING_BY_THINKING_MODE = {
    True: {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    False: {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}


class HuggingFaceClient(BaseLM):
    """
    LM client that loads a HuggingFace causal LM in-process and generates locally.

    Args:
        model_name: HF hub id or local path (e.g. "Qwen/Qwen3-4B").
        max_new_tokens: generation cap per call (overridable via sampling_args).
        chat_template_kwargs: extra kwargs for tokenizer.apply_chat_template.
            Defaults to {"enable_thinking": False} — used by Qwen3's template,
            silently ignored by templates that don't have the variable.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 2048,
        chat_template_kwargs: dict[str, Any] | None = None,
        sampling_args: dict[str, Any] | None = None,
        dtype: str = "auto",
        **kwargs,
    ):
        super().__init__(model_name=model_name, sampling_args=sampling_args, **kwargs)

        # Import lazily so the rest of rlm doesn't require torch/transformers.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # "auto" = the model's native dtype. Pass dtype="float16"/"float32"
        # to override explicitly.
        if dtype != "auto":
            dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs = (
            {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        )
        # model.generate is not thread-safe; the LM handler serves concurrent requests.
        self._generate_lock = threading.Lock()

        # Per-model usage tracking (mirrors OpenAIClient)
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        import torch

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for HuggingFace client.")

        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs,
        )
        encoded = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = encoded["input_ids"].shape[1]

        sampling = dict(self.sampling_args)
        max_new_tokens = sampling.pop("max_tokens", None) or self.max_new_tokens

        thinking = self.chat_template_kwargs.get("enable_thinking")
        gen_kwargs: dict[str, Any] = dict(SAMPLING_BY_THINKING_MODE.get(thinking, {}))
        temperature = sampling.get("temperature")
        if temperature is not None:
            if temperature == 0:
                gen_kwargs = {"do_sample": False}
            else:
                gen_kwargs.update(do_sample=True, temperature=temperature)
        for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
            if sampling.get(key) is not None:
                gen_kwargs[key] = sampling[key]

        with self._generate_lock, torch.no_grad():
            output_ids = self.model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        completion_ids = output_ids[0, prompt_tokens:]
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # Qwen3 emits a think block before the answer.
        response = response.split("</think>", 1)[-1].strip()

        self._track_cost(model, prompt_tokens, completion_ids.shape[0])
        return response

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        return await asyncio.to_thread(self.completion, prompt, model)

    def _track_cost(self, model: str, prompt_tokens: int, completion_tokens: int):
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.model_total_tokens[model] += prompt_tokens + completion_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=None,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=None,
        )
