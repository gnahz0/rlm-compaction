import json
from typing import Any

from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend

load_dotenv()

# Loading a HuggingFace model reads GBs of weights from disk. The RLM loop calls
# get_client() once per completion, so without caching the same local model is
# reloaded for every example. Reuse a client when the backend_kwargs are
# identical (same model, dtype, sampling, etc.); differing kwargs get a fresh one.
_HF_CLIENT_CACHE: dict[str, BaseLM] = {}


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Routes a specific backend and the args (as a dict) to the appropriate client if supported.
    Currently supported backends: ['openai']
    """
    if backend == "openai":
        from rlm.clients.openai import OpenAIClient

        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        from rlm.clients.openai import OpenAIClient

        assert "base_url" in backend_kwargs, (
            "base_url is required to be set to local vLLM server address for vLLM"
        )
        return OpenAIClient(**backend_kwargs)
    elif backend == "openrouter":
        from rlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "anthropic":
        from rlm.clients.anthropic import AnthropicClient

        return AnthropicClient(**backend_kwargs)
    elif backend == "hf":
        from rlm.clients.huggingface import HuggingFaceClient

        cache_key = json.dumps(backend_kwargs, sort_keys=True, default=str)
        client = _HF_CLIENT_CACHE.get(cache_key)
        if client is None:
            client = HuggingFaceClient(**backend_kwargs)
            _HF_CLIENT_CACHE[cache_key] = client
        return client
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: ['openai', 'vllm', 'openrouter', 'anthropic', 'hf']"
        )
