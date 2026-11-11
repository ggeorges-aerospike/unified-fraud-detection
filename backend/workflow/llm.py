"""
Pluggable LLM provider configuration.

``LLM_PROVIDER`` selects the backend (``gemini``, ``ollama`` or ``vllm``). Each
engine maps that to its native client:

- ADK: ``Gemini`` instance (with transient-error retries), or ``LiteLlm`` for
  Ollama/OpenAI-compatible APIs.
- LangGraph: ``ChatGoogleGenerativeAI``, ``ChatOllama`` or ``ChatOpenAI``.

``vllm`` targets any OpenAI-compatible ``/v1`` endpoint — in this project that is
vLLM on ROCm, with LMCache in front of it and Aerospike as LMCache's remote KV
tier (see ``docs/lmcache-aerospike.md``). vLLM ignores the API key, so
``VLLM_API_KEY`` defaults to a placeholder.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Optional

logger = logging.getLogger("investigation.llm")

GEMINI = "gemini"
OLLAMA = "ollama"
VLLM = "vllm"
PROVIDERS = (GEMINI, OLLAMA, VLLM)


@dataclass
class LLMConfig:
    """Runtime LLM settings (from environment)."""

    provider: str = GEMINI
    model: str = "gemini-3.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3-8B"
    vllm_api_key: str = "EMPTY"
    # Retries for transient Gemini failures (429/5xx, e.g. "503 model is
    # experiencing high demand"). The google-genai SDK does NOT retry unless
    # asked, and one 503 mid-run used to kill the whole investigation.
    retry_attempts: int = 5
    # Models tried in order when the primary keeps failing with a retryable
    # status after its own retries (a sustained "high demand" 503 on one model
    # tier while the others answer). Empty = no fallback.
    fallback_models: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = (os.environ.get("LLM_PROVIDER") or GEMINI).strip().lower()
        return cls(
            provider=provider,
            model=os.environ.get("ADK_MODEL", "gemini-3.5-flash"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "mistral"),
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            vllm_model=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
            vllm_api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            retry_attempts=int(os.environ.get("LLM_RETRY_ATTEMPTS", "5")),
            fallback_models=_split_models(os.environ.get("ADK_FALLBACK_MODELS", "")),
        )


def _split_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


# Transient statuses worth retrying. 429 = quota/rate limit, 5xx = Gemini-side
# (503 "high demand" is the one seen in practice).
RETRYABLE_HTTP_STATUS_CODES = (429, 500, 502, 503, 504)


def resolve_adk_model(config: LLMConfig) -> Any:
    """Model passed to ADK ``LlmAgent`` constructors.

    Gemini resolves to a ``Gemini`` instance carrying retry options, because the
    bare model string gives a client that never retries; the OpenAI-compatible
    vLLM endpoint needs a ``LiteLlm`` instance so the base URL travels with the
    model.
    """
    if config.provider == OLLAMA:
        # LiteLlm format: ollama/<model>
        return f"ollama/{config.ollama_model}"
    if config.provider == VLLM:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(
            model=f"openai/{config.vllm_model}",
            api_base=config.vllm_base_url,
            api_key=config.vllm_api_key,
        )
    from google.genai import types

    retry_options = types.HttpRetryOptions(
        attempts=max(1, config.retry_attempts),
        initial_delay=2,
        max_delay=30,
        http_status_codes=list(RETRYABLE_HTTP_STATUS_CODES),
    )
    fallbacks = [m for m in config.fallback_models if m != config.model]
    if fallbacks:
        return FallbackGemini(
            model=config.model, retry_options=retry_options, fallback_models=fallbacks,
        )
    from google.adk.models.google_llm import Gemini

    return Gemini(model=config.model, retry_options=retry_options)


def _is_retryable_api_error(exc: BaseException) -> bool:
    try:
        from google.genai import errors
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in RETRYABLE_HTTP_STATUS_CODES


try:
    from google.adk.models.google_llm import Gemini as _AdkGemini
except ImportError:  # pragma: no cover - google-adk optional for non-ADK engines
    _AdkGemini = None

if _AdkGemini is not None:

    class FallbackGemini(_AdkGemini):
        """ADK ``Gemini`` that re-issues a request on backup models when the primary
        stays unavailable.

        The SDK's retry policy covers a blip; this covers the case seen in demos
        where one model tier returns "503 high demand" for minutes while its
        siblings answer immediately. Only non-streaming calls are re-issued
        (nothing has been yielded yet, so no partial output is duplicated); a
        streaming call that already produced output re-raises.
        """

        fallback_models: List[str] = []

        def _fallback_llm(self, model: str) -> "_AdkGemini":
            cache = self.__dict__.setdefault("_fallback_cache", {})
            if model not in cache:
                cache[model] = _AdkGemini(model=model, retry_options=self.retry_options)
            return cache[model]

        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            yielded = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    yielded = True
                    yield response
                return
            except Exception as exc:  # noqa: BLE001 - narrowed below
                if yielded or not self.fallback_models or not _is_retryable_api_error(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "Model %s unavailable after retries (%s); falling back to %s",
                    llm_request.model or self.model, type(exc).__name__, self.fallback_models,
                )

            for fallback in self.fallback_models:
                request = llm_request.model_copy(update={"model": fallback})
                try:
                    async for response in self._fallback_llm(fallback).generate_content_async(
                        request, stream
                    ):
                        yield response
                    logger.info("Fallback model %s answered for %s", fallback, self.model)
                    return
                except Exception as exc:  # noqa: BLE001
                    if not _is_retryable_api_error(exc):
                        raise
                    logger.warning("Fallback model %s also unavailable (%s)", fallback, exc)
                    last_exc = exc
            raise last_exc

else:  # pragma: no cover
    FallbackGemini = None  # type: ignore[assignment]


def model_name(model: Any) -> str:
    """Readable model id for logs, whether ``model`` is a string or an ADK model object."""
    return str(getattr(model, "model", model))


def build_langgraph_chat_model(config: LLMConfig) -> Any:
    """Return a LangChain chat model for LangGraph nodes."""
    if config.provider == OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            temperature=0.3,
        )

    if config.provider == VLLM:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=config.vllm_base_url,
            api_key=config.vllm_api_key,
            model=config.vllm_model,
            temperature=0.3,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=api_key,
        temperature=0.3,
    )


def describe(config: LLMConfig) -> str:
    """Human-readable provider/model summary for logs and health output."""
    if config.provider == OLLAMA:
        return f"ollama:{config.ollama_model} @ {config.ollama_base_url}"
    if config.provider == VLLM:
        return f"vllm:{config.vllm_model} @ {config.vllm_base_url}"
    return f"gemini:{config.model}"
