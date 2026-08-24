"""
Pluggable LLM provider configuration.

``LLM_PROVIDER`` selects the backend (``gemini``, ``ollama`` or ``vllm``). Each
engine maps that to its native client:

- ADK: Gemini model string, or ``LiteLlm`` for Ollama/OpenAI-compatible APIs.
- LangGraph: ``ChatGoogleGenerativeAI``, ``ChatOllama`` or ``ChatOpenAI``.

``vllm`` targets any OpenAI-compatible ``/v1`` endpoint — in this project that is
vLLM on ROCm, with LMCache in front of it and Aerospike as LMCache's remote KV
tier (see ``docs/lmcache-aerospike.md``). vLLM ignores the API key, so
``VLLM_API_KEY`` defaults to a placeholder.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

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
        )


def resolve_adk_model(config: LLMConfig) -> Any:
    """Model passed to ADK ``LlmAgent`` constructors.

    Gemini resolves from a plain string; the OpenAI-compatible vLLM endpoint needs
    a ``LiteLlm`` instance so the base URL travels with the model.
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
    return config.model


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
