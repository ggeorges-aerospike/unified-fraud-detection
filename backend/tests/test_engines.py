"""Tests for the polymorphic investigation engine interface."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEngineFactory(unittest.TestCase):
    def test_supported_engines(self):
        from workflow.engines import SUPPORTED_ENGINES

        self.assertEqual(SUPPORTED_ENGINES, ("adk", "langgraph", "mock", "strands"))

    def test_unknown_engine_raises(self):
        from workflow.engines import get_engine

        fake = MagicMock()
        fake.is_connected.return_value = False
        with self.assertRaises(ValueError):
            get_engine("unknown", fake, fake)

    def test_langgraph_engine_steps(self):
        try:
            from workflow.engines.langgraph_engine import LangGraphEngine
        except ModuleNotFoundError:
            self.skipTest("langgraph not installed in test environment")

        fake = MagicMock()
        fake.is_connected.return_value = False
        eng = LangGraphEngine(fake, fake)
        self.assertEqual(eng.engine_name, "langgraph")
        ids = [s["id"] for s in eng.get_workflow_steps()]
        self.assertEqual(
            ids,
            ["alert_validation", "data_collection", "llm_agent", "report_generation"],
        )

    def test_get_engine_langgraph(self):
        try:
            from workflow.engines import get_engine
        except ModuleNotFoundError:
            self.skipTest("langgraph not installed in test environment")

        fake = MagicMock()
        fake.is_connected.return_value = False
        eng = get_engine("langgraph", fake, fake)
        self.assertEqual(eng.engine_name, "langgraph")

    def test_get_engine_mock(self):
        from workflow.engines import get_engine

        fake = MagicMock()
        fake.is_connected.return_value = False
        eng = get_engine("mock", fake, fake)
        self.assertEqual(eng.engine_name, "mock")


class TestActionCore(unittest.TestCase):
    def test_decision_sets(self):
        from workflow.action_core import (
            ALL_DECISIONS,
            DESTRUCTIVE_DECISIONS,
            NONDESTRUCTIVE_DECISIONS,
        )

        self.assertIn("temporary_freeze", DESTRUCTIVE_DECISIONS)
        self.assertIn("allow_monitor", NONDESTRUCTIVE_DECISIONS)
        self.assertEqual(ALL_DECISIONS, DESTRUCTIVE_DECISIONS | NONDESTRUCTIVE_DECISIONS)


class TestLLMConfig(unittest.TestCase):
    def test_defaults(self):
        from workflow.llm import PROVIDERS, LLMConfig

        cfg = LLMConfig.from_env()
        self.assertIn(cfg.provider, PROVIDERS)

    def test_ollama_adk_model_string(self):
        from workflow.llm import LLMConfig, resolve_adk_model

        cfg = LLMConfig(provider="ollama", ollama_model="mistral")
        self.assertEqual(resolve_adk_model(cfg), "ollama/mistral")

    def test_gemini_adk_model_retries_transient_errors(self):
        from workflow.llm import RETRYABLE_HTTP_STATUS_CODES, LLMConfig, model_name, resolve_adk_model

        cfg = LLMConfig(provider="gemini", model="gemini-3.5-flash", retry_attempts=4)
        try:
            model = resolve_adk_model(cfg)
        except ImportError:
            self.skipTest("google-adk not installed")
        # A Gemini instance (not a bare string) so the genai client retries 429/5xx;
        # the bare string yields a client that never retries, and one 503 then
        # kills the whole investigation.
        self.assertEqual(model.model, "gemini-3.5-flash")
        self.assertEqual(model_name(model), "gemini-3.5-flash")
        self.assertEqual(model.retry_options.attempts, 4)
        self.assertIn(503, model.retry_options.http_status_codes)
        self.assertEqual(tuple(model.retry_options.http_status_codes), RETRYABLE_HTTP_STATUS_CODES)

    def test_gemini_fallback_chain_reissues_on_503(self):
        import asyncio
        from unittest.mock import patch

        from workflow.llm import LLMConfig, resolve_adk_model

        cfg = LLMConfig(provider="gemini", model="gemini-3.5-flash",
                        fallback_models=["gemini-3.5-flash-lite", "gemini-3.5-flash"])
        try:
            from google.genai import errors
            from google.adk.models.llm_request import LlmRequest
            model = resolve_adk_model(cfg)
        except ImportError:
            self.skipTest("google-adk not installed")
        # The primary is never its own fallback.
        self.assertEqual(model.fallback_models, ["gemini-3.5-flash-lite"])

        calls = []

        # Both the primary (via super()) and the fallback instance hit the base
        # Gemini.generate_content_async, so one class patch covers both legs.
        async def fake_generate(self_, llm_request, stream=False):
            calls.append((type(self_).__name__, llm_request.model))
            if llm_request.model == "gemini-3.5-flash":
                raise errors.ServerError(503, {"error": {"message": "high demand"}}, None)
            yield "answer"

        Base = type(model).__mro__[1]  # google.adk Gemini
        with patch.object(Base, "generate_content_async", fake_generate):
            async def run():
                return [r async for r in model.generate_content_async(LlmRequest(model="gemini-3.5-flash"))]
            out = asyncio.run(run())
        self.assertEqual(out, ["answer"])
        self.assertEqual(calls, [("FallbackGemini", "gemini-3.5-flash"), ("Gemini", "gemini-3.5-flash-lite")])

    def test_gemini_fallback_does_not_mask_non_retryable_errors(self):
        import asyncio
        from unittest.mock import patch

        from workflow.llm import LLMConfig, resolve_adk_model

        cfg = LLMConfig(provider="gemini", model="gemini-3.5-flash", fallback_models=["gemini-3.5-flash-lite"])
        try:
            from google.genai import errors
            from google.adk.models.llm_request import LlmRequest
            model = resolve_adk_model(cfg)
        except ImportError:
            self.skipTest("google-adk not installed")

        async def primary(self_, llm_request, stream=False):
            raise errors.ClientError(400, {"error": {"message": "bad request"}}, None)
            yield  # pragma: no cover

        Base = type(model).__mro__[1]
        with patch.object(Base, "generate_content_async", primary):
            async def run():
                return [r async for r in model.generate_content_async(LlmRequest(model="gemini-3.5-flash"))]
            with self.assertRaises(errors.ClientError):
                asyncio.run(run())

    def test_vllm_from_env(self):
        import os
        from unittest.mock import patch

        from workflow.llm import VLLM, LLMConfig, describe

        env = {
            "LLM_PROVIDER": "vllm",
            "VLLM_BASE_URL": "http://vllm:8000/v1",
            "VLLM_MODEL": "Qwen/Qwen3-8B",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        self.assertEqual(cfg.provider, VLLM)
        self.assertEqual(cfg.vllm_base_url, "http://vllm:8000/v1")
        self.assertIn("Qwen/Qwen3-8B", describe(cfg))

    def test_vllm_adk_model_carries_the_base_url(self):
        from workflow.llm import LLMConfig, resolve_adk_model

        cfg = LLMConfig(provider="vllm", vllm_model="Qwen/Qwen3-8B",
                        vllm_base_url="http://vllm:8000/v1")
        try:
            model = resolve_adk_model(cfg)
        except ImportError:
            self.skipTest("google-adk not installed")
        # LiteLlm instance, so ADK routes to the OpenAI-compatible endpoint.
        self.assertEqual(model.model, "openai/Qwen/Qwen3-8B")


class TestSSEContractHelpers(unittest.TestCase):
    def test_merged_tool_calls_order(self):
        from workflow.sse_helpers import merged_tool_calls

        state = {
            "specialist_tool_calls_network_analyst": [{"tool": "detect_fraud_ring"}],
            "specialist_tool_calls_device_analyst": [{"tool": "get_device_risk_features"}],
            "specialist_tool_calls_velocity_analyst": [{"tool": "get_account_transactions"}],
            "tool_calls": [{"tool": "submit_assessment"}],
        }
        merged = merged_tool_calls(state)
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[-1]["tool"], "submit_assessment")


if __name__ == "__main__":
    unittest.main()
