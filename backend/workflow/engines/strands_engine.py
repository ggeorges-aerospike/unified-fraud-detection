"""
AWS Strands Agents investigation engine (Amazon Bedrock), with agent state in Aerospike.

Same pipeline and SSE contract as the ADK and LangGraph engines:

  alert_validation -> data_collection -> cross-case memory recall
  -> three specialists IN PARALLEL (network / device / velocity)
  -> investigator (submits the assessment) -> report -> action (analyst approval if destructive)

What Strands adds, and where its state lives (all through ``strands-aerospike``):

- **Agent sessions** — every agent (3 specialists, investigator, report writer) runs with an
  ``AerospikeSessionManager`` under one session per investigation: one record per message,
  set ``strands_sessions``.
- **Cross-case memory** — completed cases go to an ``AerospikeMemoryStore`` (lexical search on a
  list secondary index, set ``strands_memory``); new investigations recall prior cases that
  touched the same accounts, devices or counterparties.
- **Pause / resume for analyst approval** — the paused investigation's state is written to
  ``AerospikeStorage`` (set ``strands_storage``) so it survives restarts.

Models (env-configurable): specialists default to Claude Sonnet 5.5, the investigator and report
writer to Claude Opus 5.5, via Bedrock inference profiles, with adaptive thinking and an explicit
effort level. Claude 5.5 models reject temperature/top_p, ``budget_tokens`` and forced tool
choice, so none are sent, and the assessment is submitted through an ordinary tool rather than
Strands structured output (which forces a tool call).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from workflow.action_core import ALL_DECISIONS, DESTRUCTIVE_DECISIONS, execute_action
from workflow.assessment import build_evidence_summary, deterministic_assessment
from workflow.engines.base import BaseInvestigationEngine
from workflow.metrics import get_collector, remove_collector
from workflow.nodes.alert_validation import alert_validation_node
from workflow.nodes.data_collection import data_collection_node
from workflow.nodes.langgraph_agent import format_prior_cases, format_specialist_findings
from workflow.nodes.report_generation import (
    build_report_instruction,
    finalize_report,
    generate_fallback_report,
)
from workflow.specialists import (
    DEVICE_ANALYST_NAME,
    INVESTIGATOR_SYSTEM,
    NETWORK_ANALYST_NAME,
    SPECIALIST_NAMES,
    SPECIALIST_OUTPUT_KEYS,
    VELOCITY_ANALYST_NAME,
    _SPECIALIST_SPECS,
    _SPECIALIST_SYSTEM,
    _SPECIALIST_TOOLS,
)
from workflow.sse_helpers import merged_tool_calls, specialist_findings
from workflow.state import create_initial_state
from workflow.tools.investigation_tools import InvestigationTools

logger = logging.getLogger("investigation.engines.strands")

MEMORY_STORE_NAME = "fraud_cases"
SESSION_SET = "strands_sessions"
MEMORY_SET = "strands_memory"
STORAGE_SET = "strands_storage"
PAUSE_FC_ID = "strands_pause"

INVESTIGATOR_TOOLS = [
    "get_account_transactions",
    "get_counterparty_profile",
    "get_counterparty_transactions",
    "get_account_risk_features",
    "get_device_risk_features",
    "detect_fraud_ring",
    "get_transaction_network",
]
INVESTIGATOR_MAX_CALLS = 10
TOOL_RESULT_MAX_CHARS = 24_000  # keep one huge ledger from swamping an agent's context

_TOOL_CALL_KEYS = {
    NETWORK_ANALYST_NAME: "specialist_tool_calls_network_analyst",
    DEVICE_ANALYST_NAME: "specialist_tool_calls_device_analyst",
    VELOCITY_ANALYST_NAME: "specialist_tool_calls_velocity_analyst",
}


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RoleModel:
    model_id: str
    effort: str


@dataclass(frozen=True)
class StrandsConfig:
    region: str
    specialist: RoleModel
    investigator: RoleModel
    report: RoleModel
    max_tokens: int

    @classmethod
    def from_env(cls) -> "StrandsConfig":
        e = os.environ.get
        return cls(
            region=e("STRANDS_REGION") or e("AWS_REGION") or e("AWS_DEFAULT_REGION") or "us-east-1",
            specialist=RoleModel(e("STRANDS_SPECIALIST_MODEL") or "us.anthropic.claude-sonnet-5-5",
                                 e("STRANDS_SPECIALIST_EFFORT") or "medium"),
            investigator=RoleModel(e("STRANDS_INVESTIGATOR_MODEL") or "us.anthropic.claude-opus-5-5",
                                   e("STRANDS_INVESTIGATOR_EFFORT") or "high"),
            report=RoleModel(e("STRANDS_REPORT_MODEL") or "us.anthropic.claude-opus-5-5",
                             e("STRANDS_REPORT_EFFORT") or "medium"),
            max_tokens=int(e("STRANDS_MAX_TOKENS") or 16000),
        )


def short_model_name(model_id: str) -> str:
    """'us.anthropic.claude-sonnet-5-5' -> 'Sonnet 5.5' (falls back to the id)."""
    m = re.search(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", model_id or "")
    if not m:
        return model_id
    family, major, minor = m.group(1), m.group(2), m.group(3)
    return f"{family.capitalize()} {major}{'.' + minor if minor else ''}"


def _trace(node: str, type_: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "trace", "event": {"type": type_, "node": node,
                                       "timestamp": datetime.now().isoformat(), "data": data}}


def _session_id(investigation_id: str) -> str:
    # strands-aerospike session ids may not contain ':' (it separates key segments).
    return re.sub(r"[^A-Za-z0-9_.-]", "_", investigation_id)


# ---------------------------------------------------------------------------------------------
# Tools: Strands wrappers over the shared InvestigationTools, with a per-agent call budget
# ---------------------------------------------------------------------------------------------
class ToolRecorder:
    """Runs tools for one agent, records calls for the UI, and enforces its call budget."""

    def __init__(self, agent_name: str, node: str, tools: InvestigationTools, metrics: Any,
                 max_calls: int, emit: Callable[[Dict[str, Any]], None]):
        self.agent_name = agent_name
        self.node = node
        self.tools = tools
        self.metrics = metrics
        self.max_calls = max_calls
        self.emit = emit
        self.tool_calls: List[Dict[str, Any]] = []
        self.assessment: Optional[Dict[str, Any]] = None

    def call(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if len(self.tool_calls) >= self.max_calls:
            return {"success": False,
                    "error": f"Tool budget of {self.max_calls} calls reached. Stop calling tools and give your answer now."}
        params = {k: v for k, v in params.items() if v is not None}
        result = self.tools.execute_tool(name, params)
        self.tool_calls.append({"tool": name, "params": params, "result": result,
                                "timestamp": datetime.now().isoformat(), "iteration": len(self.tool_calls) + 1})
        self.metrics.track_tool_call(name)
        self.emit(_trace(self.node, "tool_call", {"tool": name, "params": params,
                                                 "iteration": len(self.tool_calls), "agent": self.agent_name}))
        text = json.dumps(result, default=str)
        if len(text) > TOOL_RESULT_MAX_CHARS:
            return {"truncated": True, "note": f"Result shortened to {TOOL_RESULT_MAX_CHARS} characters.",
                    "partial_json": text[:TOOL_RESULT_MAX_CHARS]}
        return result

    def submit(self, typology: str, risk_level: str, risk_score: int, decision: str,
               account_id: str, reasoning: str) -> Dict[str, Any]:
        decision = (decision or "").strip()
        if decision not in ALL_DECISIONS:
            return {"accepted": False,
                    "error": f"decision must be one of: {', '.join(sorted(ALL_DECISIONS))}. Submit again."}
        self.assessment = {
            "typology": typology or "unknown",
            "risk_level": (risk_level or "medium").lower(),
            "risk_score": max(0, min(100, int(risk_score or 0))),
            "decision": decision,
            "account_id": (account_id or "").strip(),
            "reasoning": reasoning or "",
            "iteration": len(self.tool_calls) + 1,
            "tool_calls_made": len(self.tool_calls),
        }
        self.emit(_trace(self.node, "assessment", {k: self.assessment[k] for k in
                                                   ("typology", "risk_level", "risk_score", "decision")}))
        return {"accepted": True, "note": "Assessment recorded. Stop now; do not call any more tools."}


def build_strands_tools(rec: ToolRecorder, names: List[str], include_submit: bool = False) -> List[Any]:
    """Strands @tool functions for the named investigation tools, bound to ``rec``."""
    from strands import tool

    @tool(name="get_account_transactions")
    def get_account_transactions(account_id: str, days: int = 30) -> dict:
        """Pull the transaction ledger for one account: amounts, times, counterparties (with their
        counterparty_user_id), locations and devices. Use it for velocity, bursts and amount anomalies.

        Args:
            account_id: The account to pull transactions for (e.g. A000613003).
            days: Days to look back, 1-90.
        """
        return rec.call("get_account_transactions", {"account_id": account_id, "days": days})

    @tool(name="get_counterparty_profile")
    def get_counterparty_profile(user_id: str) -> dict:
        """Profile of a user the suspect transacted with: name, location, signup date, risk score,
        accounts (balances, fraud flags) and devices.

        Args:
            user_id: The counterparty's user_id, taken from transaction data.
        """
        return rec.call("get_counterparty_profile", {"user_id": user_id})

    @tool(name="get_counterparty_transactions")
    def get_counterparty_transactions(user_id: str, days: int = 30) -> dict:
        """All transactions across a counterparty's accounts, to spot mule behaviour (many sources,
        rapid pass-through).

        Args:
            user_id: The counterparty's user_id.
            days: Days to look back, 1-90.
        """
        return rec.call("get_counterparty_transactions", {"user_id": user_id, "days": days})

    @tool(name="get_account_risk_features")
    def get_account_risk_features(account_id: str) -> dict:
        """Pre-computed risk features for an account: velocity, amount patterns, counterparty spread,
        device exposure and lifecycle metrics.

        Args:
            account_id: The account to get risk features for.
        """
        return rec.call("get_account_risk_features", {"account_id": account_id})

    @tool(name="get_device_risk_features")
    def get_device_risk_features(device_id: str) -> dict:
        """Pre-computed risk features for a device: how many accounts share it, how many are flagged,
        and their risk scores.

        Args:
            device_id: The device to get risk features for.
        """
        return rec.call("get_device_risk_features", {"device_id": device_id})

    @tool(name="detect_fraud_ring")
    def detect_fraud_ring(hops: int = 2) -> dict:
        """Check the graph for a coordinated ring around the suspect: shared devices, transaction links,
        and overlap between the two.

        Args:
            hops: Traversal depth, 1-3.
        """
        return rec.call("detect_fraud_ring", {"hops": hops})

    @tool(name="get_transaction_network")
    def get_transaction_network(hops: int = 2, min_amount: float = 0) -> dict:
        """Money-flow network around the suspect: who paid whom, totals, counts, and which accounts
        on the path are flagged.

        Args:
            hops: Traversal depth, 1-3.
            min_amount: Only include edges whose total amount is above this.
        """
        return rec.call("get_transaction_network", {"hops": hops, "min_amount": min_amount})

    @tool(name="submit_assessment")
    def submit_assessment(typology: str, risk_level: str, risk_score: int, decision: str,
                          account_id: str, reasoning: str) -> dict:
        """Submit the final fraud assessment exactly once, when the evidence is sufficient.

        Args:
            typology: account_takeover, money_mule, synthetic_identity, promo_abuse, friendly_fraud,
                card_testing, fraud_ring, suspicious_activity, or legitimate.
            risk_level: low, medium, high, or critical.
            risk_score: 0-100.
            decision: allow_monitor, step_up_auth, temporary_freeze, full_block, or escalate_compliance.
            account_id: The primary flagged account the decision applies to.
            reasoning: Reasoning that cites specific evidence (amounts, ids, patterns).
        """
        return rec.submit(typology, risk_level, risk_score, decision, account_id, reasoning)

    available = {
        "get_account_transactions": get_account_transactions,
        "get_counterparty_profile": get_counterparty_profile,
        "get_counterparty_transactions": get_counterparty_transactions,
        "get_account_risk_features": get_account_risk_features,
        "get_device_risk_features": get_device_risk_features,
        "detect_fraud_ring": detect_fraud_ring,
        "get_transaction_network": get_transaction_network,
    }
    out = [available[n] for n in names if n in available]
    if include_submit:
        out.append(submit_assessment)
    return out


# ---------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------
@dataclass
class AgentRun:
    text: str
    stop_reason: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class StrandsEngine(BaseInvestigationEngine):
    """AWS Strands Agents on Amazon Bedrock; sessions, memory and pauses stored in Aerospike."""

    def __init__(self, aerospike_service: Any, graph_service: Any, llm_config: Any = None,
                 config: Optional[StrandsConfig] = None,
                 agent_runner: Optional[Callable[..., Any]] = None):
        self.aerospike_service = aerospike_service
        self.graph_service = graph_service
        self.config = config or StrandsConfig.from_env()
        # (role, agent_name, system_prompt, prompt, tools, session_id, max_turns) -> AgentRun.
        # Tests replace this to drive the engine without Bedrock.
        self._run_agent = agent_runner or self._run_bedrock_agent
        self._memory = None
        self._storage = None

    # -- identity ------------------------------------------------------------------------------
    @property
    def engine_name(self) -> str:
        return "strands"

    def describe(self) -> Dict[str, Any]:
        roles = [self.config.specialist.model_id, self.config.investigator.model_id, self.config.report.model_id]
        names = list(dict.fromkeys(short_model_name(m) for m in roles))
        return {"engine": "strands", "label": "AWS Strands", "model": " · ".join(names)}

    def get_workflow_steps(self) -> List[Dict[str, str]]:
        return [
            {"id": "alert_validation", "name": "Alert Validation",
             "description": "Extract alert trigger context from flagged account", "phase": "context"},
            {"id": "data_collection", "name": "Data Collection",
             "description": "Gather profile, accounts, devices; recall related prior cases", "phase": "evidence"},
            {"id": "llm_agent", "name": "AI Investigation Agents",
             "description": "Strands agents on Amazon Bedrock: three specialists in parallel, then an investigator",
             "phase": "reasoning"},
            {"id": "report_generation", "name": "Report Generation",
             "description": "Generate detailed investigation report", "phase": "report"},
        ]

    # -- lifecycle -----------------------------------------------------------------------------
    def _aerospike_ready(self) -> bool:
        a = self.aerospike_service
        return a is not None and a.is_connected()

    async def initialize(self) -> None:
        if self._aerospike_ready():
            from strands_aerospike import AerospikeMemoryStore, AerospikeStorage

            ns, client = self.aerospike_service.namespace, self.aerospike_service.client
            self._memory = AerospikeMemoryStore(name=MEMORY_STORE_NAME, namespace=ns, set=MEMORY_SET,
                                                client=client, extraction=False)
            await self._memory.initialize()
            self._storage = AerospikeStorage(namespace=ns, set=STORAGE_SET, client=client)
        logger.info("StrandsEngine initialized (%s, region=%s)", self.describe()["model"], self.config.region)

    async def close(self) -> None:
        self._memory = None
        self._storage = None

    # -- Bedrock agent execution ---------------------------------------------------------------
    def _bedrock_model(self, role: RoleModel) -> Any:
        from botocore.config import Config as BotocoreConfig
        from strands.models import BedrockModel
        from strands.models.model import CacheConfig

        return BedrockModel(
            model_id=role.model_id,
            region_name=self.config.region,
            max_tokens=self.config.max_tokens,
            additional_request_fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": role.effort}},
            cache_config=CacheConfig(strategy="auto"),
            boto_client_config=BotocoreConfig(retries={"max_attempts": 6, "mode": "adaptive"},
                                              read_timeout=300, connect_timeout=10),
        )

    def _session_manager(self, session_id: str) -> Any:
        if not self._aerospike_ready():
            return None
        from strands_aerospike import AerospikeSessionManager

        return AerospikeSessionManager(session_id=session_id, namespace=self.aerospike_service.namespace,
                                       set=SESSION_SET, client=self.aerospike_service.client)

    async def _run_bedrock_agent(self, role: RoleModel, agent_name: str, system_prompt: Optional[str],
                                 prompt: str, tools: List[Any], session_id: str, max_turns: int) -> AgentRun:
        from strands import Agent
        from strands.types.agent import Limits

        # Session manager construction reads/creates the session record; keep it off the event loop.
        session_manager = await asyncio.to_thread(self._session_manager, session_id)
        agent = Agent(model=self._bedrock_model(role), system_prompt=system_prompt, tools=tools,
                      agent_id=agent_name, name=agent_name, session_manager=session_manager,
                      callback_handler=None)
        result = await agent.invoke_async(prompt, limits=Limits(turns=max_turns))
        usage = getattr(result.metrics, "accumulated_usage", None) or {}
        latency = (getattr(result.metrics, "accumulated_metrics", None) or {}).get("latencyMs", 0)
        return AgentRun(text=str(result).strip(), stop_reason=str(result.stop_reason), latency_ms=float(latency),
                        input_tokens=int(usage.get("inputTokens", 0)), output_tokens=int(usage.get("outputTokens", 0)))

    # -- investigation -------------------------------------------------------------------------
    async def run_investigation(self, user_id: str, investigation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        if self._memory is None and self._aerospike_ready():
            await self.initialize()
        metrics = get_collector(investigation_id)
        metrics.reset()
        state: Dict[str, Any] = dict(create_initial_state(investigation_id, user_id))
        session_id = _session_id(investigation_id)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: Dict[str, Any]) -> None:  # safe from tool threads
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def drain() -> List[Dict[str, Any]]:
            out = []
            while not queue.empty():
                out.append(queue.get_nowait())
            return out

        try:
            # 1. alert validation + 2. data collection (shared nodes, synchronous I/O)
            for step, fn in (("alert_validation", lambda s: alert_validation_node(s, self.aerospike_service)),
                             ("data_collection", lambda s: data_collection_node(s, self.aerospike_service,
                                                                                 self.graph_service))):
                metrics.start_node(step)
                yield _trace(step, "node_start", {"user_id": user_id})
                update = await asyncio.to_thread(fn, state)
                state.update(update or {})
                for ev in (update or {}).get("trace_events") or []:
                    yield {"type": "trace", "event": {**ev, "node": step}}
                if step == "data_collection":
                    state["prior_cases"] = await self._recall_cases(state, user_id, investigation_id)
                    if state["prior_cases"]:
                        yield _trace(step, "memory_recall", {"prior_cases": state["prior_cases"]})
                yield _trace(step, "node_complete", {"status": "success"})
                metrics.end_node(step)

            initial = state.get("initial_evidence") or {}
            alert = state.get("alert_evidence") or {}
            evidence = build_evidence_summary(initial, alert)

            # 3. specialists in parallel, streaming their tool calls as they happen
            metrics.start_node("llm_agent")
            yield _trace("llm_agent", "node_start", {"user_id": user_id, "engine": "strands"})
            # Create the investigation's Aerospike session once, up front: each agent's session
            # manager creates the session if it is missing, and three specialists starting at the
            # same moment would otherwise race to create it ("Session ... already exists").
            await asyncio.to_thread(self._session_manager, session_id)
            recorders = {
                name: ToolRecorder(name, "llm_agent", InvestigationTools(self.aerospike_service, self.graph_service,
                                                                          user_id, metrics),
                                   metrics, _SPECIALIST_SPECS[name]["max_calls"], emit)
                for name in SPECIALIST_NAMES
            }

            async def run_specialist(name: str) -> AgentRun:
                spec = _SPECIALIST_SPECS[name]
                system = _SPECIALIST_SYSTEM.format(role=spec["role"], focus=spec["focus"],
                                                   max_calls=spec["max_calls"], evidence=evidence)
                t0 = time.time()
                run = await self._run_agent(self.config.specialist, name, system,
                                            "Investigate your domain now, then write your findings report.",
                                            build_strands_tools(recorders[name], _SPECIALIST_TOOLS[name]),
                                            session_id, spec["max_calls"] + 3)
                metrics.track_llm_call(run.latency_ms or (time.time() - t0) * 1000, run.input_tokens,
                                       run.output_tokens, model=self.config.specialist.model_id)
                return run

            tasks = {name: asyncio.create_task(run_specialist(name)) for name in SPECIALIST_NAMES}
            pending = set(tasks.values())
            while pending:
                done, pending = await asyncio.wait(pending, timeout=0.25, return_when=asyncio.FIRST_COMPLETED)
                for ev in drain():
                    yield ev
            for ev in drain():
                yield ev

            for name, task in tasks.items():
                state[_TOOL_CALL_KEYS[name]] = recorders[name].tool_calls
                exc = task.exception()
                if exc:
                    logger.warning("[%s] specialist %s failed: %s", investigation_id, name, exc)
                    state[SPECIALIST_OUTPUT_KEYS[name]] = f"(The {name.replace('_', ' ')} could not finish: {exc})"
                    yield _trace("llm_agent", "error", {"agent": name, "error": str(exc)[:300]})
                    continue
                findings = task.result().text[:1500]
                state[SPECIALIST_OUTPUT_KEYS[name]] = findings
                if findings:
                    yield _trace("llm_agent", "specialist_finding", {"agent": name, "finding": findings})
            if all(t.exception() for t in tasks.values()):
                raise RuntimeError(f"All three specialists failed; first error: {tasks[NETWORK_ANALYST_NAME].exception()}")

            # 4. investigator — submits the assessment through a normal tool call
            inv_rec = ToolRecorder("investigator", "llm_agent",
                                   InvestigationTools(self.aerospike_service, self.graph_service, user_id, metrics),
                                   metrics, INVESTIGATOR_MAX_CALLS, emit)
            system = INVESTIGATOR_SYSTEM.format(findings=format_specialist_findings(state))
            prompt = (build_evidence_summary(initial, alert) + format_prior_cases(state.get("prior_cases") or [])
                      + "\n\nDecide now. Call submit_assessment exactly once, then stop.")
            inv_task = asyncio.create_task(self._run_agent(
                self.config.investigator, "investigator", system, prompt,
                build_strands_tools(inv_rec, INVESTIGATOR_TOOLS, include_submit=True),
                session_id, INVESTIGATOR_MAX_CALLS + 4))
            while not inv_task.done():
                await asyncio.wait({inv_task}, timeout=0.25)
                for ev in drain():
                    yield ev
            for ev in drain():
                yield ev
            run = inv_task.result()  # an investigator failure is a real error, not a silent fallback
            metrics.track_llm_call(run.latency_ms, run.input_tokens, run.output_tokens,
                                   model=self.config.investigator.model_id)
            yield _trace("llm_agent", "agent_thinking", {"agent": "investigator", "response_preview": run.text[:200],
                                                        "llm_duration_ms": round(run.latency_ms, 2)})

            assessment = inv_rec.assessment
            if not assessment:
                yield _trace("llm_agent", "error", {"agent": "investigator",
                                                    "error": f"No assessment submitted (stop={run.stop_reason}); "
                                                             "using the rule-based assessment."})
                assessment = dict(deterministic_assessment(initial, alert))
            if not assessment.get("account_id"):
                assessment["account_id"] = next(iter(initial.get("accounts") or {}), "")
            state.update({"final_assessment": assessment, "tool_calls": inv_rec.tool_calls,
                          "agent_iterations": len(inv_rec.tool_calls) + 1,
                          "tool_calls_count": len(inv_rec.tool_calls), "current_phase": "report"})
            yield _trace("llm_agent", "node_complete", {"status": "success"})
            metrics.end_node("llm_agent")

            # 5. report
            metrics.start_node("report_generation")
            yield _trace("report_generation", "node_start", {"user_id": user_id})
            report = ""
            try:
                rep = await self._run_agent(self.config.report, "report_writer", None,
                                            build_report_instruction(state) + "\n\nWrite the full markdown report now.",
                                            [], session_id, 2)
                metrics.track_llm_call(rep.latency_ms, rep.input_tokens, rep.output_tokens,
                                       model=self.config.report.model_id)
                report = finalize_report(rep.text, state) if rep.text.strip() else ""
            except Exception as exc:
                logger.warning("[%s] report agent failed, using template: %s", investigation_id, exc)
                yield _trace("report_generation", "error", {"error": f"Report model failed; template used: {exc}"[:300]})
            state["report_markdown"] = report or generate_fallback_report(state)
            # Send the report as soon as it exists: the UI closes its stream when a pause
            # arrives, so anything sent after the pause event never reaches it.
            yield {"type": "state_update", "data": {**self._ui_state(state), "current_phase": "report"}}
            yield _trace("report_generation", "node_complete", {"status": "success"})
            metrics.end_node("report_generation")

            # 6. action — destructive decisions pause for the analyst
            async for ev in self._act(state, user_id, investigation_id, approval=None):
                yield ev

        except Exception as exc:
            logger.error("Strands workflow error: %s\n%s", exc, traceback.format_exc())
            yield {"type": "metrics", "investigation_id": investigation_id, "data": metrics.get_metrics()}
            yield {"type": "error", "investigation_id": investigation_id, "user_id": user_id, "error": str(exc)}
            remove_collector(investigation_id)

    async def _act(self, state: Dict[str, Any], user_id: str, investigation_id: str,
                   approval: Optional[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        assessment = state.get("final_assessment") or {}
        decision = (assessment.get("decision") or "allow_monitor").strip()
        account_id = assessment.get("account_id") or next(iter((state.get("initial_evidence") or {}).get("accounts") or {}), "")
        reason = (assessment.get("reasoning") or "")[:300]
        enacted: List[Dict[str, Any]] = list(state.get("enacted_actions") or [])

        if approval is None:
            yield _trace("llm_agent", "action_proposed", {"decision": decision, "account_id": account_id, "reason": reason})
            if decision in DESTRUCTIVE_DECISIONS:
                payload = {
                    "investigation_id": investigation_id, "user_id": user_id, "decision": decision,
                    "account_id": account_id, "reason": reason, "fc_id": PAUSE_FC_ID, "current_step": "llm_agent",
                    "hint": (f"The AI agent recommends **{decision.replace('_', ' ')}** on account {account_id}. "
                             f"Reason: {reason}. Approve this action?"),
                }
                await self._save_pause(investigation_id, state)
                # State (with the report) BEFORE the pause event: the UI stops reading at the pause.
                yield {"type": "state_update", "data": {**self._ui_state(state), "current_phase": "awaiting_decision"}}
                yield _trace("llm_agent", "action_confirmation_required", payload)
                yield {"type": "action_confirmation_required", "data": payload}
                yield {"type": "_paused", "data": payload}
                return
            if decision in ALL_DECISIONS:
                enacted.append(await asyncio.to_thread(execute_action, decision, account_id, reason))
        else:
            override = approval.get("override")
            if not approval.get("approved") and override:
                enacted.append(await asyncio.to_thread(execute_action, override, account_id,
                                                       "Analyst override of the agent's recommendation"))
            elif approval.get("approved") and decision in ALL_DECISIONS:
                enacted.append(await asyncio.to_thread(execute_action, decision, account_id, reason))

        state["enacted_actions"] = enacted
        async for ev in self._finish(state, user_id, investigation_id):
            yield ev

    async def resume_investigation(self, user_id: str, investigation_id: str, fc_id: str, approved: bool,
                                   hint: str = "", payload: Optional[Dict[str, Any]] = None,
                                   override: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if self._storage is None and self._aerospike_ready():
            await self.initialize()
        yield _trace("llm_agent", "action_decision", {"approved": bool(approved), "decision": (payload or {}).get("decision"),
                                                     "account_id": (payload or {}).get("account_id"), "override": override})
        state = await self._load_pause(investigation_id)
        if state is None:
            yield {"type": "error", "investigation_id": investigation_id, "user_id": user_id,
                   "error": "Paused investigation state not found in Aerospike; re-run the investigation."}
            return
        async for ev in self._act(state, user_id, investigation_id,
                                  approval={"approved": bool(approved), "override": override}):
            yield ev
        await self._delete_pause(investigation_id)

    async def _finish(self, state: Dict[str, Any], user_id: str, investigation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            await self._store_case(state, user_id, investigation_id)
        except Exception as exc:
            logger.warning("[%s] case memory store failed: %s", investigation_id, exc)
        yield {"type": "state_update", "data": {**self._ui_state(state), "enacted_actions": state.get("enacted_actions", []),
                                                "current_phase": "report", "workflow_status": "completed"}}
        metrics = get_collector(investigation_id)
        yield {"type": "metrics", "investigation_id": investigation_id, "data": metrics.get_metrics()}
        yield {"type": "complete", "investigation_id": investigation_id, "user_id": user_id}
        remove_collector(investigation_id)

    @staticmethod
    def _ui_state(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_assessment": state.get("final_assessment"),
            "report_markdown": state.get("report_markdown", ""),
            "specialist_findings": specialist_findings(state),
            "tool_calls": merged_tool_calls(state),
            "agent_iterations": state.get("agent_iterations", 0),
            "initial_evidence": state.get("initial_evidence"),
            "prior_cases": state.get("prior_cases", []),
        }

    # -- pause snapshot (AerospikeStorage) -----------------------------------------------------
    @staticmethod
    def _pause_key(investigation_id: str) -> str:
        return f"investigations/{_session_id(investigation_id)}/paused.json"

    async def _save_pause(self, investigation_id: str, state: Dict[str, Any]) -> None:
        if self._storage is None:
            raise RuntimeError("Cannot pause for approval: Aerospike storage unavailable")
        keep = {k: v for k, v in state.items() if k != "trace_events"}
        await self._storage.write(self._pause_key(investigation_id), json.dumps(keep, default=str).encode())

    async def _load_pause(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        if self._storage is None:
            return None
        raw = await self._storage.read(self._pause_key(investigation_id))
        return json.loads(raw) if raw else None

    async def _delete_pause(self, investigation_id: str) -> None:
        if self._storage is not None:
            try:
                await self._storage.delete(self._pause_key(investigation_id))
            except Exception as exc:
                logger.debug("pause cleanup failed: %s", exc)

    # -- cross-case memory (AerospikeMemoryStore) ----------------------------------------------
    @staticmethod
    def _entities(user_id: str, state: Dict[str, Any]) -> List[str]:
        initial = state.get("initial_evidence") or {}
        ids = [user_id, *(initial.get("accounts") or {}).keys(), *(initial.get("devices") or {}).keys()]
        return [i for i in dict.fromkeys(ids) if i]

    @staticmethod
    def _counterparties(state: Dict[str, Any]) -> List[str]:
        out = set()
        for tc in merged_tool_calls(state):
            if tc.get("tool") in ("get_counterparty_profile", "get_counterparty_transactions"):
                uid = (tc.get("params") or {}).get("user_id")
                if uid:
                    out.add(uid)
            elif tc.get("tool") == "get_account_transactions":
                for txn in ((tc.get("result") or {}).get("transactions") or []):
                    if txn.get("counterparty_user_id"):
                        out.add(txn["counterparty_user_id"])
        return sorted(out)

    async def _recall_cases(self, state: Dict[str, Any], user_id: str, investigation_id: str) -> List[Dict[str, Any]]:
        if self._memory is None:
            return []
        query_entities = self._entities(user_id, state)
        try:
            entries = await self._memory.search(" ".join(query_entities), {"max_search_results": 20})
        except Exception as exc:
            logger.warning("[%s] memory recall failed: %s", investigation_id, exc)
            return []
        want, out, seen = {e.lower(): e for e in query_entities}, [], set()
        for entry in entries:
            case = entry.metadata or {}
            inv = case.get("investigation_id")
            if not inv or inv in seen or inv == investigation_id or case.get("user_id") == user_id:
                continue
            matched = sorted(want[e.lower()] for e in case.get("entities") or [] if e.lower() in want)
            if not matched:
                continue
            seen.add(inv)
            out.append({k: case.get(k) for k in ("investigation_id", "account_id", "user_id", "holder",
                                                 "typology", "decision", "status")} | {"matched_on": matched})
            if len(out) >= 5:
                break
        return out

    async def _store_case(self, state: Dict[str, Any], user_id: str, investigation_id: str) -> None:
        if self._memory is None:
            return
        initial = state.get("initial_evidence") or {}
        assessment = state.get("final_assessment") or {}
        enacted = state.get("enacted_actions") or []
        profile = initial.get("profile") or {}
        entities = list(dict.fromkeys(self._entities(user_id, state) + self._counterparties(state)))
        case = {
            "investigation_id": investigation_id, "user_id": user_id,
            "account_id": assessment.get("account_id") or next(iter(initial.get("accounts") or {}), None),
            "holder": profile.get("name") or profile.get("account_holder") or "",
            "typology": assessment.get("typology"),
            "decision": enacted[0].get("action") if enacted else assessment.get("decision"),
            "status": "completed", "entities": entities,
        }
        content = (f"Case {investigation_id}\n"
                   f"entities: {' '.join(entities)}\n"
                   f"holder: {case['holder']} | typology: {case['typology']} | decision: {case['decision']}")
        await self._memory.add(content, metadata=case)
