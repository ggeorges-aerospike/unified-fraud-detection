"""AWS Strands engine: pipeline, SSE contract, pause/resume, budgets and memory — no Bedrock calls.

The engine's agent runner is replaced by a scripted fake that calls the real Strands @tool
functions the engine built, so tool wiring, recording and budgets are exercised for real.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

import workflow.engines.strands_engine as se
from workflow.engines.strands_engine import AgentRun, RoleModel, StrandsConfig, StrandsEngine, short_model_name

CONFIG = StrandsConfig(
    region="us-east-1",
    specialist=RoleModel("us.anthropic.claude-sonnet-5-5", "medium"),
    investigator=RoleModel("us.anthropic.claude-opus-5-5", "high"),
    report=RoleModel("us.anthropic.claude-opus-5-5", "medium"),
    max_tokens=16000,
)


# --------------------------------------------------------------------------- fakes
class FakeTools:
    def __init__(self, aerospike, graph, user_id, metrics=None):
        self.user_id = user_id

    def execute_tool(self, name, params):
        return {"success": True, "tool": name, "params": params}


class FakeStorage:
    def __init__(self):
        self.data: Dict[str, bytes] = {}

    async def write(self, key, value):
        self.data[key] = value

    async def read(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)


class Entry:
    def __init__(self, content, metadata):
        self.content, self.metadata = content, metadata


class FakeMemory:
    def __init__(self, entries=None):
        self.added: List[Any] = []
        self.entries = entries or []

    async def add(self, content, metadata=None):
        self.added.append((content, metadata))
        return "k"

    async def search(self, query, options=None):
        return list(self.entries)


def tools_by_name(tools):
    return {t.tool_name: t for t in tools}


def scripted_runner(decision="allow_monitor", fail: Optional[set] = None, submit=True, calls=None):
    """Specialists call each allowed tool once; the investigator submits ``decision``."""
    fail = fail or set()
    seen: List[str] = []

    async def run(role, name, system, prompt, tools, session_id, max_turns):
        seen.append(name)
        if name in fail:
            raise RuntimeError(f"{name} boom")
        t = tools_by_name(tools)
        if name == "investigator":
            t["get_account_risk_features"](account_id="A1")
            if submit:
                t["submit_assessment"](typology="money_mule", risk_level="high", risk_score=82,
                                       decision=decision, account_id="", reasoning="fan-out to 39 new recipients")
            return AgentRun("done", "end_turn", 1200.0, 3000, 400)
        if name == "report_writer":
            assert tools == []
            return AgentRun("# Report\nAll good.", "end_turn", 900.0, 2000, 800)
        for tname, tool in list(t.items())[: (calls or {}).get(name, 1)]:
            kwargs = {"account_id": "A1"} if "account" in tname else {"user_id": "U2"} if "counterparty" in tname \
                else {"device_id": "D1"} if "device" in tname else {"hops": 1}
            tool(**kwargs)
        return AgentRun(f"- {name}: finding with numbers", "end_turn", 800.0, 1500, 200)

    run.seen = seen
    return run


@pytest.fixture
def engine(monkeypatch, fake_aerospike, fake_graph):
    monkeypatch.setattr(se, "InvestigationTools", FakeTools)
    monkeypatch.setattr(se, "alert_validation_node", lambda s, a: {"alert_evidence": {"flag_reason": "velocity"}})
    monkeypatch.setattr(se, "data_collection_node", lambda s, a, g: {
        "initial_evidence": {"profile": {"name": "Ana Lima"}, "accounts": {"A1": {}}, "devices": {"D1": {}}}})
    monkeypatch.setattr(se, "build_evidence_summary", lambda i, a: "EVIDENCE")
    monkeypatch.setattr(se, "build_report_instruction", lambda s: "WRITE REPORT")
    monkeypatch.setattr(se, "finalize_report", lambda raw, s: raw)
    actions: List[tuple] = []
    monkeypatch.setattr(se, "execute_action", lambda d, a, r: actions.append((d, a, r)) or {"action": d, "account_id": a})
    eng = StrandsEngine(fake_aerospike, fake_graph, config=CONFIG,
                        agent_runner=scripted_runner(calls={"network_analyst": 4}))
    eng._storage, eng._memory = FakeStorage(), FakeMemory()
    eng.actions = actions
    return eng


async def collect(agen):
    return [e async for e in agen]


def traces(events, type_):
    return [e["event"] for e in events if e["type"] == "trace" and e["event"]["type"] == type_]


# --------------------------------------------------------------------------- tests
async def test_full_run_non_destructive_completes_and_stores_case(engine):
    events = await collect(engine.run_investigation("U1", "inv-1"))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "complete" and "error" not in kinds and "_paused" not in kinds
    steps = [e["node"] for e in traces(events, "node_start")]
    assert steps == ["alert_validation", "data_collection", "llm_agent", "report_generation"]
    agents = {e["data"]["agent"] for e in traces(events, "tool_call")}
    assert agents == {"network_analyst", "device_analyst", "velocity_analyst", "investigator"}
    assert len(traces(events, "specialist_finding")) == 3
    final = [e for e in events if e["type"] == "state_update"][-1]["data"]
    assert final["workflow_status"] == "completed"
    assert final["final_assessment"]["decision"] == "allow_monitor"
    assert final["final_assessment"]["account_id"] == "A1"          # filled from evidence
    assert final["report_markdown"].startswith("# Report")
    assert set(final["specialist_findings"]) == {"network_analyst", "device_analyst", "velocity_analyst"}
    assert all(final["specialist_findings"].values())
    assert engine.actions == [("allow_monitor", "A1", "fan-out to 39 new recipients")]
    [(content, case)] = engine._memory.added
    assert content.startswith("Case inv-1") and case["decision"] == "allow_monitor"
    assert {"U1", "A1", "D1", "U2"} <= set(case["entities"])          # counterparty U2 included


async def test_destructive_decision_pauses_then_resumes_with_approval(engine):
    engine._run_agent = scripted_runner(decision="full_block")
    events = await collect(engine.run_investigation("U1", "inv-2"))
    assert events[-1]["type"] == "_paused"
    payload = events[-1]["data"]
    assert (payload["decision"], payload["fc_id"]) == ("full_block", se.PAUSE_FC_ID)
    assert engine.actions == []
    assert engine._storage.data, "paused state must be saved for resume"

    resumed = await collect(engine.resume_investigation("U1", "inv-2", se.PAUSE_FC_ID, approved=True,
                                                        payload=payload))
    assert resumed[-1]["type"] == "complete"
    assert engine.actions == [("full_block", "A1", "fan-out to 39 new recipients")]
    assert not engine._storage.data, "pause snapshot cleaned up"


async def test_rejected_with_override_enacts_override(engine):
    engine._run_agent = scripted_runner(decision="temporary_freeze")
    await collect(engine.run_investigation("U1", "inv-3"))
    await collect(engine.resume_investigation("U1", "inv-3", se.PAUSE_FC_ID, approved=False,
                                              override="step_up_auth"))
    assert engine.actions == [("step_up_auth", "A1", "Analyst override of the agent's recommendation")]


async def test_resume_without_snapshot_is_an_error(engine):
    events = await collect(engine.resume_investigation("U1", "missing", se.PAUSE_FC_ID, approved=True))
    assert events[-1]["type"] == "error"


async def test_specialist_failure_is_reported_and_others_continue(engine):
    engine._run_agent = scripted_runner(fail={"device_analyst"})
    events = await collect(engine.run_investigation("U1", "inv-4"))
    assert events[-1]["type"] == "complete"
    errs = traces(events, "error")
    assert errs and errs[0]["data"]["agent"] == "device_analyst"
    final = [e for e in events if e["type"] == "state_update"][-1]["data"]
    assert "could not finish" in final["specialist_findings"]["device_analyst"]


async def test_investigator_failure_is_an_error_not_a_fake_decision(engine):
    engine._run_agent = scripted_runner(fail={"investigator"})
    events = await collect(engine.run_investigation("U1", "inv-5"))
    assert events[-1]["type"] == "error" and "investigator boom" in events[-1]["error"]
    assert engine.actions == []


async def test_missing_assessment_falls_back_visibly(engine, monkeypatch):
    monkeypatch.setattr(se, "deterministic_assessment", lambda i, a: {
        "typology": "suspicious_activity", "risk_level": "medium", "risk_score": 50,
        "decision": "allow_monitor", "account_id": "", "reasoning": "rules"})
    engine._run_agent = scripted_runner(submit=False)
    events = await collect(engine.run_investigation("U1", "inv-6"))
    assert any("No assessment submitted" in e["data"]["error"] for e in traces(events, "error"))
    assert events[-1]["type"] == "complete"


def test_tool_budget_and_submit_validation():
    events = []
    rec = se.ToolRecorder("device_analyst", "llm_agent", FakeTools(None, None, "U1"),
                          type("M", (), {"track_tool_call": lambda self, n: None})(), 2, events.append)
    tools = tools_by_name(se.build_strands_tools(rec, ["get_device_risk_features"], include_submit=True))
    assert set(tools) == {"get_device_risk_features", "submit_assessment"}
    for _ in range(3):
        out = tools["get_device_risk_features"](device_id="D1")
    assert out["success"] is False and "budget" in out["error"]
    assert len(rec.tool_calls) == 2
    bad = tools["submit_assessment"](typology="x", risk_level="high", risk_score=500, decision="nuke",
                                     account_id="A1", reasoning="r")
    assert bad["accepted"] is False and rec.assessment is None
    ok = tools["submit_assessment"](typology="x", risk_level="HIGH", risk_score=500, decision="full_block",
                                    account_id="A1", reasoning="r")
    assert ok["accepted"] and rec.assessment["risk_score"] == 100 and rec.assessment["risk_level"] == "high"


def test_tool_specs_have_descriptions_and_typed_params():
    rec = se.ToolRecorder("x", "llm_agent", FakeTools(None, None, "U1"), None, 1, lambda e: None)
    tools = tools_by_name(se.build_strands_tools(rec, se.INVESTIGATOR_TOOLS, include_submit=True))
    spec = tools["get_account_transactions"].tool_spec
    props = spec["inputSchema"]["json"]["properties"]
    assert props["account_id"]["type"] == "string" and props["days"]["type"] == "integer"
    assert "account_id" in spec["inputSchema"]["json"]["required"]
    assert "ledger" in spec["description"]


async def test_memory_recall_filters_self_and_matches_entities(engine):
    engine._memory = FakeMemory([
        Entry("Case old-1", {"investigation_id": "old-1", "user_id": "U9", "account_id": "A9", "decision": "full_block",
                             "typology": "money_mule", "entities": ["U9", "A9", "a1"]}),
        Entry("Case own", {"investigation_id": "own", "user_id": "U1", "entities": ["U1", "A1"]}),
        Entry("Case unrelated", {"investigation_id": "x", "user_id": "U8", "entities": ["U8"]}),
    ])
    events = await collect(engine.run_investigation("U1", "inv-7"))
    [recall] = traces(events, "memory_recall")
    [case] = recall["data"]["prior_cases"]
    assert case["investigation_id"] == "old-1" and case["matched_on"] == ["A1"]


def test_identity_and_registry(fake_aerospike, fake_graph):
    eng = StrandsEngine(fake_aerospike, fake_graph, config=CONFIG)
    assert eng.engine_name == "strands"
    assert eng.describe() == {"engine": "strands", "label": "AWS Strands", "model": "Sonnet 5.5 · Opus 5.5"}
    assert short_model_name("us.anthropic.claude-haiku-5-5") == "Haiku 5.5"
    assert short_model_name("anthropic.claude-opus-5") == "Opus 5"
    from workflow.engines import SUPPORTED_ENGINES, get_engine
    assert "strands" in SUPPORTED_ENGINES
    assert get_engine("strands", fake_aerospike, fake_graph).engine_name == "strands"


def test_config_from_env(monkeypatch):
    for k in ("STRANDS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION", "STRANDS_INVESTIGATOR_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = StrandsConfig.from_env()
    assert cfg.region == "us-east-1" and cfg.specialist.model_id.endswith("sonnet-5-5")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("STRANDS_INVESTIGATOR_MODEL", "us.anthropic.claude-sonnet-5-5")
    cfg = StrandsConfig.from_env()
    assert cfg.region == "us-west-2" and cfg.investigator.model_id.endswith("sonnet-5-5")


async def test_session_created_once_before_parallel_specialists(engine):
    """Three specialists sharing one Aerospike session must not race to create it."""
    created, order = [], []
    engine._session_manager = lambda sid: created.append(sid)
    inner = engine._run_agent

    async def tracking(role, name, *a):
        order.append((name, len(created)))
        return await inner(role, name, *a)

    engine._run_agent = tracking
    await collect(engine.run_investigation("U1", "inv:8"))
    assert created == ["inv_8"]                                   # once, with a colon-free id
    assert all(n_created == 1 for _, n_created in order)          # before any agent ran


async def test_report_reaches_the_ui_before_the_pause(engine):
    """The UI closes its stream on action_confirmation_required; the report must arrive first."""
    engine._run_agent = scripted_runner(decision="full_block")
    events = await collect(engine.run_investigation("U1", "inv-9"))
    pause_at = next(i for i, e in enumerate(events) if e["type"] == "action_confirmation_required")
    before = [e for e in events[:pause_at] if e["type"] == "state_update"]
    assert before and before[-1]["data"]["report_markdown"].startswith("# Report")
    assert before[-1]["data"]["final_assessment"]["decision"] == "full_block"
