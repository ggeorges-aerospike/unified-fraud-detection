"""Tier-2 decision-model triage: truth table, mock/live client contracts, and the service write path."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pytest

from services.triage import (
    AUTO_CLEAR,
    ESCALATE,
    ESCALATE_URGENT,
    PROVIDERS,
    DecisionModelError,
    Facts,
    MockDecisionModel,
    SystemOneHttpModel,
    TriageService,
    decide,
    model_from_env,
)
from services.triage.questions import AUTO_CLEAR_MAX_RISK, QUESTIONS, SCAM_PATTERNS, request_questions


# --------------------------------------------------------------------------- helpers
def nouls(**p: float) -> Dict[str, Any]:
    base = {
        "email_matches_name": 0.95,
        "recipient_is_customer_alias": 0.05,
        "impossible_travel": 0.05,
        "activity_fits_profile": 0.9,
        "fraud_pattern_present": 0.05,
    }
    base.update(p)
    ans: Dict[str, Any] = {k: {"type": "noul", "noul": v} for k, v in base.items()}
    ans["scam_pattern"] = {"type": "choice", "choice": "mule_fan_out", "confidence": 0.6,
                           "probabilities": {k: 0.25 for k in SCAM_PATTERNS}}
    return ans


# --------------------------------------------------------------------------- truth table
class TestTruthTable:
    def test_benign_clears(self):
        d = decide(nouls(), Facts(risk_score=55, within_bank_norms=True))
        assert (d.outcome, d.rule) == (AUTO_CLEAR, "benign_activity")
        assert d.pattern is None

    def test_outside_bank_norms_blocks_auto_clear(self):
        for within in (False, None):
            d = decide(nouls(), Facts(risk_score=55, within_bank_norms=within, norms_detail="39 distinct recipients"))
            assert (d.outcome, d.rule) == (ESCALATE, "outside_bank_norms")

    def test_norms_check_catches_velocity_burst(self):
        from services.triage.state import norms_check
        burst = {"transfers_sent": 39, "distinct_recipients": 39, "largest_transfer_usd": 4794.75,
                 "days_between_first_and_last": 24.8}
        ok, detail = norms_check(burst)
        assert not ok and "39 distinct recipients" in detail and "transfers a week" in detail
        assert norms_check({"transfers_sent": 2, "distinct_recipients": 2, "largest_transfer_usd": 4200,
                            "days_between_first_and_last": 9})[0] is True

    def test_score_guardrail_blocks_auto_clear(self):
        d = decide(nouls(), Facts(risk_score=AUTO_CLEAR_MAX_RISK, within_bank_norms=True))
        assert (d.outcome, d.rule) == (ESCALATE, "score_guardrail")

    def test_device_tied_to_confirmed_fraud_is_urgent_regardless_of_answers(self):
        d = decide(nouls(), Facts(risk_score=50, device_tied_to_confirmed_fraud=True))
        assert (d.outcome, d.rule) == (ESCALATE_URGENT, "device_confirmed_fraud")

    def test_pattern_plus_flagged_recipient_is_urgent(self):
        d = decide(nouls(fraud_pattern_present=0.9), Facts(risk_score=50, flagged_recipients=2))
        assert (d.outcome, d.rule) == (ESCALATE_URGENT, "pattern_with_flagged_recipients")
        assert d.pattern == "mule_fan_out"

    @pytest.mark.parametrize("extra", [
        {"recipient_is_customer_alias": 0.9},
        {"impossible_travel": 0.9},
        {"email_matches_name": 0.1},
    ])
    def test_pattern_corroborated_is_urgent(self, extra):
        d = decide(nouls(fraud_pattern_present=0.9, **extra), Facts(risk_score=50))
        assert (d.outcome, d.rule) == (ESCALATE_URGENT, "pattern_corroborated")

    def test_pattern_alone_escalates(self):
        d = decide(nouls(fraud_pattern_present=0.9), Facts(risk_score=50))
        assert (d.outcome, d.rule) == (ESCALATE, "pattern_uncorroborated")

    def test_unsettled_answer_never_clears(self):
        d = decide(nouls(activity_fits_profile=0.5), Facts(risk_score=50))
        assert (d.outcome, d.rule) == (ESCALATE, "unsettled")
        assert d.readings["activity_fits_profile"] == "unsettled"

    def test_missing_answers_escalate(self):
        d = decide({}, Facts(risk_score=10))
        assert d.outcome == ESCALATE

    def test_flagged_recipient_without_pattern_blocks_clear(self):
        d = decide(nouls(), Facts(risk_score=50, flagged_recipients=1))
        assert d.outcome == ESCALATE

    def test_choice_ignored_without_companion_noul(self):
        d = decide(nouls(fraud_pattern_present=0.5), Facts(risk_score=50))
        assert d.pattern is None


def test_model_specific_bands_change_readings():
    from services.triage.questions import thresholds_for
    ans = nouls(fraud_pattern_present=0.5, activity_fits_profile=0.42, impossible_travel=0.4)
    assert decide(ans, Facts(risk_score=50)).outcome == ESCALATE                      # generic bands: unsettled
    d = decide(ans, Facts(risk_score=50, within_bank_norms=True), thresholds_for("decider"))
    assert (d.outcome, d.rule) == (AUTO_CLEAR, "benign_activity")                       # decider bands settle it
    for q, (no_at, yes_at) in thresholds_for("decider").items():
        assert 0 <= no_at < yes_at <= 1, q


def test_failed_model_call_never_auto_clears_under_any_bands():
    from services.triage.questions import MODEL_THRESHOLDS, thresholds_for
    for mode in ["default", *MODEL_THRESHOLDS]:
        assert decide({}, Facts(risk_score=10), thresholds_for(mode)).outcome == ESCALATE


def test_no_question_asks_for_the_decision():
    """Guidance: judgments from the model, decisions in code."""
    for q in QUESTIONS.values():
        text = q["instructions"].lower()
        assert "escalat" not in text and "is this fraud" not in text and "should" not in text


# --------------------------------------------------------------------------- mock contract
def _state(**over) -> Dict[str, Any]:
    now = datetime(2026, 11, 11, 12, 0)
    s = {
        "customer": {"name": "Kaitlyn Castaneda PhD", "email": "kaitlyn.castaneda.phd@chapman.com",
                     "age": 54, "occupation": "Accountant", "home_city": "San Diego"},
        "accounts": [{"bank": "Regions", "type": "checking", "days_old": 700}],
        "devices": [{"description": "Safari on iOS (mobile)", "first_seen_days_ago": 400}],
        "recent_transfers": [
            {"sent_at": (now - timedelta(days=i * 4)).isoformat(), "amount_usd": 120.0,
             "recipient": "Landlord LLC", "from_city": "San Diego, CA", "device": "Safari on iOS"}
            for i in range(4)
        ],
        "recipients": [{"name": "Landlord LLC", "times_paid": 4, "total_usd": 480.0, "times_received_from": 0}],
        "known_fraud_patterns": SCAM_PATTERNS,
    }
    s.update(over)
    return s


class TestMockDecisionModel:
    def test_answers_match_openapi_shapes(self):
        resp = MockDecisionModel().ask(_state(), request_questions())
        assert set(resp) == {"model", "answers", "usage"}
        assert set(resp["usage"]) == {"input_tokens", "output_tokens"}
        for name, q in QUESTIONS.items():
            a = resp["answers"][name]
            assert a["type"] == q["type"]
            if q["type"] == "noul":
                assert 0 <= a["noul"] <= 1
            else:
                assert set(a) == {"type", "choice", "confidence", "probabilities"}
                assert set(a["probabilities"]) == set(SCAM_PATTERNS)
                assert abs(sum(a["probabilities"].values()) - 1) < 0.01
                assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)

    def test_quiet_customer_auto_clears(self):
        resp = MockDecisionModel().ask(_state(), request_questions())
        assert decide(resp["answers"], Facts(risk_score=50, within_bank_norms=True)).outcome == AUTO_CLEAR

    def test_email_name_match_and_mismatch(self):
        m = MockDecisionModel()
        assert m._email_matches_name(_state()) >= 0.7
        bad = _state(customer={"name": "Kaitlyn Castaneda", "email": "xq7z.trader99@mail.com"})
        assert m._email_matches_name(bad) <= 0.3

    def test_recipient_alias_fuzzy_match(self):
        s = _state(recipients=[{"name": "Katlyn Castaneda", "times_paid": 3, "total_usd": 9000}])
        assert MockDecisionModel()._recipient_is_customer_alias(s) >= 0.7
        assert MockDecisionModel()._recipient_is_customer_alias(_state()) <= 0.3

    def test_impossible_travel(self):
        t = datetime(2026, 11, 11, 12, 0)
        hops = [{"sent_at": (t + timedelta(minutes=40 * i)).isoformat(), "amount_usd": 900,
                 "recipient": f"R{i}", "from_city": c} for i, c in
                enumerate(["Miami, FL", "Seattle, WA", "Boston, MA", "Denver, CO"])]
        assert MockDecisionModel()._impossible_travel(_state(recent_transfers=hops)) >= 0.7
        assert MockDecisionModel()._impossible_travel(_state()) <= 0.3

    def test_two_way_flows_flag_ring(self):
        t = datetime(2026, 11, 11, 12, 0)
        tx = [{"sent_at": (t + timedelta(hours=5 * i)).isoformat(), "amount_usd": 4000,
               "recipient": f"Member {i % 3}", "from_city": "San Diego, CA"} for i in range(9)]
        recips = [{"name": f"Member {i}", "times_paid": 3, "total_usd": 12000, "times_received_from": 3} for i in range(3)]
        resp = MockDecisionModel().ask(_state(recent_transfers=tx, recipients=recips), request_questions())
        assert resp["answers"]["scam_pattern"]["choice"] == "circular_ring"
        assert decide(resp["answers"], Facts(risk_score=60)).readings["fraud_pattern_present"] == "yes"

    def test_mule_burst_flags_pattern(self):
        t = datetime(2026, 11, 11, 12, 0)
        burst = [{"sent_at": (t + timedelta(minutes=20 * i)).isoformat(), "amount_usd": 3000,
                  "recipient": f"Person {i}", "from_city": "San Diego, CA"} for i in range(25)]
        recips = [{"name": f"Person {i}", "times_paid": 1, "total_usd": 3000} for i in range(25)]
        resp = MockDecisionModel().ask(_state(recent_transfers=burst, recipients=recips), request_questions())
        d = decide(resp["answers"], Facts(risk_score=60))
        assert d.readings["fraud_pattern_present"] == "yes"
        assert d.pattern == "mule_fan_out"
        assert d.outcome in (ESCALATE, ESCALATE_URGENT)


# --------------------------------------------------------------------------- live client
class TestSystemOneHttpModel:
    def _client(self, handler, provider="jev", **kw) -> SystemOneHttpModel:
        return SystemOneHttpModel(PROVIDERS[provider], transport=httpx.MockTransport(handler), max_attempts=3, **kw)

    def test_jev_sends_openapi_request_with_key_and_model(self):
        seen: List[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(200, json={"model": "jev-1.14", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}})

        m = self._client(handler, api_key="k-test")
        out = m.ask({"a": 1}, {"q": {"type": "noul", "instructions": "x?"}})
        assert out["model"] == "jev-1.14" and m.name == "jev-1.14"
        req = seen[0]
        assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
        assert req.headers["authorization"] == "Bearer k-test"
        assert json.loads(req.content) == {"model": "jev-latest", "state": {"a": 1},
                                           "questions": {"q": {"type": "noul", "instructions": "x?"}}}

    @pytest.mark.parametrize("provider,url", [
        ("decider", "http://host.docker.internal:8000/v1/systemone"),
        ("kev", "http://host.docker.internal:8009/v1/systemone"),
    ])
    def test_self_hosted_needs_no_key_and_lets_server_pick_model(self, provider, url):
        seen: List[httpx.Request] = []

        def handler(req):
            seen.append(req)
            return httpx.Response(200, json={"model": f"{provider}-served", "answers": {},
                                             "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_ms": 12.5})

        m = self._client(handler, provider=provider)
        m.ask({"a": 1}, {"q": {"type": "noul", "instructions": "x?"}})
        assert str(seen[0].url) == url
        assert "authorization" not in seen[0].headers
        assert "model" not in json.loads(seen[0].content)
        assert m.name == f"{provider}-served"

    def test_retries_5xx_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("services.triage.decision_models.time.sleep", lambda s: None)
        calls = iter([503, 200])

        def handler(req):
            code = next(calls)
            return httpx.Response(code, json={"model": "m", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}})

        assert self._client(handler, api_key="k")._max_attempts == 3
        assert self._client(handler, api_key="k").ask({}, {})["model"] == "m"

    def test_4xx_raises_without_retry(self):
        n = {"calls": 0}

        def handler(req):
            n["calls"] += 1
            return httpx.Response(422, json={"detail": []})

        with pytest.raises(DecisionModelError):
            self._client(handler, api_key="k").ask({}, {})
        assert n["calls"] == 1

    def test_connection_refused_raises_model_error(self, monkeypatch):
        monkeypatch.setattr("services.triage.decision_models.time.sleep", lambda s: None)

        def handler(req):
            raise httpx.ConnectError("refused")

        with pytest.raises(DecisionModelError, match="unavailable"):
            self._client(handler, provider="decider").ask({}, {})

    def test_jev_requires_key(self):
        with pytest.raises(ValueError):
            SystemOneHttpModel(PROVIDERS["jev"])


class TestModelFromEnv:
    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch):
        for k in ("DECISION_MODEL", "DECISION_MODEL_URL", "DECISION_MODEL_NAME", "DECISION_MODEL_API_KEY",
                  "JEV_MODE", "JEV_MODEL", "TYPESAFE_API_KEY"):
            monkeypatch.delenv(k, raising=False)

    def test_default_is_mock(self):
        assert model_from_env().mode == "mock"

    def test_decider_preset_and_overrides(self, monkeypatch):
        monkeypatch.setenv("DECISION_MODEL", "decider")
        m = model_from_env()
        assert (m.mode, m.label, m.base_url) == ("decider", "Strands Decider", "http://host.docker.internal:8000")
        monkeypatch.setenv("DECISION_MODEL_URL", "https://decider.example.com")
        monkeypatch.setenv("DECISION_MODEL_NAME", "strands-decider-12B-gemma4-v1-2610")
        m = model_from_env()
        assert m.base_url == "https://decider.example.com" and m.name == "strands-decider-12B-gemma4-v1-2610"

    def test_legacy_jev_mode_live_maps_to_jev(self, monkeypatch):
        monkeypatch.setenv("JEV_MODE", "live")
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        assert model_from_env().mode == "jev"

    def test_off_and_unknown(self, monkeypatch):
        monkeypatch.setenv("DECISION_MODEL", "off")
        assert model_from_env() is None
        monkeypatch.setenv("DECISION_MODEL", "gpt")
        with pytest.raises(ValueError):
            model_from_env()


# --------------------------------------------------------------------------- service write path
class FakeAerospike:
    """Just enough of AerospikeService for the triage path, in memory."""

    namespace = "test"

    def __init__(self):
        self.sets: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.txns: Dict[str, List[Dict[str, Any]]] = {}

    def get(self, set_name, key):
        rec = self.sets.get(set_name, {}).get(key)
        return dict(rec) if rec else None

    def put(self, set_name, key, data, ttl=0):
        self.sets.setdefault(set_name, {}).setdefault(key, {}).update(
            {k: v for k, v in data.items() if v is not None})
        return True

    def batch_get(self, keys):
        out = []
        for ns, s, k in keys:
            rec = self.get(s, k)
            out.append(((ns, s, k), {}, rec) if rec else None)
        return out

    def batch_get_transactions(self, account_ids, days=7):
        return {a: list(self.txns.get(a, [])) for a in account_ids}

    def update_flagged_account(self, key, updates):
        return self.put("flagged_accounts", key, updates)

    def scan_all(self, set_name, limit=10000):
        return list(self.sets.get(set_name, {}).values())[:limit]

    def _query_equals(self, set_name, bin_name, value, limit=10000):
        return [dict(r) for r in self.sets.get(set_name, {}).values() if r.get(bin_name) == value][:limit]


def _seed(fake: FakeAerospike, user_id="U1", device_fraud=False):
    fake.put("users", user_id, {
        "user_id": user_id, "name": "Ana Lima", "email": "ana.lima@example.com", "age": 40,
        "occupation": "Teacher", "location": "Austin", "signup_date": "2024-01-01T00:00:00Z",
        "accounts": {"A1": {"bank_name": "Chase", "type": "checking", "created_date": "2024-01-01T00:00:00Z"}},
        "devices": {"D1": {"browser": "Chrome", "os": "macOS", "type": "desktop", "first_seen": "2024-02-01T00:00:00Z"}},
    })
    fake.put("users", "U2", {"user_id": "U2", "name": "Rent Co"})
    fake.put("device_fact", "D1", {"fraud": device_fraud})
    fake.put("flagged_accounts", user_id, {"user_id": user_id, "status": "pending_review",
                                            "risk_score": 52, "risk_factors": ["Many recipients"]})
    now = datetime.now()
    fake.txns["A1"] = [
        {"direction": "out", "timestamp": (now - timedelta(days=5 * i)).isoformat(), "amount": 90.0,
         "counterparty_user_id": "U2", "location": "Austin, TX", "device_id": "D1"}
        for i in range(3)
    ]


class TestTriageService:
    def test_auto_clear_writes_status_and_log(self):
        fake = FakeAerospike()
        _seed(fake)
        out = TriageService(fake, MockDecisionModel()).triage_user("U1")
        assert out["outcome"] == AUTO_CLEAR
        rec = fake.get("flagged_accounts", "U1")
        assert rec["status"] == "auto_cleared"
        assert rec["triage"]["rule"] == "benign_activity"
        assert rec["resolved_by"] == "decision-model-triage"
        [log] = fake.sets["triage_log"].values()
        sent = json.loads(log["request"])
        assert sent["state"]["customer"]["name"] == "Ana Lima"
        assert sent["state"]["recipients"][0]["name"] == "Rent Co"   # names, not ids
        assert "U2" not in log["request"]

    def test_confirmed_fraud_device_escalates_urgent_and_stays_pending(self):
        fake = FakeAerospike()
        _seed(fake, device_fraud=True)
        out = TriageService(fake, MockDecisionModel()).triage_user("U1")
        assert out["outcome"] == ESCALATE_URGENT
        rec = fake.get("flagged_accounts", "U1")
        assert rec["status"] == "pending_review" and rec["priority"] == "urgent"

    def test_jev_failure_escalates_and_records_error(self):
        class Down:
            name, mode, label = "strands-decider", "decider", "Strands Decider"

            def ask(self, state, questions):
                raise DecisionModelError("503")

        fake = FakeAerospike()
        _seed(fake)
        out = TriageService(fake, Down()).triage_user("U1")
        assert (out["outcome"], out["rule"]) == (ESCALATE, "model_unavailable")
        assert fake.get("flagged_accounts", "U1")["status"] == "pending_review"
        assert "503" in fake.get("flagged_accounts", "U1")["triage"]["error"]

    def test_triage_many_and_funnel(self):
        fake = FakeAerospike()
        _seed(fake, "U1")
        svc = TriageService(fake, MockDecisionModel(), concurrency=4,
                            last_detection=lambda: {"users_evaluated": 100, "end_time": "t"})
        svc._investigated_users = lambda: set()
        assert svc.pending_user_ids() == ["U1"]
        run = svc.triage_many(["U1", "U1", "missing"])
        assert run["requested"] == 2 and run["decided"] == 1
        assert run["outcomes"][AUTO_CLEAR] == 1
        assert svc.pending_user_ids() == []
        f = svc.funnel()
        assert f["predictive"] == {"evaluated": 100, "flagged": 1, "threshold": 50.0, "last_run": "t"}
        assert f["decision_model"]["auto_cleared"] == 1
        assert f["agents"]["awaiting"] == 0

    def test_history_parses_request(self):
        fake = FakeAerospike()
        _seed(fake)
        svc = TriageService(fake, MockDecisionModel())
        svc.triage_user("U1")
        [h] = svc.history("U1")
        assert h["request"]["model"] == "mock-decider"
