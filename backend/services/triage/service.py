"""Tier 2 of the fraud funnel: decision-model triage between predictive scoring and the agents.

For each flagged customer: read state from Aerospike -> ask the decision model the questions in
``questions.py`` -> apply the truth table in ``decision.py`` -> write the decision
back to Aerospike (an append-only ``triage_log`` record plus a summary on the
flagged-account record).

Outcomes:
  auto_clear       -> status ``auto_cleared``; never reaches an agent or analyst
  escalate         -> stays ``pending_review`` (priority normal) for an AI investigation
  escalate_urgent  -> stays ``pending_review`` with priority ``urgent`` (sorted first)

A failed model call is recorded as an error and escalates: no answer is never read as
"benign".
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from .decision import AUTO_CLEAR, ESCALATE, ESCALATE_URGENT, decide
from .decision_models import DecisionModel, DecisionModelError
from .questions import request_questions, skipped_readings, thresholds_for
from .state import build_state

logger = logging.getLogger("fraud_detection.triage")

STATUS_AUTO_CLEARED = "auto_cleared"
HUMAN_DECIDED = ("confirmed_fraud", "cleared", "temporarily_frozen", "monitoring")


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1], 2)


class TriageService:
    def __init__(
        self,
        aerospike,
        model: Optional[DecisionModel],
        concurrency: int = 16,
        last_detection: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        risk_threshold: Optional[Callable[[], float]] = None,
    ):
        self._as = aerospike
        self.model = model
        self.concurrency = max(1, concurrency)
        self._last_detection = last_detection or (lambda: None)
        self._risk_threshold = risk_threshold or (lambda: 50.0)
        self.last_run: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------ one user
    def triage_user(self, user_id: str) -> Dict[str, Any]:
        from services.aerospike_service import SET_FLAGGED_ACCOUNTS, SET_TRIAGE_LOG

        if not self.model:
            return {"user_id": user_id, "error": "Triage disabled (DECISION_MODEL=off)"}
        flagged = self._as.get(SET_FLAGGED_ACCOUNTS, user_id)
        if not flagged:
            return {"user_id": user_id, "error": "Not a flagged customer"}

        assembled = build_state(self._as, user_id, flagged)
        if not assembled:
            return {"user_id": user_id, "error": "Customer record missing"}

        questions = request_questions(assembled.state)
        known = skipped_readings(assembled.state)
        bands = thresholds_for(self.model.mode)
        request = {"model": self.model.name, "state": assembled.state, "questions": questions}
        t0 = time.perf_counter()
        error: Optional[str] = None
        response: Dict[str, Any] = {}
        try:
            response = self.model.ask(assembled.state, questions)
            decision = decide(response.get("answers") or {}, assembled.facts, bands, known)
            outcome, rule, explanation = decision.outcome, decision.rule, decision.explanation
            readings, pattern = decision.readings, decision.pattern
        except DecisionModelError as e:
            error = str(e)
            outcome, rule = ESCALATE, "model_unavailable"
            explanation = "The decision model did not answer; escalating rather than guessing."
            readings, pattern = {}, None
        model_ms = round((time.perf_counter() - t0) * 1000, 2)

        now = datetime.now()
        summary = {
            "outcome": outcome,
            "rule": rule,
            "explanation": explanation,
            "pattern": pattern,
            "readings": readings,
            "thresholds": {q: list(b) for q, b in bands.items()},
            "model": response.get("model") or self.model.name,
            "mode": self.model.mode,
            "decided_at": now.isoformat(),
            "model_ms": model_ms,
            "read_ms": assembled.read_ms,
            "keys_read": assembled.keys_read,
            "error": error,
        }
        log = {
            **{k: v for k, v in summary.items() if v is not None},
            "user_id": user_id,
            "risk_score": assembled.facts.risk_score,
            "facts": {
                "device_confirmed_fraud": assembled.facts.device_tied_to_confirmed_fraud,
                "flagged_recipients": assembled.facts.flagged_recipients,
                "within_bank_norms": assembled.facts.within_bank_norms,
                "outside_only_by_amount": assembled.facts.outside_only_by_amount,
                "norms_detail": assembled.facts.norms_detail,
            },
            "answers": response.get("answers") or {},
            "input_tokens": (response.get("usage") or {}).get("input_tokens", 0),
            # The exact payload, so any decision can be audited and replayed.
            "request": json.dumps(request, default=str),
        }
        t1 = time.perf_counter()
        self._as.put(SET_TRIAGE_LOG, f"{user_id}:{int(now.timestamp() * 1000)}", log)

        updates: Dict[str, Any] = {
            "triage": {k: v for k, v in summary.items() if v is not None},
            "priority": "urgent" if outcome == ESCALATE_URGENT else "normal",
        }
        if outcome == AUTO_CLEAR:
            updates.update({
                "status": STATUS_AUTO_CLEARED,
                "resolution": STATUS_AUTO_CLEARED,
                "resolution_date": now.isoformat(),
                "resolution_notes": f"[Decision-model triage ({summary['model']}): {rule}] {explanation}",
                "resolved_by": "decision-model-triage",
            })
        self._as.update_flagged_account(user_id, updates)
        write_ms = round((time.perf_counter() - t1) * 1000, 2)

        return {"user_id": user_id, **summary, "write_ms": write_ms,
                "risk_score": assembled.facts.risk_score, "answers": response.get("answers") or {}}

    # ------------------------------------------------------------------ many
    def triage_many(self, user_ids: Iterable[str]) -> Dict[str, Any]:
        ids = list(dict.fromkeys(u for u in user_ids if u))
        started = datetime.now()
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        if ids and self.model:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(ids))) as pool:
                for r in pool.map(self._safe_triage, ids):
                    results.append(r)
        wall_s = time.perf_counter() - t0

        ok = [r for r in results if "outcome" in r]
        counts = {o: sum(1 for r in ok if r["outcome"] == o) for o in (AUTO_CLEAR, ESCALATE, ESCALATE_URGENT)}
        model_lat = [r["model_ms"] for r in ok]
        reads = [r["read_ms"] for r in ok]
        writes = [r["write_ms"] for r in ok]
        run = {
            "started_at": started.isoformat(),
            "mode": getattr(self.model, "mode", "off"),
            "model": getattr(self.model, "name", None),
            "requested": len(ids),
            "decided": len(ok),
            "errors": [r for r in results if "outcome" not in r or r.get("error")][:20],
            "outcomes": counts,
            "wall_seconds": round(wall_s, 3),
            "decisions_per_second": round(len(ok) / wall_s, 1) if wall_s > 0 and ok else 0,
            "concurrency": self.concurrency,
            "latency_ms": {
                "model_p50": _pct(model_lat, 50), "model_p95": _pct(model_lat, 95),
                "state_read_p50": _pct(reads, 50), "state_read_p95": _pct(reads, 95),
                "write_p50": _pct(writes, 50), "write_p95": _pct(writes, 95),
            },
            "keys_read": sum(r.get("keys_read", 0) for r in ok),
        }
        self.last_run = run
        logger.info("Triage run: %s decided in %.2fs (%s)", len(ok), wall_s, counts)
        return run

    def _safe_triage(self, user_id: str) -> Dict[str, Any]:
        try:
            return self.triage_user(user_id)
        except Exception as e:  # one bad record must not sink the batch
            logger.exception("Triage failed for %s", user_id)
            return {"user_id": user_id, "error": str(e)}

    def pending_user_ids(self, include_triaged: bool = False) -> List[str]:
        out = []
        for a in self._flagged():
            if a.get("status") != "pending_review":
                continue
            if include_triaged or not a.get("triage"):
                out.append(a.get("user_id"))
        return [u for u in out if u]

    # ------------------------------------------------------------------ reads
    def _flagged(self) -> List[Dict[str, Any]]:
        from services.aerospike_service import SET_FLAGGED_ACCOUNTS

        return self._as.scan_all(SET_FLAGGED_ACCOUNTS, limit=100000)

    def history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        from services.aerospike_service import SET_TRIAGE_LOG

        rows = self._as._query_equals(SET_TRIAGE_LOG, "user_id", user_id, limit=500)
        rows.sort(key=lambda r: r.get("decided_at", ""), reverse=True)
        for r in rows:
            try:
                r["request"] = json.loads(r.get("request") or "{}")
            except ValueError:
                pass
        return rows[:limit]

    def _investigated_users(self) -> set:
        from services.aerospike_service import SET_INVESTIGATIONS

        users: set = set()
        try:
            scan = self._as.client.scan(self._as.namespace, SET_INVESTIGATIONS)
            scan.select("user_id")
            scan.foreach(lambda rec: users.add(rec[2].get("user_id")) if rec and rec[2] else None)
        except Exception as e:
            logger.warning("Could not count investigations: %s", e)
        users.discard(None)
        return users

    def funnel(self) -> Dict[str, Any]:
        accounts = self._flagged()
        job = self._last_detection() or {}
        triaged = [a for a in accounts if a.get("triage")]
        outcome = lambda a: (a.get("triage") or {}).get("outcome")  # noqa: E731
        escalated = [a for a in triaged if outcome(a) in (ESCALATE, ESCALATE_URGENT)]
        investigated = self._investigated_users()
        reached_agent = [a for a in accounts if a.get("user_id") in investigated]
        return {
            "predictive": {
                "evaluated": job.get("users_evaluated"),
                "flagged": len(accounts),
                "threshold": self._risk_threshold(),
                "last_run": job.get("end_time"),
            },
            "decision_model": {
                "enabled": self.enabled,
                "mode": getattr(self.model, "mode", "off"),
                "model": getattr(self.model, "name", None),
                "provider": getattr(self.model, "label", None),
                "triaged": len(triaged),
                "untriaged_pending": sum(1 for a in accounts if a.get("status") == "pending_review" and not a.get("triage")),
                "auto_cleared": sum(1 for a in accounts if a.get("status") == STATUS_AUTO_CLEARED),
                "escalated": len(escalated),
                "urgent": sum(1 for a in escalated if outcome(a) == ESCALATE_URGENT),
                "last_run": self.last_run,
            },
            "agents": {
                "investigated": len(reached_agent),
                "awaiting": sum(1 for a in accounts if a.get("status") == "pending_review"
                                and a.get("user_id") not in investigated),
            },
            "human": {
                "decided": sum(1 for a in accounts if a.get("status") in HUMAN_DECIDED),
                "by_status": {s: sum(1 for a in accounts if a.get("status") == s) for s in HUMAN_DECIDED},
            },
        }
