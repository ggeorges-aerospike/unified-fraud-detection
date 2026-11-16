"""Decision-model clients for tier-2 triage: one HTTP client for every System One server, and a mock.

TypeSafe's hosted Jev, the open-source Kev, and Amazon's open-source Strands Decider all serve
the same ``POST /v1/systemone`` contract (https://api.typesafe.ai/openapi.json, v0.2.0)::

    request:  {"model"?: str, "state": ..., "questions": {name: {type, instructions, criteria}}}
    response: {"model": str, "answers": {name: Answer}, "usage": {...}}   (+ "latency_ms" on Kev/Decider)

so choosing a model is configuration (URL, model name, key), not code. ``DECISION_MODEL`` picks a
preset; ``DECISION_MODEL_URL`` / ``DECISION_MODEL_NAME`` / ``DECISION_MODEL_API_KEY`` override it.

We call the HTTP contract directly rather than through ``strands.decisions``: that SDK interface
has no field for option/yes-no criteria, which these questions rely on.

The mock reads only the ``state`` it is sent — the same input a real model gets — and answers with
simple heuristics tuned to this demo's data generator. It keeps CI and offline runs working; it is
not a measure of any real model's quality.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import httpx

logger = logging.getLogger("fraud_detection.triage.models")


class DecisionModelError(RuntimeError):
    """A failed call. Callers must treat this as 'no answer', never as a 'no'."""


class DecisionModel(Protocol):
    name: str      # model name to show (updated from the server's response)
    mode: str      # provider id: mock | decider | kev | jev
    label: str     # human label for the provider

    def ask(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    base_url: str
    model: Optional[str]          # sent as "model"; None = let the server use the one it serves
    key_env: Optional[str] = None
    key_required: bool = False


PROVIDERS: Dict[str, Provider] = {
    # Amazon's open-source decision model. Local: `strands-decider serve <model> --port 8000`;
    # on AWS, the repo's SageMaker/Lambda stacks take the same request body.
    "decider": Provider("decider", "Strands Decider", "http://host.docker.internal:8000", None),
    # Open-source Jev-compatible models. Local: `python -m kev.serve --run jaredpalmer/kev-4b --port 8009`.
    "kev": Provider("kev", "Kev", "http://host.docker.internal:8009", None),
    # TypeSafe's hosted Jev.
    "jev": Provider("jev", "TypeSafe Jev", "https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY", True),
}


# ---------------------------------------------------------------------------
# HTTP client (any /v1/systemone server)
# ---------------------------------------------------------------------------
class SystemOneHttpModel:
    def __init__(
        self,
        provider: Provider,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if provider.key_required and not api_key:
            raise ValueError(f"{provider.label} needs an API key ({provider.key_env} or DECISION_MODEL_API_KEY)")
        self.mode = provider.id
        self.label = provider.label
        self._model = model or provider.model
        self.name = self._model or provider.label
        self.base_url = base_url or provider.base_url
        self._max_attempts = max_attempts
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s, headers=headers, transport=transport)

    def ask(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"state": state, "questions": questions}
        if self._model:
            payload["model"] = self._model
        last: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._client.post("/v1/systemone", json=payload)
            except httpx.HTTPError as e:
                last = e
            else:
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("model"):
                        self.name = body["model"]
                    return body
                if resp.status_code not in (429, 503, 529) and resp.status_code < 500:
                    raise DecisionModelError(
                        f"{self.label} rejected the request ({resp.status_code}): {resp.text[:500]}")
                last = DecisionModelError(f"{self.label} returned {resp.status_code}")
            if attempt < self._max_attempts:
                time.sleep(min(2.0, 0.25 * 2 ** (attempt - 1)))
        raise DecisionModelError(f"{self.label} at {self.base_url} unavailable after {self._max_attempts} attempts: {last}")


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "phd", "md", "dds", "dvm", "jr", "sr", "ii", "iii", "iv"}


def _name_tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[^a-z]+", (name or "").lower()) if t and t not in _TITLES]


def _clamp(p: float) -> float:
    return round(max(0.01, min(0.99, p)), 3)


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


class MockDecisionModel:
    """Deterministic heuristic answers over the state, with optional simulated latency."""

    mode = "mock"
    label = "Mock decision model"

    def __init__(self, latency_ms: tuple[int, int] = (0, 0), model: str = "mock-decider"):
        self.name = model
        self._latency = latency_ms

    # -- per-question heuristics ------------------------------------------------
    @staticmethod
    def _email_matches_name(state: Dict[str, Any]) -> float:
        c = state.get("customer") or {}
        toks = _name_tokens(c.get("name", ""))
        local = (c.get("email") or "").split("@")[0].lower()
        if not toks or not local:
            return 0.5
        parts = [p for p in re.split(r"[^a-z]+", local) if p]
        last, first = toks[-1], toks[0]
        if last in parts and (first in parts or first[0] in parts or local.startswith(first[0])):
            return 0.96
        if last in local:
            return 0.85
        ratio = difflib.SequenceMatcher(None, "".join(toks), "".join(parts)).ratio()
        return _clamp(0.1 + ratio * 0.8)

    @staticmethod
    def _payee_is_customer(state: Dict[str, Any]) -> float:
        """Typed payee name vs the customer's own name, for payees typed differently from the holder."""
        me_toks = _name_tokens((state.get("customer") or {}).get("name", ""))
        me = " ".join(sorted(me_toks))
        best = 0.0
        for p in state.get("mismatched_payees") or []:
            typed = [t for t in _name_tokens(p.get("payee_name_entered", "")) if t not in ("savings", "account")]
            holder = " ".join(sorted(_name_tokens(p.get("account_holder", ""))))
            typed_s = " ".join(sorted(typed))
            if not typed_s:
                continue
            to_me = difflib.SequenceMatcher(None, me, typed_s).ratio()
            to_holder = difflib.SequenceMatcher(None, holder, typed_s).ratio()
            # nickname/initial of the customer: shares the surname and a first-name initial
            if me_toks and typed and typed[-1] == me_toks[-1] and typed[0][0] == me_toks[0][0]:
                to_me = max(to_me, 0.9)
            if to_me > to_holder:
                best = max(best, to_me)
        if best >= 0.85:
            return 0.92
        if best >= 0.75:
            return 0.55
        return _clamp(0.03 + best * 0.1)

    _SCAM_WORDS = ("gift cards", "gift card", "crypto", "wallet recovery", "bail", "guaranteed", "customs",
                   "release fee", "irs", "social security", "lottery", "visa fees", "tech support", "refund reversal")
    _SPECIFIC_WORDS = ("closing", "escrow", "down payment", "tuition", "tax", "invoice", "deposit",
                       "contractor", "remodel", "roof", "venue")

    @staticmethod
    def _mentions(text: str, words) -> bool:
        """Whole-word / whole-phrase match ("irs" must not match "first")."""
        t = (text or "").lower()
        return any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)

    def _scam_language(self, state: Dict[str, Any]) -> float:
        memos = [t.get("memo") or "" for t in state.get("recent_transfers") or []]
        hits = sum(1 for m in memos if self._mentions(m, self._SCAM_WORDS))
        return 0.94 if hits else 0.04

    def _large_payment_explained(self, state: Dict[str, Any]) -> float:
        large = state.get("large_transfers") or []
        if not large:
            return 0.5
        ok = [self._mentions(t.get("memo") or "", self._SPECIFIC_WORDS) and len(t.get("memo") or "") > 12
              for t in large]
        return 0.9 if all(ok) else 0.08

    @staticmethod
    def _impossible_travel(state: Dict[str, Any]) -> float:
        rows = []
        for t in state.get("recent_transfers") or []:
            ts = _parse_ts(t.get("sent_at", ""))
            if ts:
                rows.append((ts, (t.get("from_city") or "").split(",")[0].strip().lower()))
        rows.sort()
        jumps = sum(
            1 for (t0, c0), (t1, c1) in zip(rows, rows[1:])
            if c0 and c1 and c0 != c1 and (t1 - t0).total_seconds() < 3 * 3600
        )
        return {0: 0.06, 1: 0.5}.get(jumps, 0.88 if jumps < 4 else 0.95)

    @staticmethod
    def _activity(state: Dict[str, Any]) -> Dict[str, float]:
        tx = state.get("recent_transfers") or []
        n = len(tx)
        amounts = [float(t.get("amount_usd") or 0) for t in tx]
        avg = sum(amounts) / n if n else 0.0
        times = sorted(filter(None, (_parse_ts(t.get("sent_at", "")) for t in tx)))
        span_days = max(1.0, (times[-1] - times[0]).total_seconds() / 86400) if len(times) > 1 else 1.0
        # Too few transfers to call it a rate: two payments on one day is not "14 a week".
        per_week = n / span_days * 7 if n >= 3 else float(n)
        recips = state.get("recipients") or []
        once = sum(1 for r in recips if (r.get("times_paid") or 0) <= 1)
        summ = state.get("transfer_summary") or {}
        two_way = sum(1 for r in recips if (r.get("times_paid") or 0) >= 1 and (r.get("times_received_from") or 0) >= 1)
        new_devices = sum(1 for d in state.get("devices") or [] if d.get("first_seen_days_ago", 999) <= 30)
        acct_new = any((a.get("days_old") or 9999) <= 60 for a in state.get("accounts") or [])
        if summ:  # prefer the code-computed summary over the (capped) listed transfers
            n_all = summ.get("transfers_sent", n)
            span = summ.get("days_between_first_and_last") or 1.0
            per_week = n_all / max(1.0, span) * 7 if n_all >= 3 else float(n_all)
            n = n_all
        return {
            "n": n, "avg": avg, "per_week": per_week,
            "uniq": summ.get("distinct_recipients", len(recips)),
            "once": once, "two_way": summ.get("recipients_who_also_paid_customer", two_way),
            "new_devices": new_devices,
            "acct_new": 1.0 if acct_new else 0.0,
            "total": avg * n if summ else sum(amounts),
            "max": summ.get("largest_transfer_usd", max(amounts) if amounts else 0.0),
        }

    def _activity_fits_profile(self, a: Dict[str, float]) -> float:
        if a["n"] == 0:
            return 0.9
        # Everyday payments (rent, bills, a used car) top out around $5k; single
        # transfers of $10k+ or 15+ a week are beyond an ordinary profile.
        if a["per_week"] <= 6 and a["max"] <= 5000:
            return 0.88
        if a["per_week"] >= 15 or a["max"] >= 10000:
            return 0.08
        return 0.45

    def _patterns(self, a: Dict[str, float]) -> Dict[str, float]:
        def sig(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        return {
            "mule_fan_out": sig((a["once"] - 6) / 2 + (a["per_week"] - 10) / 6),
            # A ring needs money flowing both ways with at least two recipients.
            "circular_ring": sig((a["two_way"] - 1.5) * 2.5 + (a["avg"] - 1500) / 3000) if a["two_way"] >= 2 else 0.03,
            # A single large payment alone is not a pattern: takeover needs a new device,
            # bust-out needs a new account, each alongside large outflows.
            "account_takeover": sig((a["max"] - 9000) / 3000) if a["new_devices"] else 0.03,
            "bust_out": sig((a["total"] - 30000) / 10000) if a["acct_new"] else 0.03,
        }

    # -- the API --------------------------------------------------------------
    def ask(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        lo, hi = self._latency
        if hi > 0:
            time.sleep(random.uniform(lo, hi) / 1000.0)

        act = self._activity(state)
        pats = self._patterns(act)
        computed: Dict[str, Any] = {
            "email_matches_name": self._email_matches_name(state),
            "payee_is_customer": self._payee_is_customer(state),
            "scam_language": self._scam_language(state),
            "large_payment_explained": self._large_payment_explained(state),
            "impossible_travel": self._impossible_travel(state),
            "activity_fits_profile": self._activity_fits_profile(act),
            "fraud_pattern_present": _clamp(max(pats.values()) if act["n"] else 0.03),
        }
        answers: Dict[str, Any] = {}
        for name, q in questions.items():
            qtype = q.get("type")
            if qtype == "noul":
                answers[name] = {"type": "noul", "noul": computed.get(name, 0.5)}
            elif qtype == "choice":
                opts = list((q.get("criteria") or {}).keys())
                raw = {o: pats.get(o, 0.05) + 1e-3 for o in opts}
                total = sum(raw.values()) or 1.0
                probs = {o: round(v / total, 3) for o, v in raw.items()}
                top = max(probs, key=probs.get)
                answers[name] = {"type": "choice", "choice": top,
                                 "confidence": probs[top], "probabilities": probs}
            elif qtype == "score":
                levels = q.get("criteria") or ["level"]
                probs = {str(i): round(1 / len(levels), 3) for i in range(len(levels))}
                answers[name] = {"type": "score", "score": (len(levels) - 1) / 2, "confidence": 0.0,
                                 "legend": {str(i): l for i, l in enumerate(levels)}, "probabilities": probs}
        size = len(json.dumps(state, default=str)) + len(json.dumps(questions))
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": size // 4, "output_tokens": 4 * len(answers)}}


# ---------------------------------------------------------------------------
def _parse_latency(raw: str) -> tuple[int, int]:
    try:
        parts = [int(p) for p in raw.replace(" ", "").split("-") if p]
    except ValueError:
        return (0, 0)
    if not parts:
        return (0, 0)
    return (parts[0], parts[-1])


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip() != "":
            return v.strip()
    return default


def model_from_env() -> Optional[DecisionModel]:
    """DECISION_MODEL: mock (default) | decider | kev | jev | off. Returns None when off.

    JEV_MODE (mock | live | off) is still honoured when DECISION_MODEL is unset; live means jev.
    """
    choice = _env("DECISION_MODEL").lower()
    if not choice:
        legacy = _env("JEV_MODE", default="mock").lower()
        choice = {"live": "jev"}.get(legacy, legacy)
    if choice == "off":
        return None
    if choice == "mock":
        return MockDecisionModel(latency_ms=_parse_latency(
            _env("DECISION_MODEL_MOCK_LATENCY_MS", "JEV_MOCK_LATENCY_MS", default="70-250")))
    provider = PROVIDERS.get(choice)
    if not provider:
        raise ValueError(f"Unknown DECISION_MODEL={choice!r}; use one of mock, {', '.join(PROVIDERS)}, off")
    api_key = _env("DECISION_MODEL_API_KEY") or (os.environ.get(provider.key_env, "").strip() if provider.key_env else "")
    return SystemOneHttpModel(
        provider,
        base_url=_env("DECISION_MODEL_URL") or None,
        model=_env("DECISION_MODEL_NAME", "JEV_MODEL") or None,
        api_key=api_key or None,
        timeout_s=float(_env("DECISION_MODEL_TIMEOUT_S", default="30")),
    )
