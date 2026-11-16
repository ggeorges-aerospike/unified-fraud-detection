"""Assemble the decision model's ``state`` for one flagged customer from Aerospike.

This is the Aerospike half of the decision tier: Decision models keep no memory between calls,
so everything the model judges is read here, per request, in a handful of batch reads.
Timings and key counts are returned alongside the state so the UI can show what the
read path cost.

State follows TypeSafe's guidance: what a person would look at (names, not ids),
facts code already knows stated plainly, and nothing irrelevant.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .decision import Facts
from .questions import SCAM_PATTERNS

STATE_WINDOW_DAYS = 30
# Self-hosted decision models have small context windows (Strands Decider 2B: 4,096 tokens).
# Show the most recent transfers and recipients in full and summarize the rest in code.
MAX_TRANSFERS = 15
MAX_RECIPIENTS = 15

# Facts about this bank's customers that the model cannot know on its own.
BANK_NORMS = {
    "typical_transfer": "Most customers send individual transfers under $5,000.",
    "typical_frequency": "Most customers send fewer than 5 transfers a week.",
}
# The same norms as numbers, checked in code (arithmetic is never a model question).
NORM_MAX_TRANSFER_USD = 5000
NORM_MAX_PER_WEEK = 5
NORM_MAX_DISTINCT_RECIPIENTS = 10


def norms_check(summary: Dict[str, Any]) -> tuple:
    """(within_norms, detail) from a state's ``transfer_summary``."""
    out = []
    n = summary.get("transfers_sent", 0) or 0
    span = summary.get("days_between_first_and_last") or 0
    if (summary.get("largest_transfer_usd") or 0) > NORM_MAX_TRANSFER_USD:
        out.append(f"largest transfer ${summary['largest_transfer_usd']:,.0f}")
    if n >= 3:
        per_week = n / max(1.0, span) * 7
        if per_week > NORM_MAX_PER_WEEK:
            out.append(f"{per_week:.0f} transfers a week")
    if (summary.get("distinct_recipients") or 0) > NORM_MAX_DISTINCT_RECIPIENTS:
        out.append(f"{summary['distinct_recipients']} distinct recipients")
    return (not out, "; ".join(out))


@dataclass
class AssembledState:
    state: Dict[str, Any]
    facts: Facts
    read_ms: float
    keys_read: int
    breakdown: Dict[str, int] = field(default_factory=dict)


def _days_ago(iso: Optional[str], now: datetime) -> Optional[int]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
    return max(0, (now - dt).days)


def build_state(aerospike, user_id: str, flagged: Dict[str, Any]) -> Optional[AssembledState]:
    """Read everything the model needs about ``user_id``. None if the user record is missing."""
    from services.aerospike_service import SET_DEVICE_FACT, SET_FLAGGED_ACCOUNTS, SET_USERS

    ns = aerospike.namespace
    now = datetime.now()
    t0 = time.perf_counter()
    breakdown: Dict[str, int] = {}

    user = aerospike.get(SET_USERS, user_id)
    breakdown["customer"] = 1
    if not user:
        return None
    accounts: Dict[str, Any] = user.get("accounts") or {}
    devices: Dict[str, Any] = user.get("devices") or {}

    # Device facts (one batch): confirmed-fraud links are a fact, not a question.
    dev_ids = list(devices.keys())
    dev_recs = aerospike.batch_get([(ns, SET_DEVICE_FACT, d) for d in dev_ids])
    breakdown["device_facts"] = len(dev_ids)
    device_fraud = any(bool(r and r[2] and r[2].get("fraud")) for r in dev_recs)

    # Transactions (one batch over daily partitions for every account).
    txns_by_acct = aerospike.batch_get_transactions(list(accounts.keys()), days=STATE_WINDOW_DAYS)
    breakdown["transaction_days"] = len(accounts) * (STATE_WINDOW_DAYS + 1)
    all_txns = [t for ts in txns_by_acct.values() for t in ts]
    outgoing = sorted(
        (t for t in all_txns if t.get("direction") == "out"),
        key=lambda t: t.get("timestamp", ""),
        reverse=True,
    )
    # Money coming back from the people this customer pays is what separates a
    # ring from paying the same landlord every month; count it per recipient.
    received_from = defaultdict(int)
    for t in all_txns:
        if t.get("direction") == "in" and t.get("counterparty_user_id"):
            received_from[t["counterparty_user_id"]] += 1

    # Counterparties (two batches): names for the model, flagged status for code.
    cp_ids = sorted({t.get("counterparty_user_id") for t in outgoing if t.get("counterparty_user_id")} - {user_id})
    cp_users = aerospike.batch_get([(ns, SET_USERS, u) for u in cp_ids])
    cp_flags = aerospike.batch_get([(ns, SET_FLAGGED_ACCOUNTS, u) for u in cp_ids])
    breakdown["recipients"] = 2 * len(cp_ids)
    names = {u: (r[2].get("name") if r and r[2] else None) for u, r in zip(cp_ids, cp_users)}
    # Only decided cases count: with a sensitive predictive tier, a merely pending
    # flag on a recipient is the noise this tier exists to filter.
    flagged_recips = sum(
        1 for r in cp_flags
        if r and r[2] and r[2].get("status") in ("confirmed_fraud", "temporarily_frozen")
    )

    read_ms = (time.perf_counter() - t0) * 1000

    transfers: List[Dict[str, Any]] = []
    for t in outgoing:
        cp = t.get("counterparty_user_id")
        dev = devices.get(t.get("device_id") or "") or {}
        transfers.append({
            "sent_at": t.get("timestamp", ""),
            "amount_usd": round(float(t.get("amount") or 0), 2),
            "recipient": names.get(cp) or "Unknown recipient",
            "from_city": t.get("location", ""),
            "device": f"{dev.get('browser', 'unknown browser')} on {dev.get('os', 'unknown OS')}" if dev else "unknown device",
            "_received_from": received_from.get(cp, 0),
        })

    state = shape_state(
        customer={
            "name": user.get("name", ""),
            "email": user.get("email", ""),
            "age": user.get("age"),
            "occupation": user.get("occupation", ""),
            "home_city": user.get("location", ""),
            "customer_for_days": _days_ago(user.get("signup_date"), now),
        },
        accounts=[
            {"bank": a.get("bank_name", ""), "type": a.get("type", ""), "days_old": _days_ago(a.get("created_date"), now)}
            for a in accounts.values()
        ],
        devices=[
            {"description": f"{d.get('browser', '')} on {d.get('os', '')} ({d.get('type', '')})",
             "first_seen_days_ago": _days_ago(d.get("first_seen"), now)}
            for d in devices.values()
        ],
        transfers=transfers,
    )
    within, detail = norms_check(state["transfer_summary"])
    facts = Facts(
        risk_score=float(flagged.get("risk_score") or 0),
        device_tied_to_confirmed_fraud=device_fraud,
        flagged_recipients=flagged_recips,
        within_bank_norms=within,
        norms_detail=detail,
    )
    return AssembledState(state, facts, round(read_ms, 2), sum(breakdown.values()), breakdown)


def shape_state(
    customer: Dict[str, Any],
    accounts: List[Dict[str, Any]],
    devices: List[Dict[str, Any]],
    transfers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn everything read for a customer into the model's ``state``.

    ``transfers`` is every outgoing transfer in the window, newest first; an optional
    ``_received_from`` on each is how many times that recipient paid the customer back.
    Pure (no I/O), so offline evaluations shape saved data exactly as production does.
    """
    paid: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"times_paid": 0, "total_usd": 0.0, "times_received_from": 0})
    for t in transfers:
        who = t["recipient"]
        paid[who]["times_paid"] += 1
        paid[who]["total_usd"] = round(paid[who]["total_usd"] + t["amount_usd"], 2)
        paid[who]["times_received_from"] = max(paid[who]["times_received_from"], t.get("_received_from", 0))

    amounts = sorted(t["amount_usd"] for t in transfers)
    times = sorted(t["sent_at"] for t in transfers if t.get("sent_at"))
    span_days = None
    if len(times) > 1:
        try:
            span_days = round((datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])).total_seconds() / 86400, 1)
        except ValueError:
            span_days = None
    recipients = sorted(paid.items(), key=lambda kv: (-kv[1]["times_paid"], kv[0]))

    return {
        "customer": customer,
        "accounts": accounts,
        "devices": devices,
        # Computed in code from every transfer in the window, not just the ones listed below.
        "transfer_summary": {
            "window_days": STATE_WINDOW_DAYS,
            "transfers_sent": len(transfers),
            "distinct_recipients": len(paid),
            "recipients_who_also_paid_customer": sum(1 for _, v in recipients if v["times_received_from"] > 0),
            "largest_transfer_usd": amounts[-1] if amounts else 0,
            "median_transfer_usd": amounts[len(amounts) // 2] if amounts else 0,
            "days_between_first_and_last": span_days,
            "distinct_cities": len({t.get("from_city") for t in transfers if t.get("from_city")}),
        },
        "bank_norms": BANK_NORMS,
        "recent_transfers": [
            {k: (v[:16] if k == "sent_at" and isinstance(v, str) else v) for k, v in t.items() if not k.startswith("_")}
            for t in transfers[:MAX_TRANSFERS]
        ],
        "recipients": [{"name": n, **v} for n, v in recipients[:MAX_RECIPIENTS]],
        "known_fraud_patterns": SCAM_PATTERNS,
    }
