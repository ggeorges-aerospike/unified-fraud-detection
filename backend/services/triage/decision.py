"""The triage truth table: the decision model's judgments + facts code knows -> an outcome.

Kept as plain, ordered rules so every outcome is visible, testable, and shown in the
UI as the rule that fired. Every combination of inputs lands on exactly one row; the
final row is the catch-all, so an unsettled answer (a Noul between NOUL_NO and
NOUL_YES) always escalates rather than clearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .questions import AUTO_CLEAR_MAX_RISK, NOUL_NO, NOUL_YES, thresholds_for

AUTO_CLEAR = "auto_clear"
ESCALATE = "escalate"
ESCALATE_URGENT = "escalate_urgent"

OUTCOMES = (AUTO_CLEAR, ESCALATE, ESCALATE_URGENT)


def read_noul(p: Optional[float], band: tuple = (NOUL_NO, NOUL_YES)) -> str:
    """'yes' | 'no' | 'unsettled' (a missing answer counts as unsettled)."""
    if p is None:
        return "unsettled"
    no_at, yes_at = band
    if p >= yes_at:
        return "yes"
    if p <= no_at:
        return "no"
    return "unsettled"


@dataclass
class Facts:
    """Things code already knows exactly — never asked of the model."""

    risk_score: float
    device_tied_to_confirmed_fraud: bool = False
    flagged_recipients: int = 0
    # Arithmetic against the bank's norms (see state.BANK_NORMS) — computed, never asked.
    # None = unknown (treated as outside, so it can never enable an auto-clear).
    within_bank_norms: Optional[bool] = None
    norms_detail: str = ""


@dataclass
class Decision:
    outcome: str
    rule: str
    explanation: str
    readings: Dict[str, str] = field(default_factory=dict)
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "rule": self.rule,
            "explanation": self.explanation,
            "readings": self.readings,
            "pattern": self.pattern,
        }


def _noul(answers: Dict[str, Any], name: str) -> Optional[float]:
    a = answers.get(name) or {}
    v = a.get("noul")
    return float(v) if isinstance(v, (int, float)) else None


def decide(answers: Dict[str, Any], facts: Facts, bands: Optional[Dict[str, tuple]] = None) -> Decision:
    """Apply the truth table. ``answers`` is the ``answers`` object of a /v1/systemone response;
    ``bands`` are the per-question reading thresholds (``thresholds_for(model.mode)``)."""
    bands = bands or thresholds_for("default")
    r = {
        name: read_noul(_noul(answers, name), bands.get(name, (NOUL_NO, NOUL_YES)))
        for name in (
            "email_matches_name",
            "recipient_is_customer_alias",
            "impossible_travel",
            "activity_fits_profile",
            "fraud_pattern_present",
        )
    }
    choice = (answers.get("scam_pattern") or {}).get("choice")
    # The pattern name is only meaningful when the companion Noul says one applies.
    pattern = choice if r["fraud_pattern_present"] == "yes" else None

    def d(outcome: str, rule: str, why: str) -> Decision:
        return Decision(outcome, rule, why, r, pattern)

    # --- Urgent: hard evidence from code, or a pattern plus a corroborating judgment
    if facts.device_tied_to_confirmed_fraud:
        return d(ESCALATE_URGENT, "device_confirmed_fraud",
                 "A device on this customer is tied to a confirmed fraud case.")
    if facts.flagged_recipients > 0 and r["fraud_pattern_present"] == "yes":
        return d(ESCALATE_URGENT, "pattern_with_flagged_recipients",
                 f"Fraud pattern present and {facts.flagged_recipients} recipient(s) already flagged.")
    corroborating: List[str] = [
        label for label, hit in (
            ("recipient is a name variant of the customer", r["recipient_is_customer_alias"] == "yes"),
            ("impossible travel between transfers", r["impossible_travel"] == "yes"),
            ("email does not match the customer's name", r["email_matches_name"] == "no"),
        ) if hit
    ]
    if r["fraud_pattern_present"] == "yes" and corroborating:
        return d(ESCALATE_URGENT, "pattern_corroborated",
                 "Fraud pattern present, corroborated by: " + "; ".join(corroborating) + ".")

    # --- Auto-clear: every judgment settled in the customer's favour, no hard facts
    clean = (
        r["fraud_pattern_present"] == "no"
        and r["activity_fits_profile"] == "yes"
        and r["impossible_travel"] == "no"
        and r["recipient_is_customer_alias"] == "no"
        and r["email_matches_name"] != "no"
        and facts.flagged_recipients == 0
    )
    if clean and facts.within_bank_norms is not True:
        return d(ESCALATE, "outside_bank_norms",
                 "Judgments look benign, but the activity is outside the bank's norms"
                 + (f" ({facts.norms_detail})" if facts.norms_detail else "") + "; an investigator should look.")
    if clean and facts.risk_score >= AUTO_CLEAR_MAX_RISK:
        return d(ESCALATE, "score_guardrail",
                 f"Judgments look benign, but a predictive score of {facts.risk_score:.0f} "
                 f"(>= {AUTO_CLEAR_MAX_RISK}) always goes to an investigator.")
    if clean:
        return d(AUTO_CLEAR, "benign_activity",
                 "No fraud pattern, activity fits the customer's profile, and no identity or travel red flags.")

    # --- Everything else: an agent investigates (includes unsettled answers)
    if r["fraud_pattern_present"] == "yes":
        return d(ESCALATE, "pattern_uncorroborated",
                 "Fraud pattern present without corroborating identity or travel signals.")
    unsettled = [k for k, v in r.items() if v == "unsettled"]
    if unsettled:
        return d(ESCALATE, "unsettled",
                 "Not settled by the available state: " + ", ".join(unsettled) + ".")
    return d(ESCALATE, "mixed_signals", "Signals point both ways; an investigator should look.")
