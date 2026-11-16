"""Every word the decision model sees, and every threshold applied to its answers, in one place.

TypeSafe's integration guidance (it applies to any System One model): ask for small judgments a person could make at
a glance, and keep the decision itself in code (``decision.py``). Nothing here asks
"is this fraud?" or "should this be escalated?" — those are the truth table's job.

Plain lookups (is a device tied to confirmed fraud, is a recipient already flagged,
the predictive score) are facts code already knows; they go in the state or straight
into the truth table rather than becoming questions.

Request/response shapes follow the System One contract (https://api.typesafe.ai/openapi.json,
v0.2.0), served identically by Jev, Kev and Strands Decider:
``questions`` is a map of name -> {type, instructions, criteria}.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Thresholds for reading Noul answers (probability the statement is true).
# TypeSafe's suggested starting points: above 0.7 reads as yes, below 0.3 as no,
# and the band between is "not settled" — which gets its own row in the table.
# ---------------------------------------------------------------------------
NOUL_YES = 0.7
NOUL_NO = 0.3

# Per-model overrides, as (no_at_or_below, yes_at_or_above) per question. Smaller self-hosted
# models rank well but compress probabilities toward the middle, so the generic bands leave
# almost everything unsettled. Each override is fitted on labelled demo data with a train /
# held-out split, required to auto-clear no held-out fraud, then backed off for margin.
#
# Strands Decider 2B (strands-decider-2B-qwen3.5-v1-2610), measured 2026-11-16 on labelled demo data:
#  - fraud_pattern_present: fitted on 320 customers (160 fit / 160 held out, 0 held-out fraud cleared).
#  - scam_language / payee_is_customer: "yes" only raises priority to urgent (never clears), so it
#    sits just above the highest benign score seen (0.31 / 0.45); "no" sits just above ordinary memos.
#  - large_payment_explained keeps the generic 0.7: "yes" can enable an auto-clear, and vague
#    fraud memos reached 0.60.
MODEL_THRESHOLDS: Dict[str, Dict[str, tuple]] = {
    "decider": {
        "fraud_pattern_present": (0.56, 0.66),
        "scam_language": (0.25, 0.40),
        "payee_is_customer": (0.30, 0.50),
    },
}


def thresholds_for(mode: str) -> Dict[str, tuple]:
    """Reading bands for every Noul question for a model provider (mock/jev use the defaults)."""
    bands = {q: (NOUL_NO, NOUL_YES) for q, spec in QUESTIONS_SPEC_NOULS.items()}
    bands.update(MODEL_THRESHOLDS.get(mode, {}))
    return bands


# Guardrail: predictive scores at or above this never auto-clear, whatever the model says.
AUTO_CLEAR_MAX_RISK = 80

# Scam-pattern Choice options. Names describe the content (they are sent to the
# model); each description is a concrete situation.
SCAM_PATTERNS: Dict[str, str] = {
    "mule_fan_out": "Money is sent to many different recipients the customer has not paid before, within a few days",
    "circular_ring": "Money moves back and forth among the same small group of people",
    "account_takeover": "Large transfers start suddenly from new devices and do not fit the customer's earlier activity",
    "bust_out": "A newly opened or credit account runs up large outgoing amounts quickly",
}

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "email_matches_name": {
        "type": "noul",
        "instructions": "Does `customer.email` belong to the person named in `customer.name`?",
        "criteria": {
            "true": "The email address uses the customer's name, initials, or an obvious nickname of it",
            "false": "The email address uses a different person's name, a business unrelated to them, or random characters",
        },
    },
    # Asked only when code found a typed payee name that differs from the account holder's.
    "payee_is_customer": {
        "type": "noul",
        "instructions": "Is any `payee_name_entered` in `mismatched_payees` the customer's own name, written differently?",
        "criteria": {
            "true": "An entered payee name is the customer's name with a misspelling, swapped order, nickname, initial, or a word like 'Savings' added",
            "false": "Every entered payee name is a nickname or short form of the actual account holder, or another person entirely",
        },
    },
    "scam_language": {
        "type": "noul",
        "instructions": "Does any `memo` in `recent_transfers` describe a payment that scammers commonly demand?",
        "criteria": {
            "true": "A memo mentions gift cards, crypto recovery or wallet fees, bail or an emergency for a relative, guaranteed investment returns, customs or release fees, a government agency demanding payment, a lottery fee, or travel costs for someone met online",
            "false": "Memos describe rent, bills, purchases, gifts, or paying back friends and family, or are blank or vague",
        },
    },
    # Asked only when code found transfers over the bank's typical size.
    "large_payment_explained": {
        "type": "noul",
        "instructions": "Does every transfer in `large_transfers` have a memo naming a specific, ordinary purpose?",
        "criteria": {
            "true": "Each memo names something concrete and ordinary, such as a home closing with an address, a car purchase, tuition, taxes, or a contractor invoice",
            "false": "A memo is blank or vague (like 'payment', 'funds', or 'as discussed'), or describes something unusual",
        },
    },
    # Choice always picks an option, so it is paired with a Noul that asks whether
    # any pattern applies at all (the guidance's "null case" companion).
    "fraud_pattern_present": {
        "type": "noul",
        "instructions": "Does the activity in `recent_transfers` match any pattern described in `known_fraud_patterns`?",
        "criteria": {
            "true": "The transfers clearly show one of the described patterns",
            "false": "The transfers look like ordinary activity that none of the descriptions fit",
        },
    },
    "scam_pattern": {
        "type": "choice",
        "instructions": "Which pattern in `known_fraud_patterns` best describes the activity in `recent_transfers`?",
        "criteria": SCAM_PATTERNS,
    },
}


QUESTIONS_SPEC_NOULS = {q: spec for q, spec in QUESTIONS.items() if spec["type"] == "noul"}


# Questions that only make sense when code found something to judge. When skipped, the
# truth table uses the reading code already knows (nothing to judge -> that reading).
CONDITIONAL = {
    "payee_is_customer": ("mismatched_payees", "no"),
    "large_payment_explained": ("large_transfers", "n/a"),
}


def request_questions(state: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """The ``questions`` object for a /v1/systemone request (a copy, safe to mutate).

    With a ``state``, conditional questions whose input list is empty are left out: an empty
    list is something code already knows, and the model can read emptiness as a signal.
    """
    out = {}
    for name, q in QUESTIONS.items():
        if state is not None and name in CONDITIONAL and not state.get(CONDITIONAL[name][0]):
            continue
        out[name] = dict(q)
    return out


def skipped_readings(state: Dict[str, Any]) -> Dict[str, str]:
    """Readings code supplies for the conditional questions it did not ask."""
    return {name: reading for name, (field, reading) in CONDITIONAL.items() if not state.get(field)}
