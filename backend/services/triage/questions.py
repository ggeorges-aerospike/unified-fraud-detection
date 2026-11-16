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

from typing import Any, Dict

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
# Strands Decider 2B (strands-decider-2B-qwen3.5-v1-2610), fitted 2026-11-11 on 320 labelled
# customers (160 fit / 160 held out): held-out benign auto-cleared 45%, held-out fraud 0 of 80.
MODEL_THRESHOLDS: Dict[str, Dict[str, tuple]] = {
    "decider": {
        "fraud_pattern_present": (0.56, 0.66),
        "activity_fits_profile": (0.28, 0.38),
        "impossible_travel": (0.55, 0.70),
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
    "recipient_is_customer_alias": {
        "type": "noul",
        "instructions": "Is any person in `recipients` the customer themself under a variant of their name?",
        "criteria": {
            "true": "A recipient's name is the customer's name with a misspelling, swapped order, initial, or added title",
            "false": "Every recipient's name belongs to a clearly different person",
        },
    },
    "impossible_travel": {
        "type": "noul",
        "instructions": "Were transfers in `recent_transfers` sent from cities too far apart to travel between in the time separating them?",
        "criteria": {
            "true": "Two transfers hours apart come from cities a flight or more apart",
            "false": "The cities and times between transfers could be covered by ordinary travel, such as transfers days apart or from the same region",
        },
    },
    "activity_fits_profile": {
        "type": "noul",
        "instructions": "Do the amounts and frequency in `recent_transfers` fit an ordinary person with the occupation and age in `customer`?",
        "criteria": {
            "true": "Transfer sizes and counts look like everyday bills, rent, and payments to the same few people for someone in this job",
            "false": "Transfer sizes or counts are far beyond what this customer's job and age would explain",
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


def request_questions() -> Dict[str, Dict[str, Any]]:
    """The ``questions`` object for a /v1/systemone request (a copy, safe to mutate)."""
    return {name: dict(q) for name, q in QUESTIONS.items()}
