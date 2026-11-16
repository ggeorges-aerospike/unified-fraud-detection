"""Decision-model triage (tier 2): model judgments + a code truth table, state in Aerospike.

Any System One server works as the model: Strands Decider, Kev, or TypeSafe Jev (see
``decision_models.py``).
"""

from .decision import AUTO_CLEAR, ESCALATE, ESCALATE_URGENT, Decision, Facts, decide
from .decision_models import (
    PROVIDERS,
    DecisionModel,
    DecisionModelError,
    MockDecisionModel,
    SystemOneHttpModel,
    model_from_env,
)
from .service import STATUS_AUTO_CLEARED, TriageService

__all__ = [
    "AUTO_CLEAR", "ESCALATE", "ESCALATE_URGENT", "Decision", "Facts", "decide",
    "PROVIDERS", "DecisionModel", "DecisionModelError", "MockDecisionModel", "SystemOneHttpModel",
    "model_from_env", "STATUS_AUTO_CLEARED", "TriageService",
]
