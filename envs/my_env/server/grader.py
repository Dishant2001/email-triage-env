from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from ..models import Email, EmailPriority, EnvConfig, MyAction
except ImportError:  # pragma: no cover
    from models import Email, EmailPriority, EnvConfig, MyAction
from .dynamics import remaining_sla, urgency_score


@dataclass
class GradeBreakdown:
    sla_breach: bool
    sla_score: float
    prioritization_score: float
    action_score: float
    response_score: float
    cost_penalty: float
    idle_penalty: float
    total: float


def _best_email(pending: List[Email], current_time: int) -> Email | None:
    if not pending:
        return None
    return max(pending, key=lambda e: urgency_score(e, current_time))


def _response_keyword_score(action: MyAction, email: Email) -> float:
    if action.action_type != "reply":
        return 0.0
    required = [k.strip().lower() for k in email.required_response_keywords if k.strip()]
    if not required:
        # minimal deterministic proxy
        return 0.3 if len(action.response.strip()) >= 12 else 0.0
    text = action.response.lower()
    hits = sum(1 for k in required if k in text)
    return 0.3 * (hits / max(1, len(required)))


def grade_step(
    *,
    action: MyAction,
    chosen_email: Email,
    pending_before: List[Email],
    current_time: int,
    config: EnvConfig,
) -> GradeBreakdown:
    total = 0.0

    # SLA
    sla_breach = (current_time - chosen_email.created_time) > chosen_email.sla_limit
    sla_score = -1.0 if sla_breach else 0.5
    total += sla_score

    # Prioritization (did we pick the most urgent?)
    prioritization_score = 0.0
    best = _best_email(pending_before, current_time)
    if best is not None and chosen_email.email_id == best.email_id:
        prioritization_score = 0.4
    total += prioritization_score

    # Action correctness (deterministic ground truth label)
    action_score = 0.3 if action.action_type == chosen_email.ground_truth_action else 0.0
    total += action_score

    # Response rubric (keyword-based / length proxy)
    response_score = _response_keyword_score(action, chosen_email)
    total += response_score

    # Costs / penalties
    action_cost = float(config.action_costs.get(action.action_type, 0.0))
    cost_penalty = -action_cost
    idle_penalty = -float(config.per_step_idle_cost)
    total += cost_penalty + idle_penalty

    return GradeBreakdown(
        sla_breach=sla_breach,
        sla_score=sla_score,
        prioritization_score=prioritization_score,
        action_score=action_score,
        response_score=response_score,
        cost_penalty=cost_penalty,
        idle_penalty=idle_penalty,
        total=total,
    )

