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
    throughput_score: float
    breach_load_penalty: float
    cost_penalty: float
    idle_penalty: float
    consequence_signal: float
    total: float


def effective_sla_limit(email: Email, sla_pressure_offset: int) -> int:
    """Tighter effective window when entanglement increased pressure (bad prior decisions)."""
    return max(1, int(email.sla_limit) - int(sla_pressure_offset))


def _best_email(pending: List[Email], current_time: int) -> Email | None:
    if not pending:
        return None
    return max(pending, key=lambda e: urgency_score(e, current_time))


def step_reward_bounds(config: EnvConfig) -> tuple[float, float]:
    """Bounds for mapping invalid-step penalties to [0, 1] (fixed penalty envelope)."""
    costs = list(config.action_costs.values()) if config.action_costs else [0.0]
    max_c = max(costs)
    min_c = min(costs)
    idle = float(config.per_step_idle_cost)
    lo = -1.0 - max_c - idle - 0.32
    hi = 0.5 + 0.4 + 0.3 + 0.3 + 0.08 - min_c - idle
    return lo, hi


def normalize_step_reward_to_unit(raw: float, config: EnvConfig) -> float:
    lo, hi = step_reward_bounds(config)
    if hi <= lo:
        return 0.5
    x = (raw - lo) / (hi - lo)
    return clip01(x)


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def grade_step(
    *,
    action: MyAction,
    chosen_email: Email,
    pending_before: List[Email],
    current_time: int,
    config: EnvConfig,
    sla_pressure_offset: int = 0,
    hidden_pending_count: int = 0,
) -> GradeBreakdown:
    eff_limit = effective_sla_limit(chosen_email, sla_pressure_offset)
    sla_breach = (current_time - chosen_email.created_time) > eff_limit
    sla_score = -0.4 if sla_breach else 0.1

    best = _best_email(pending_before, current_time)
    picked_most_urgent = best is not None and chosen_email.email_id == best.email_id
    prioritization_score = 0.2 if picked_most_urgent else 0.0

    response_score = 0.0
    if action.action_type == "reply":
        text = action.response
        if len(text) > 20 and any(c in text for c in ".!?,"):
            response_score = 0.1

    consequence_signal = 0.0
    if action.action_type == "reply" and hidden_pending_count > 0:
        consequence_signal += 0.2
    if (
        action.action_type == "escalate"
        and chosen_email.priority == EmailPriority.critical
        and remaining_sla(chosen_email, current_time) < 10
    ):
        consequence_signal += 0.15
    if action.action_type == "archive" and chosen_email.priority in (
        EmailPriority.high,
        EmailPriority.critical,
    ):
        consequence_signal -= 0.2

    action_cost = float(config.action_costs.get(action.action_type, 0.0))
    cost_penalty = -action_cost

    base = (
        sla_score
        + prioritization_score
        + response_score
        + consequence_signal
        + cost_penalty
    )
    total = max(0.0, min(1.0, (base + 0.6) / 1.2))

    return GradeBreakdown(
        sla_breach=sla_breach,
        sla_score=sla_score,
        prioritization_score=prioritization_score,
        action_score=0.0,
        response_score=response_score,
        throughput_score=0.0,
        breach_load_penalty=0.0,
        cost_penalty=cost_penalty,
        idle_penalty=0.0,
        consequence_signal=consequence_signal,
        total=total,
    )
