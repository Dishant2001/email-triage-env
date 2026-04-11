from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

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


def _response_quality_score(action: MyAction, email: Email) -> float:
    """
    Reply rubric: keyword coverage + structure (anti keyword-dump).
    Max contribution 0.3 to match legacy scale.
    """
    if action.action_type != "reply":
        return 0.0
    text = action.response.strip()
    if len(text) < 8:
        return 0.0
    words = text.split()
    lower = text.lower()
    has_sentence_shape = any(ch in text for ch in ".!?\n") and len(words) >= 6
    structure = 1.0 if has_sentence_shape else 0.35

    required = [k.strip().lower() for k in email.required_response_keywords if k.strip()]
    if not required:
        base = 0.12 * structure
        if len(words) >= 12:
            base += 0.12
        return min(0.3, base)

    hits = sum(1 for k in required if k in lower)
    kw_frac = hits / max(1, len(required))
    kw_part = 0.18 * kw_frac
    if len(words) < 6 and hits == len(required):
        kw_part *= 0.35
    struct_part = 0.12 * structure
    return min(0.3, kw_part + struct_part)


def _grade_step_emergent(
    *,
    action: MyAction,
    chosen_email: Email,
    pending_before: List[Email],
    current_time: int,
    config: EnvConfig,
    sla_pressure_offset: int,
    hidden_pending_count: int,
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


def grade_step(
    *,
    action: MyAction,
    chosen_email: Email,
    pending_before: List[Email],
    current_time: int,
    config: EnvConfig,
    sla_pressure_offset: int = 0,
    episode_sla_breach_count: int = 0,
    reward_mode: Literal["legacy", "emergent", "hybrid"] = "hybrid",
    oracle_weight: float = 0.35,
    hidden_pending_count: int = 0,
) -> GradeBreakdown:
    if reward_mode == "emergent":
        return _grade_step_emergent(
            action=action,
            chosen_email=chosen_email,
            pending_before=pending_before,
            current_time=current_time,
            config=config,
            sla_pressure_offset=sla_pressure_offset,
            hidden_pending_count=hidden_pending_count,
        )

    eff_limit = effective_sla_limit(chosen_email, sla_pressure_offset)
    sla_breach = (current_time - chosen_email.created_time) > eff_limit
    sla_score = -1.0 if sla_breach else 0.5

    prioritization_score = 0.0
    best = _best_email(pending_before, current_time)
    if best is not None and chosen_email.email_id == best.email_id:
        prioritization_score = 0.4

    oracle_match = 1.0 if action.action_type == chosen_email.ground_truth_action else 0.0
    if reward_mode == "legacy":
        action_score = 0.3 * oracle_match
    else:
        action_score = 0.3 * float(oracle_weight) * oracle_match

    response_score = _response_quality_score(action, chosen_email)

    throughput_score = 0.08 if not sla_breach else 0.0
    breach_load_penalty = -0.04 * float(min(8, max(0, episode_sla_breach_count)))

    if reward_mode == "legacy":
        throughput_score = 0.0
        breach_load_penalty = 0.0

    action_cost = float(config.action_costs.get(action.action_type, 0.0))
    cost_penalty = -action_cost
    idle_penalty = -float(config.per_step_idle_cost)

    total = (
        sla_score
        + prioritization_score
        + action_score
        + response_score
        + throughput_score
        + breach_load_penalty
        + cost_penalty
        + idle_penalty
    )

    return GradeBreakdown(
        sla_breach=sla_breach,
        sla_score=sla_score,
        prioritization_score=prioritization_score,
        action_score=action_score,
        response_score=response_score,
        throughput_score=throughput_score,
        breach_load_penalty=breach_load_penalty,
        cost_penalty=cost_penalty,
        idle_penalty=idle_penalty,
        consequence_signal=0.0,
        total=total,
    )
