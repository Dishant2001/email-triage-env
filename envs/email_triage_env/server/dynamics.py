from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from ..models import CustomerTier, Email, EmailPriority, EmailStatus, EnvConfig
except ImportError:  # pragma: no cover
    # When uvicorn imports this package as top-level `server.*`,
    # relative import one level up may fail.
    from models import CustomerTier, Email, EmailPriority, EmailStatus, EnvConfig
from .inbox_types import ArrivalTemplate, make_email


@dataclass
class DynamicsResult:
    time_advance: int
    new_emails: List[Email]


def remaining_sla(email: Email, current_time: int) -> int:
    return email.sla_limit - (current_time - email.created_time)


def urgency_score(email: Email, current_time: int) -> float:
    # Higher is more urgent.
    remaining = remaining_sla(email, current_time)
    urgency = 1.0 / max(1, remaining + 1)
    priority_weight = {
        EmailPriority.critical: 1.2,
        EmailPriority.high: 1.0,
        EmailPriority.medium: 0.6,
        EmailPriority.low: 0.2,
    }[email.priority]
    tier_weight = {CustomerTier.vip: 1.0, CustomerTier.premium: 0.6, CustomerTier.standard: 0.2}[  # noqa: E501
        email.customer_tier
    ]
    base = 0.55 * priority_weight + 0.25 * tier_weight + 0.20 * urgency
    return base + float(getattr(email, "urgency_adjustment", 0.0))


def select_top_n_pending(emails: Iterable[Email], current_time: int, top_n: int) -> List[Email]:
    pending = [e for e in emails if e.status == EmailStatus.pending]
    pending.sort(key=lambda e: urgency_score(e, current_time), reverse=True)
    return pending[:top_n]


def maybe_generate_arrivals(
    *,
    rng: random.Random,
    current_time: int,
    config: EnvConfig,
    next_email_id: int,
    templates: List[ArrivalTemplate],
    remaining_budget: int,
    last_reply_thread_id: Optional[str] = None,
) -> List[Email]:
    if not config.arrivals_enabled or remaining_budget <= 0 or not templates:
        return []

    # Simple deterministic-ish stochastic process:
    # - At most 1 new email per step
    # - Higher chance earlier in the episode
    # - Slightly higher after agent activity in a thread (entanglement)
    p = 0.25 if current_time < 5 else 0.15
    if last_reply_thread_id:
        p = min(0.82, p + 0.12)
    if rng.random() > p:
        return []

    t = rng.choice(templates)
    thread_id = t.thread_id or last_reply_thread_id or f"t{rng.randint(1, 3)}"
    return [
        make_email(
            email_id=str(next_email_id),
            subject=t.subject,
            body=t.body,
            priority=t.priority,
            customer_tier=t.customer_tier,
            created_time=current_time,
            sla_limit=t.sla_limit,
            thread_id=thread_id,
            ground_truth_action=t.ground_truth_action,
            required_response_keywords=t.required_response_keywords,
        )
    ]


def apply_action_dynamics(
    *,
    action_type: str,
    config: EnvConfig,
) -> DynamicsResult:
    time_advance = int(config.action_durations.get(action_type, 1))
    if time_advance < 1:
        time_advance = 1
    return DynamicsResult(time_advance=time_advance, new_emails=[])


def next_email_id(emails: List[Email], fallback: int = 1) -> int:
    max_id = 0
    for e in emails:
        try:
            max_id = max(max_id, int(e.email_id))
        except Exception:
            continue
    return max(max_id + 1, fallback)

