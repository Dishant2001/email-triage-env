"""Causal follow-ups and entanglement: replies/escalations shape future inbox (not isolated items)."""

from __future__ import annotations

import random
from typing import List

try:
    from ..models import CustomerTier, Email, EmailPriority, EnvConfig, MyAction
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority, EnvConfig, MyAction

from .inbox_types import make_email


def thread_key(email: Email) -> str:
    return (email.thread_id or "").strip() or email.email_id


def maybe_reply_followup(
    *,
    rng: random.Random,
    config: EnvConfig,
    chosen: Email,
    action: MyAction,
    next_email_id: int,
    current_time: int,
) -> List[Email]:
    if not config.thread_followups_enabled or action.action_type != "reply":
        return []
    if rng.random() > 0.55:
        return []
    tid = thread_key(chosen)
    subj = f"Re: {chosen.subject[:60]}"
    body = (
        f"Thanks for your reply. One more question on thread {tid}: "
        f"can you confirm the next step and expected timeline?"
    )
    return [
        make_email(
            email_id=str(next_email_id),
            thread_id=tid,
            subject=subj[:120],
            body=body,
            priority=EmailPriority.medium,
            customer_tier=chosen.customer_tier,
            created_time=current_time,
            sla_limit=max(5, min(14, chosen.sla_limit)),
            ground_truth_action="reply",
            required_response_keywords=["confirm", "timeline", "next"],
        )
    ]


def maybe_escalation_echo(
    *,
    rng: random.Random,
    config: EnvConfig,
    chosen: Email,
    action: MyAction,
    next_email_id: int,
    current_time: int,
) -> List[Email]:
    if not config.escalation_echo_enabled or action.action_type != "escalate":
        return []
    if rng.random() > 0.45:
        return []
    tid = thread_key(chosen)
    return [
        make_email(
            email_id=str(next_email_id),
            thread_id=tid,
            subject=f"[Internal] Escalation logged for {tid}",
            body=(
                "On-call ticket opened from customer thread. "
                "Please monitor for customer follow-up in the same thread."
            ),
            priority=EmailPriority.low,
            customer_tier=CustomerTier.standard,
            created_time=current_time,
            sla_limit=20,
            ground_truth_action="archive",
            required_response_keywords=[],
        )
    ]
