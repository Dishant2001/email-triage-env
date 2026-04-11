"""Follow-up mail, echoes, and in-place entanglement mutations on pending rows."""

from __future__ import annotations

import random
from typing import List

try:
    from ..models import CustomerTier, Email, EmailPriority, EmailStatus, EnvConfig, MyAction
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority, EmailStatus, EnvConfig, MyAction

from .dynamics import select_top_n_pending, urgency_score
from .inbox_types import make_email


def thread_key(email: Email) -> str:
    return (email.thread_id or "").strip() or email.email_id


def apply_entanglement_state_mutations(
    *,
    emails: List[Email],
    pending_before: List[Email],
    chosen: Email,
    action: MyAction,
    current_time: int,
    top_n: int,
    config: EnvConfig,
) -> None:
    if not config.entanglement_enabled:
        return

    if action.action_type == "reply":
        tid = thread_key(chosen)
        for e in emails:
            if e.status != EmailStatus.pending:
                continue
            if e.email_id == chosen.email_id:
                continue
            if thread_key(e) != tid:
                continue
            e.urgency_adjustment -= 0.15
            e.thread_context_updated = True

    elif action.action_type == "escalate":
        snd = chosen.sender
        for e in emails:
            if e.status != EmailStatus.pending:
                continue
            if e.email_id == chosen.email_id:
                continue
            if e.sender != snd:
                continue
            e.sla_limit = max(1, int(e.sla_limit) - 10)

    elif action.action_type == "archive" and chosen.ground_truth_action != "archive":
        visible = select_top_n_pending(emails, current_time, top_n=top_n)
        visible_ids = {x.email_id for x in visible}
        hidden = [
            e
            for e in pending_before
            if e.email_id not in visible_ids and e.email_id != chosen.email_id
        ]
        hidden.sort(key=lambda x: urgency_score(x, current_time), reverse=True)
        for e in hidden[:2]:
            e.sla_limit = max(1, int(e.sla_limit) - 5)


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
            sender=chosen.sender,
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
            sender=chosen.sender,
        )
    ]
