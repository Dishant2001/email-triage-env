from __future__ import annotations

from typing import List

try:
    from ..models import CustomerTier, Email, EmailPriority
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority
from .inbox_types import ArrivalTemplate, make_email


def starter_inbox() -> List[Email]:
    return [
        make_email(
            email_id="1",
            thread_id="t1",
            subject="Urgent client issue: checkout failing",
            body="Payments failing for multiple users. Need immediate investigation and ETA.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.premium,
            created_time=0,
            sla_limit=5,
            ground_truth_action="escalate",
            required_response_keywords=["eta", "investigat", "sorry"],
        ),
        make_email(
            email_id="2",
            thread_id="t2",
            subject="FYI: meeting notes from yesterday",
            body="Sharing notes. No action required; archive if reviewed.",
            priority=EmailPriority.low,
            customer_tier=CustomerTier.standard,
            created_time=0,
            sla_limit=20,
            ground_truth_action="archive",
            required_response_keywords=[],
        ),
        make_email(
            email_id="3",
            thread_id="t3",
            subject="Refund request - order #1842",
            body="Requesting refund due to damaged item. Ask for photos and confirm timeline.",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            created_time=1,
            sla_limit=10,
            ground_truth_action="reply",
            required_response_keywords=["refund", "photos", "timeline"],
        ),
        make_email(
            email_id="4",
            thread_id="t1",
            subject="VIP escalation: outage reported",
            body="VIP account reports outage. Please escalate to on-call immediately.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.vip,
            created_time=2,
            sla_limit=6,
            ground_truth_action="escalate",
            required_response_keywords=[],
        ),
    ]


def arrival_templates() -> List[ArrivalTemplate]:
    return [
        ArrivalTemplate(
            subject="Password reset not working",
            body="User cannot reset password. Please advise next steps.",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            sla_limit=12,
            ground_truth_action="reply",
            required_response_keywords=["reset", "link"],
        ),
        ArrivalTemplate(
            subject="VIP: production down",
            body="Production appears down for VIP customer; needs immediate escalation.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.vip,
            sla_limit=4,
            ground_truth_action="escalate",
            required_response_keywords=[],
        ),
        ArrivalTemplate(
            subject="Newsletter subscription confirmation",
            body="Customer asks if subscribed; low urgency.",
            priority=EmailPriority.low,
            customer_tier=CustomerTier.standard,
            sla_limit=25,
            ground_truth_action="archive",
            required_response_keywords=[],
        ),
    ]

