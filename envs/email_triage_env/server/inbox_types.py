from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

try:
    from ..models import CustomerTier, Email, EmailPriority
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority


@dataclass(frozen=True)
class ArrivalTemplate:
    """Fields for templated arrival emails (dataclass, not Pydantic)."""

    subject: str
    body: str
    priority: EmailPriority
    customer_tier: CustomerTier
    sla_limit: int
    ground_truth_action: Literal["reply", "escalate", "archive"]
    required_response_keywords: List[str]
    thread_id: Optional[str] = None


def make_email(
    *,
    email_id: str,
    subject: str,
    body: str,
    priority: EmailPriority,
    customer_tier: CustomerTier,
    created_time: int,
    sla_limit: int,
    thread_id: str,
    ground_truth_action: Literal["reply", "escalate", "archive"],
    required_response_keywords: List[str],
    sender: str = "support@acme.com",
) -> Email:
    return Email(
        email_id=email_id,
        thread_id=thread_id,
        subject=subject,
        body=body,
        priority=priority,
        customer_tier=customer_tier,
        created_time=created_time,
        sla_limit=sla_limit,
        ground_truth_action=ground_truth_action,
        required_response_keywords=list(required_response_keywords),
        sender=sender,
    )

