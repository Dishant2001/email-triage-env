from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

try:
    from ..models import CustomerTier, Email, EmailPriority
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority


@dataclass(frozen=True)
class ArrivalTemplate:
    """Template for stochastic email arrivals.

    We keep this separate from the Pydantic models to avoid binding environment
    logic directly to validation concerns.
    """

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
    )

