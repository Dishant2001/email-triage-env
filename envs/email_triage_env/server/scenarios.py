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
        ArrivalTemplate(
            subject="Billing: charged twice for same subscription",
            body="I see two identical charges on my card this month. Please investigate and confirm a refund timeline.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.premium,
            sla_limit=8,
            ground_truth_action="reply",
            required_response_keywords=["refund", "charge", "investigat"],
        ),
        ArrivalTemplate(
            subject="Security: suspicious login alert from unknown device",
            body="I received an alert about a login from another country. I did not authorize this—what should I do right now?",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.standard,
            sla_limit=6,
            ground_truth_action="escalate",
            required_response_keywords=[],
        ),
        ArrivalTemplate(
            subject="Shipment delayed—need updated delivery date",
            body="Order #7721 was supposed to arrive last week. Can you provide a revised ETA and tracking status?",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            sla_limit=14,
            ground_truth_action="reply",
            required_response_keywords=["tracking", "eta", "order"],
        ),
        ArrivalTemplate(
            subject="Request to delete my account and personal data",
            body="Under privacy regulations I want my account closed and my data deleted. Please confirm the process and timeline.",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            sla_limit=18,
            ground_truth_action="reply",
            required_response_keywords=["delete", "data", "confirm"],
        ),
        ArrivalTemplate(
            subject="Quarterly vendor invoice attached—payment question",
            body="We received invoice INV-9034. Which PO should this be matched to, and who approves payment?",
            priority=EmailPriority.low,
            customer_tier=CustomerTier.premium,
            sla_limit=22,
            ground_truth_action="reply",
            required_response_keywords=["invoice", "payment", "approve"],
        ),
        ArrivalTemplate(
            subject="Integration partner: webhook failures since last deploy",
            body="Our integration started failing with 500s after your Saturday deploy. Need engineering eyes and a status page update.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.premium,
            sla_limit=5,
            ground_truth_action="escalate",
            required_response_keywords=[],
        ),
        ArrivalTemplate(
            subject="Cannot access admin dashboard after role change",
            body="My admin role was updated yesterday but I still get permission denied on the dashboard. Please fix or tell me who can.",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            sla_limit=12,
            ground_truth_action="reply",
            required_response_keywords=["permission", "access", "role"],
        ),
        ArrivalTemplate(
            subject="FYI: parking policy update for next week",
            body="Facilities sent a note about visitor parking—no action needed from support, just archiving for the team.",
            priority=EmailPriority.low,
            customer_tier=CustomerTier.standard,
            sla_limit=30,
            ground_truth_action="archive",
            required_response_keywords=[],
        ),
        ArrivalTemplate(
            subject="VIP: competitor offering to migrate our workload",
            body="Our CIO is asking for an urgent executive call—we are evaluating leaving unless we get a concrete retention plan today.",
            priority=EmailPriority.high,
            customer_tier=CustomerTier.vip,
            sla_limit=4,
            ground_truth_action="escalate",
            required_response_keywords=[],
        ),
        ArrivalTemplate(
            subject="Tax form W-9 request for contractor payout",
            body="Finance needs a W-9 on file before we can release the January payout. What is the secure upload link?",
            priority=EmailPriority.medium,
            customer_tier=CustomerTier.standard,
            sla_limit=16,
            ground_truth_action="reply",
            required_response_keywords=["w-9", "upload", "secure"],
        ),
    ]

