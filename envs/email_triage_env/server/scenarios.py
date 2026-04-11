from __future__ import annotations

import random
from collections import Counter
from typing import List, Literal

try:
    from ..models import CustomerTier, Email, EmailPriority
except ImportError:  # pragma: no cover
    from models import CustomerTier, Email, EmailPriority
from .inbox_types import ArrivalTemplate, make_email

DOMAIN_KEYWORDS: List[str] = [
    "ticket",
    "resolved",
    "ETA",
    "escalated",
    "team",
    "investigating",
    "update",
    "acknowledged",
    "priority",
    "deadline",
    "follow-up",
    "reviewed",
    "confirmed",
    "assigned",
    "closed",
]

SENDER_POOL: List[str] = [
    "support@acme.com",
    "billing@globex.com",
    "ops@initech.net",
    "ceo@umbrella.org",
    "noreply@hooli.io",
    "admin@piedpiper.com",
]


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
            sender="billing@globex.com",
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
            sender="noreply@hooli.io",
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
            sender="support@acme.com",
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
            sender="ceo@umbrella.org",
        ),
    ]


def _assign_thread_ids(
    rng: random.Random, n: int, profile: int
) -> List[str]:
    """Return one thread id per email index; clusters only for medium/high bands."""
    if 1 <= profile <= 5:
        return [f"thread-{i}" for i in range(n)]

    num_clusters = 1
    if 6 <= profile <= 10:
        num_clusters = rng.randint(1, 2)
    else:
        num_clusters = rng.randint(2, 3)

    idx = list(range(n))
    rng.shuffle(idx)
    tids = [f"thread-{i}" for i in range(n)]
    ptr = 0
    c_id = 0
    while ptr < n - 1 and num_clusters > 0:
        size = min(rng.randint(2, 3), n - ptr)
        if size < 2:
            ptr += 1
            continue
        label = f"tcl{c_id}-{rng.randint(0, 9999)}"
        c_id += 1
        for k in range(size):
            tids[idx[ptr + k]] = label
        ptr += size
        num_clusters -= 1
    return tids


def _sample_priorities(rng: random.Random, n: int, profile: int) -> List[EmailPriority]:
    if 1 <= profile <= 5:
        level_count = rng.randint(1, 2)
        pool = rng.sample([EmailPriority.low, EmailPriority.medium], k=level_count)
        return [rng.choice(pool) for _ in range(n)]

    if 6 <= profile <= 10:
        pool = [EmailPriority.low, EmailPriority.medium, EmailPriority.high]
        prios = [rng.choice(pool) for _ in range(n)]
        return prios

    weights = [0.12, 0.28, 0.60]
    levels = [EmailPriority.medium, EmailPriority.high, EmailPriority.critical]
    return [rng.choices(levels, weights=weights, k=1)[0] for _ in range(n)]


def _sample_sla(rng: random.Random, priority: EmailPriority, profile: int) -> int:
    if 1 <= profile <= 5:
        return rng.randint(30, 60)

    if 6 <= profile <= 10:
        if rng.random() < 0.42:
            return rng.randint(5, 14)
        return rng.randint(12, 40)

    return rng.randint(5, 20)


def _sample_ground_truth(
    rng: random.Random, priority: EmailPriority
) -> Literal["reply", "escalate", "archive"]:
    r = rng.random()
    if r < 0.5:
        return "reply"
    if r < 0.75:
        return "escalate"
    if priority in (EmailPriority.high, EmailPriority.critical):
        return "reply" if rng.random() < 0.5 else "escalate"
    return "archive"


def _reply_keywords(rng: random.Random, ground_truth: str) -> List[str]:
    if ground_truth != "reply":
        return []
    k = rng.randint(2, min(4, len(DOMAIN_KEYWORDS)))
    return rng.sample(DOMAIN_KEYWORDS, k)


def _assign_senders(rng: random.Random, n: int, profile: int) -> List[str]:
    ns = rng.randint(3, min(6, len(SENDER_POOL)))
    pool = rng.sample(SENDER_POOL, k=ns)

    if 11 <= profile <= 15:
        weights = [rng.random() + 0.5 for _ in pool]
        return [rng.choices(pool, weights=weights, k=1)[0] for _ in range(n)]

    return [rng.choice(pool) for _ in range(n)]


def _enforce_medium_tight_sla(
    rng: random.Random, slas: List[int], priorities: List[EmailPriority], profile: int
) -> None:
    if not (6 <= profile <= 10):
        return
    if any(s < 15 for s in slas):
        return
    i = rng.randrange(len(slas))
    slas[i] = rng.randint(5, 14)


def _enforce_high_sender_repeat(senders: List[str], profile: int, rng: random.Random) -> None:
    """Ensure at least two distinct senders each appear on 2+ emails (high band)."""
    if not (11 <= profile <= 15) or len(senders) < 4:
        return
    counts = Counter(senders)
    if sum(1 for c in counts.values() if c >= 2) >= 2:
        return
    idx = list(range(len(senders)))
    rng.shuffle(idx)
    a, b = rng.sample(SENDER_POOL, k=2)
    senders[idx[0]] = a
    senders[idx[1]] = a
    senders[idx[2]] = b
    senders[idx[3]] = b


def generate_starter_inbox(*, seed: int, profile: int) -> List[Email]:
    """
    Parametric starter inbox. ``profile == 0`` returns the fixed legacy fixture.

    Profiles 1–5: low complexity; 6–10: medium (clusters, mixed SLAs);
    11–15: high (critical-heavy, tight SLAs, sender repetition for entanglement).

    Randomness uses ``random.Random(seed)`` only so the result is independent of
    any other RNG in the process.
    """
    if profile == 0:
        return starter_inbox()

    rng = random.Random(int(seed))
    if 1 <= profile <= 5:
        n = rng.randint(4, 6)
    elif 6 <= profile <= 10:
        n = rng.randint(7, 10)
    else:
        n = rng.randint(10, 14)

    thread_ids = _assign_thread_ids(rng, n, profile)
    priorities = _sample_priorities(rng, n, profile)
    slas = [_sample_sla(rng, priorities[i], profile) for i in range(n)]
    _enforce_medium_tight_sla(rng, slas, priorities, profile)
    senders = _assign_senders(rng, n, profile)
    _enforce_high_sender_repeat(senders, profile, rng)

    tiers = list(CustomerTier)
    emails: List[Email] = []
    for i in range(n):
        prio = priorities[i]
        sla = max(1, slas[i])
        gt = _sample_ground_truth(rng, prio)
        kw = _reply_keywords(rng, gt)
        created = rng.randint(0, min(3, max(0, sla // 4)))
        subj = f"Inbox item {i + 1} (ref {rng.randint(10000, 99999)})"
        body = (
            f"Message from {senders[i]} regarding thread {thread_ids[i]}. "
            f"Please triage with SLA window {sla}."
        )
        emails.append(
            make_email(
                email_id=str(i + 1),
                thread_id=thread_ids[i],
                subject=subj[:120],
                body=body[:500],
                priority=prio,
                customer_tier=rng.choice(tiers),
                created_time=created,
                sla_limit=sla,
                ground_truth_action=gt,
                required_response_keywords=kw,
                sender=senders[i],
            )
        )
    return emails


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

