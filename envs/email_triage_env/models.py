# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for EmailTriageEnv / EmailTriageEnvironment.

The inbox triage environment where an agent triages
multiple emails under SLA deadlines.

Agent-facing observations use :class:`PublicEmail` only (no grader labels).
Full :class:`Email` records (including ground truth) live on the server for scoring.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class MyAction(Action):
    """Select an email and take an action (reply/escalate/archive)."""

    email_id: str = Field(..., description="Target email id")
    action_type: Literal["reply", "escalate", "archive"] = Field(
        ..., description="Action to take for the selected email"
    )
    response: str = Field(
        default="",
        description="Optional response text (primarily for action_type='reply')",
    )


class EmailPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EmailStatus(str, Enum):
    pending = "pending"
    processed = "processed"


class CustomerTier(str, Enum):
    standard = "standard"
    premium = "premium"
    vip = "vip"


class PublicEmail(BaseModel):
    """Email fields exposed to policies (observations); no grader-only labels."""

    email_id: str
    thread_id: str = Field(
        default="",
        description="Thread/conversation identifier. Emails in the same thread are related.",
    )
    subject: str
    body: str
    priority: EmailPriority
    customer_tier: CustomerTier = Field(default=CustomerTier.standard)
    created_time: int
    sla_limit: int
    status: EmailStatus = EmailStatus.pending
    thread_reply_excerpt: str = Field(
        default="",
        description="Prior agent reply in this thread (if any); populated server-side for observations.",
    )


class Email(PublicEmail):
    """Full email record including deterministic labels used only for grading (server-side)."""

    ground_truth_action: Literal["reply", "escalate", "archive"] = Field(
        default="reply",
        description="Ideal action for grading; omitted from agent observations.",
    )
    required_response_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for reply scoring; omitted from agent observations.",
    )


def to_public_email(email: PublicEmail) -> PublicEmail:
    """Strip grader-only fields for observations and optional state export."""
    keys = set(PublicEmail.model_fields.keys())
    # thread_reply_excerpt is agent-facing; keep if present on Email dump
    return PublicEmail.model_validate(email.model_dump(include=keys))


class EnvConfig(BaseModel):
    """Reset-time configuration knobs to control difficulty and observability."""

    top_n: int = Field(default=5, ge=1, description="Number of pending emails shown in observation")
    arrivals_enabled: bool = Field(default=True, description="Whether new emails can arrive over time")
    max_new_emails: int = Field(default=3, ge=0, description="Max number of new emails that can arrive per episode")
    seed: Optional[int] = Field(default=None, description="Seed for deterministic dynamics")
    action_durations: Dict[str, int] = Field(
        default_factory=lambda: {"reply": 2, "escalate": 1, "archive": 1},
        description="How much virtual time each action consumes",
    )
    action_costs: Dict[str, float] = Field(
        default_factory=lambda: {"reply": 0.05, "escalate": 0.15, "archive": 0.02},
        description="Direct per-action costs to discourage overuse (e.g., escalation fatigue)",
    )
    per_step_idle_cost: float = Field(
        default=0.01, ge=0.0, description="Small cost per step to discourage unnecessary actions"
    )
    expose_grader_labels_in_state: bool = Field(
        default=False,
        description=(
            "If True, GET /state includes ground_truth_action and required_response_keywords on emails "
            "(for offline eval / debugging). If False, state emails match agent-visible PublicEmail "
            "(recommended for RL training clients that call state())."
        ),
    )
    scenario_profile: int = Field(
        default=0,
        ge=0,
        le=15,
        description=(
            "0 = fixed legacy starter inbox (reproducible benchmarks). "
            "1–15 = procedural inbox variants (composition / SLA mix) from the same seed."
        ),
    )
    reward_mode: Literal["legacy", "emergent", "hybrid"] = Field(
        default="hybrid",
        description=(
            "legacy: step reward weights oracle labels heavily. "
            "emergent: SLA, prioritization, throughput, breach load—minimal oracle. "
            "hybrid: blend (see oracle_weight)."
        ),
    )
    oracle_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="In hybrid mode, weight on ground-truth action match; remainder is emergent-only for that component.",
    )
    thread_followups_enabled: bool = Field(
        default=True,
        description="After a reply, may inject a follow-up email in the same thread.",
    )
    escalation_echo_enabled: bool = Field(
        default=True,
        description="After escalate, may inject a low-urgency internal ticket tied to the thread.",
    )
    entanglement_enabled: bool = Field(
        default=True,
        description="Bad archive (vs hidden label) increases SLA pressure on remaining pending mail.",
    )
    bad_archive_pressure_delta: int = Field(
        default=1,
        ge=0,
        description="How much effective SLA window tightens per mistaken archive (when entanglement_enabled).",
    )


class MyState(State):
    """Full environment state. ``emails`` may hold :class:`Email` instances server-side."""

    current_time: int = Field(default=0, ge=0, description="Virtual time step counter")
    emails: List[PublicEmail] = Field(
        default_factory=list,
        description="Inbox; server stores Email subclass with labels when not exported",
    )
    config: EnvConfig = Field(default_factory=EnvConfig, description="Episode configuration")
    new_emails_added: int = Field(default=0, ge=0, description="How many new emails have been injected so far")
    thread_replies: Dict[str, str] = Field(
        default_factory=dict,
        description="Latest agent reply text per thread_id (or email_id if thread empty).",
    )
    sla_pressure_offset: int = Field(
        default=0,
        ge=0,
        description="Tightens effective SLA limits for pending mail after bad decisions.",
    )
    episode_sla_breach_count: int = Field(
        default=0,
        ge=0,
        description="Running count of SLA breaches this episode (consequence signal).",
    )


class MyObservation(Observation):
    """Observation snapshot of the inbox (full or filtered) + last step info."""

    current_time: int = Field(default=0, ge=0, description="Current virtual time")
    inbox: List[PublicEmail] = Field(
        default_factory=list,
        description="Pending emails snapshot (top-N); never includes grader labels",
    )
    hidden_pending_count: int = Field(
        default=0,
        ge=0,
        description="How many pending emails exist but are hidden due to top-N observation",
    )
    last_email_id: Optional[str] = Field(
        default=None, description="Email id handled in the last action (if any)"
    )
    last_action_type: Optional[str] = Field(
        default=None, description="Action type taken in the last step (if any)"
    )
    sla_breach: bool = Field(
        default=False, description="Whether SLA was breached on last step"
    )
    time_advance: int = Field(default=0, ge=0, description="Virtual time consumed by last action")
    action_cost: float = Field(default=0.0, description="Direct cost applied for last action")


class MyReward(BaseModel):
    """
    Typed reward payload for OpenEnv spec compliance.

    Note: the OpenEnv runtime transports reward as a float, but environments can
    also expose a typed breakdown via metadata and evaluation harnesses.
    """

    total: float = Field(..., description="Total scalar reward for the step")
    sla_breach: bool = Field(..., description="Whether SLA was breached for chosen email")
    sla_score: float = Field(..., description="SLA component")
    prioritization_score: float = Field(..., description="Urgency selection component")
    action_score: float = Field(..., description="Ground-truth action match component")
    response_score: float = Field(..., description="Deterministic response rubric component")
    throughput_score: float = Field(default=0.0, description="Small bonus when SLA not breached this step")
    breach_load_penalty: float = Field(default=0.0, description="Penalty scaling with prior breaches this episode")
    cost_penalty: float = Field(..., description="Action cost penalty (negative)")
    idle_penalty: float = Field(..., description="Per-step idle penalty (negative)")
