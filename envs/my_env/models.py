# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is an inbox management environment where an agent triages
multiple emails under SLA deadlines.
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


class Email(BaseModel):
    """Email item in the inbox."""

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
    ground_truth_action: Literal["reply", "escalate", "archive"] = Field(
        default="reply",
        description="Deterministic ideal action for grading (environment-internal label).",
    )
    required_response_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords required in a good reply (simple deterministic response rubric).",
    )


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


class MyState(State):
    """Full environment state (inbox + virtual clock)."""

    current_time: int = Field(default=0, ge=0, description="Virtual time step counter")
    emails: List[Email] = Field(default_factory=list, description="Full inbox contents")
    config: EnvConfig = Field(default_factory=EnvConfig, description="Episode configuration")
    new_emails_added: int = Field(default=0, ge=0, description="How many new emails have been injected so far")


class MyObservation(Observation):
    """Observation snapshot of the inbox (full or filtered) + last step info."""

    current_time: int = Field(default=0, ge=0, description="Current virtual time")
    inbox: List[Email] = Field(default_factory=list, description="Pending emails snapshot (top-N)")
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
    cost_penalty: float = Field(..., description="Action cost penalty (negative)")
    idle_penalty: float = Field(..., description="Per-step idle penalty (negative)")
