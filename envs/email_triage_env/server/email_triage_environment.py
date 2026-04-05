# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
EmailTriageEnvironment: OpenEnv ``Environment`` implementation.

AI-powered inbox management (multi-email triage with SLA awareness).

The environment simulates a dynamic inbox. Each step, an agent selects one pending
email and takes an action: reply, escalate, or archive. A virtual clock advances
by 1 each step, and SLA breaches are penalized.
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailStatus, EnvConfig, MyAction, MyObservation, MyState, to_public_email
except ImportError:
    from models import EmailStatus, EnvConfig, MyAction, MyObservation, MyState, to_public_email

try:
    from .dynamics import (
        apply_action_dynamics,
        maybe_generate_arrivals,
        next_email_id,
        select_top_n_pending,
    )
    from .grader import clip01, grade_step, normalize_step_reward_to_unit
    from .scenarios import arrival_templates, starter_inbox
except ImportError:  # pragma: no cover
    from server.dynamics import (
        apply_action_dynamics,
        maybe_generate_arrivals,
        next_email_id,
        select_top_n_pending,
    )
    from server.grader import clip01, grade_step, normalize_step_reward_to_unit
    from server.scenarios import arrival_templates, starter_inbox


class EmailTriageEnvironment(Environment):
    """
    Inbox triage server environment (``EmailTriageEnvironment``).

    - **State**: full inbox (all emails) + virtual clock (`current_time`)
    - **Observation**: snapshot of pending emails (currently returns full pending list)
    - **Action**: select email + action_type (reply/escalate/archive) + optional response
    - **Reward**: heuristic multi-factor score capturing SLA, prioritization, and correctness
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting its own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        self._arrival_templates = arrival_templates()
        self._starter_emails = starter_inbox()
        self._state = MyState(
            episode_id=str(uuid4()),
            step_count=0,
            current_time=0,
            emails=[],
            config=EnvConfig(),
            new_emails_added=0,
        )

    def _pending(self):
        return [e for e in self._state.emails if e.status == EmailStatus.pending]

    def _public_inbox(self, visible_emails):
        """Map internal Email rows to agent-visible PublicEmail (no grader labels)."""
        return [to_public_email(e) for e in visible_emails]

    def _observe(self) -> MyObservation:
        pending = self._pending()
        top_n = int(self._state.config.top_n)
        visible = select_top_n_pending(self._state.emails, self._state.current_time, top_n=top_n)
        hidden = max(0, len(pending) - len(visible))
        return MyObservation(
            done=len(pending) == 0,
            reward=None,
            current_time=self._state.current_time,
            inbox=self._public_inbox(visible),
            hidden_pending_count=hidden,
            last_email_id=None,
            last_action_type=None,
            sla_breach=False,
            time_advance=0,
            action_cost=0.0,
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:  # type: ignore[override]
        """
        Reset the environment.

        Returns:
            Initial inbox observation
        """
        self._reset_rubric()
        config = EnvConfig.model_validate(kwargs.get("config", {})) if "config" in kwargs else EnvConfig()
        # Allow seed passed via standard reset arg or inside config; config wins.
        effective_seed = config.seed if config.seed is not None else seed
        if effective_seed is not None:
            self._rng.seed(int(effective_seed))

        self._state = MyState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_time=0,
            emails=[e.model_copy(deep=True) for e in self._starter_emails],
            config=config,
            new_emails_added=0,
        )

        observation = self._observe()
        return self._apply_transform(observation)

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        """
        Execute one triage step:
        - Validate action (email exists + pending)
        - Compute reward (SLA + prioritization + correctness + response quality)
        - Mark email processed
        - Advance virtual time
        """
        pending_before = self._pending()
        chosen = next((e for e in pending_before if e.email_id == action.email_id), None)
        if chosen is None:
            # Invalid selection or already processed -> negative reward, still advance time.
            dyn = apply_action_dynamics(action_type=action.action_type, config=self._state.config)
            self._state.step_count += 1
            self._state.current_time += dyn.time_advance
            action_cost = float(self._state.config.action_costs.get(action.action_type, 0.0))
            observation = MyObservation(
                current_time=self._state.current_time,
                inbox=self._public_inbox(
                    select_top_n_pending(
                        self._state.emails, self._state.current_time, top_n=int(self._state.config.top_n)
                    )
                ),
                hidden_pending_count=max(0, len(self._pending()) - int(self._state.config.top_n)),
                last_email_id=action.email_id,
                last_action_type=action.action_type,
                sla_breach=False,
                done=len(self._pending()) == 0,
                reward=normalize_step_reward_to_unit(
                    -1.0 - action_cost - float(self._state.config.per_step_idle_cost),
                    self._state.config,
                ),
                time_advance=dyn.time_advance,
                action_cost=action_cost,
                metadata={"error": "invalid_email_id_or_not_pending"},
            )
            return self._apply_transform(observation)

        dyn = apply_action_dynamics(action_type=action.action_type, config=self._state.config)
        breakdown = grade_step(
            action=action,
            chosen_email=chosen,
            pending_before=pending_before,
            current_time=self._state.current_time,
            config=self._state.config,
        )
        reward = normalize_step_reward_to_unit(breakdown.total, self._state.config)
        sla_breach = breakdown.sla_breach

        # Apply action: mark email processed
        for e in self._state.emails:
            if e.email_id == chosen.email_id:
                e.status = EmailStatus.processed
                break

        # Time passes (variable duration)
        self._state.step_count += 1
        self._state.current_time += dyn.time_advance

        # Stochastic arrivals (deterministic via seed)
        remaining_budget = max(0, int(self._state.config.max_new_emails) - int(self._state.new_emails_added))
        new_emails = maybe_generate_arrivals(
            rng=self._rng,
            current_time=self._state.current_time,
            config=self._state.config,
            next_email_id=next_email_id(self._state.emails, fallback=1),
            templates=self._arrival_templates,
            remaining_budget=remaining_budget,
        )
        if new_emails:
            self._state.emails.extend(new_emails)
            self._state.new_emails_added += len(new_emails)

        pending_after = self._pending()
        visible = select_top_n_pending(
            self._state.emails, self._state.current_time, top_n=int(self._state.config.top_n)
        )
        hidden = max(0, len(pending_after) - len(visible))
        observation = MyObservation(
            current_time=self._state.current_time,
            inbox=self._public_inbox(visible),
            hidden_pending_count=hidden,
            last_email_id=chosen.email_id,
            last_action_type=action.action_type,
            sla_breach=sla_breach,
            done=len(pending_after) == 0,
            reward=reward,
            time_advance=dyn.time_advance,
            action_cost=float(self._state.config.action_costs.get(action.action_type, 0.0)),
            metadata={
                "grade": breakdown.__dict__,
                "new_emails": [e.email_id for e in new_emails],
            },
        )

        # Optional rubric override, if provided externally
        if self.rubric is not None:
            observation.reward = clip01(float(self._apply_rubric(action, observation)))
        return self._apply_transform(observation)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        When ``config.expose_grader_labels_in_state`` is False (default), emails omit
        grader-only fields so training clients can safely call ``state()`` without
        leaking the answer key.
        """
        s = self._state
        if s.config.expose_grader_labels_in_state:
            return s
        return MyState(
            episode_id=s.episode_id,
            step_count=s.step_count,
            current_time=s.current_time,
            emails=[to_public_email(e) for e in s.emails],
            config=s.config,
            new_emails_added=s.new_emails_added,
        )
