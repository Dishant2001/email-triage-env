# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv inbox triage: pick a pending email, reply / escalate / archive, clock advances per step."""

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
    from .consequences import (
        apply_entanglement_state_mutations,
        maybe_escalation_echo,
        maybe_reply_followup,
        thread_key,
    )
    from .dynamics import (
        apply_action_dynamics,
        maybe_generate_arrivals,
        next_email_id,
        select_top_n_pending,
    )
    from .grader import clip01, grade_step, normalize_step_reward_to_unit
    from .scenarios import arrival_templates, generate_starter_inbox, starter_inbox
except ImportError:  # pragma: no cover
    from server.consequences import (
        apply_entanglement_state_mutations,
        maybe_escalation_echo,
        maybe_reply_followup,
        thread_key,
    )
    from server.dynamics import (
        apply_action_dynamics,
        maybe_generate_arrivals,
        next_email_id,
        select_top_n_pending,
    )
    from server.grader import clip01, grade_step, normalize_step_reward_to_unit
    from server.scenarios import arrival_templates, generate_starter_inbox, starter_inbox


class EmailTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        self._arrival_templates = arrival_templates()
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

    def _visible_public_rows(self, visible_emails):
        replies = self._state.thread_replies
        out: list = []
        for e in visible_emails:
            pub = to_public_email(e)
            key = thread_key(e)
            excerpt = replies.get(key, "")
            out.append(pub.model_copy(update={"thread_reply_excerpt": excerpt}))
        return out

    def _observe(self) -> MyObservation:
        pending = self._pending()
        top_n = int(self._state.config.top_n)
        visible = select_top_n_pending(self._state.emails, self._state.current_time, top_n=top_n)
        hidden = max(0, len(pending) - len(visible))
        return MyObservation(
            done=len(pending) == 0,
            reward=None,
            current_time=self._state.current_time,
            inbox=self._visible_public_rows(visible),
            hidden_pending_count=hidden,
            last_email_id=None,
            last_action_type=None,
            sla_breach=False,
            time_advance=0,
            action_cost=0.0,
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:  # type: ignore[override]
        self._reset_rubric()
        config = EnvConfig.model_validate(kwargs.get("config", {})) if "config" in kwargs else EnvConfig()
        effective_seed = config.seed if config.seed is not None else seed
        if effective_seed is not None:
            self._rng.seed(int(effective_seed))

        if config.scenario_profile == 0:
            starter = [e.model_copy(deep=True) for e in starter_inbox()]
        else:
            seed_for_scenario = int(effective_seed) if effective_seed is not None else 0
            starter = [e.model_copy(deep=True) for e in generate_starter_inbox(seed=seed_for_scenario, profile=config.scenario_profile)]

        self._state = MyState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_time=0,
            emails=starter,
            config=config,
            new_emails_added=0,
            thread_replies={},
            sla_pressure_offset=0,
            episode_sla_breach_count=0,
        )

        observation = self._observe()
        return self._apply_transform(observation)

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        pending_before = self._pending()
        chosen = next((e for e in pending_before if e.email_id == action.email_id), None)
        if chosen is None:
            dyn = apply_action_dynamics(action_type=action.action_type, config=self._state.config)
            self._state.step_count += 1
            self._state.current_time += dyn.time_advance
            action_cost = float(self._state.config.action_costs.get(action.action_type, 0.0))
            observation = MyObservation(
                current_time=self._state.current_time,
                inbox=self._visible_public_rows(
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
        visible_pre = select_top_n_pending(
            self._state.emails,
            self._state.current_time,
            top_n=int(self._state.config.top_n),
        )
        hidden_pending_count = max(0, len(pending_before) - len(visible_pre))
        breakdown = grade_step(
            action=action,
            chosen_email=chosen,
            pending_before=pending_before,
            current_time=self._state.current_time,
            config=self._state.config,
            sla_pressure_offset=self._state.sla_pressure_offset,
            hidden_pending_count=hidden_pending_count,
        )
        reward = float(breakdown.total)
        sla_breach = breakdown.sla_breach

        if breakdown.sla_breach:
            self._state.episode_sla_breach_count += 1

        if action.action_type == "reply" and action.response.strip():
            tid = thread_key(chosen)
            text = action.response.strip()
            self._state.thread_replies[tid] = text[:480]

        if (
            self._state.config.entanglement_enabled
            and action.action_type == "archive"
            and chosen.ground_truth_action != "archive"
        ):
            self._state.sla_pressure_offset += int(self._state.config.bad_archive_pressure_delta)

        apply_entanglement_state_mutations(
            emails=self._state.emails,
            pending_before=pending_before,
            chosen=chosen,
            action=action,
            current_time=self._state.current_time,
            top_n=int(self._state.config.top_n),
            config=self._state.config,
        )

        for e in self._state.emails:
            if e.email_id == chosen.email_id:
                e.status = EmailStatus.processed
                break

        self._state.step_count += 1
        self._state.current_time += dyn.time_advance

        last_thread: str | None = None
        if action.action_type in ("reply", "escalate"):
            last_thread = thread_key(chosen) or None

        start_c = next_email_id(self._state.emails, fallback=1)
        consequence_batch: list = []
        consequence_batch.extend(
            maybe_reply_followup(
                rng=self._rng,
                config=self._state.config,
                chosen=chosen,
                action=action,
                next_email_id=start_c,
                current_time=self._state.current_time,
            )
        )
        nxt = start_c + len(consequence_batch)
        consequence_batch.extend(
            maybe_escalation_echo(
                rng=self._rng,
                config=self._state.config,
                chosen=chosen,
                action=action,
                next_email_id=nxt,
                current_time=self._state.current_time,
            )
        )
        if consequence_batch:
            self._state.emails.extend(consequence_batch)
            self._state.new_emails_added += len(consequence_batch)

        arrival_next_id = next_email_id(self._state.emails, fallback=1)
        remaining_budget = max(0, int(self._state.config.max_new_emails) - int(self._state.new_emails_added))
        new_emails = maybe_generate_arrivals(
            rng=self._rng,
            current_time=self._state.current_time,
            config=self._state.config,
            next_email_id=arrival_next_id,
            templates=self._arrival_templates,
            remaining_budget=remaining_budget,
            last_reply_thread_id=last_thread,
        )
        if new_emails:
            self._state.emails.extend(new_emails)
            self._state.new_emails_added += len(new_emails)

        pending_after = self._pending()
        visible = select_top_n_pending(
            self._state.emails, self._state.current_time, top_n=int(self._state.config.top_n)
        )
        hidden = max(0, len(pending_after) - len(visible))
        injected_ids = [e.email_id for e in consequence_batch] + [e.email_id for e in new_emails]
        observation = MyObservation(
            current_time=self._state.current_time,
            inbox=self._visible_public_rows(visible),
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
                "new_emails": injected_ids,
                "episode_stats": {
                    "sla_breach_count": self._state.episode_sla_breach_count,
                    "sla_pressure_offset": self._state.sla_pressure_offset,
                },
            },
        )

        if self.rubric is not None:
            observation.reward = clip01(float(self._apply_rubric(action, observation)))
        return self._apply_transform(observation)

    @property
    def state(self) -> State:
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
            thread_replies=dict(s.thread_replies),
            sla_pressure_offset=s.sla_pressure_offset,
            episode_sla_breach_count=s.episode_sla_breach_count,
        )
