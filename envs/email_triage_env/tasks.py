from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openenv.core.client_types import StepResult

try:
    from .models import MyAction, MyObservation, MyState, PublicEmail
except ImportError:
    from models import MyAction, MyObservation, MyState, PublicEmail


@dataclass(frozen=True)
class TaskSpec:
    """
    A concrete objective for an agent.

    Each task defines:
    - reset-time config (seed, top_n, arrivals settings)
    - a deterministic task-level grader returning a score in [0.0, 1.0]

    Graders receive ``observation_chain``: ``[obs_0, obs_1, ...]`` where ``obs_0`` is the
    initial observation and ``obs_{i+1}`` is the observation **after** step *i* (it carries
    ``metadata`` for that step: ``grade``, ``new_emails``, errors).
    """

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    reset_config: Dict[str, Any]
    max_steps: int
    grader: Callable[
        [List[Tuple[MyObservation, Optional[MyAction]]], MyState, List[MyObservation]],
        float,
    ]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _urgency_key(email: PublicEmail, current_time: int) -> Tuple[int, int, int]:
    remaining = int(email.sla_limit) - (int(current_time) - int(email.created_time))
    priority_rank = {"high": 3, "medium": 2, "low": 1}.get(str(email.priority), 0)
    tier_rank = {"vip": 3, "premium": 2, "standard": 1}.get(str(email.customer_tier), 0)
    return (remaining, -priority_rank, -tier_rank)


def _most_urgent(pending: List[PublicEmail], current_time: int) -> Optional[PublicEmail]:
    if not pending:
        return None
    return sorted(pending, key=lambda e: _urgency_key(e, current_time))[0]


def task_easy_single_urgent_first() -> TaskSpec:
    """
    Easy: handle the single most urgent visible email correctly in the first step.
    """

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        del final_state
        if not trajectory:
            return 0.0
        first_obs, first_action = trajectory[0]
        if first_action is None:
            return 0.0
        if not first_obs.inbox:
            return 0.0

        urgent = _most_urgent(list(first_obs.inbox), int(first_obs.current_time))
        if urgent is None:
            return 0.0

        selection = 0.5 if str(first_action.email_id) == str(urgent.email_id) else 0.0

        # Use post-step observation metadata (no reliance on grader labels in state).
        post = observation_chain[1] if len(observation_chain) > 1 else None
        action_ok = 0.0
        if post is not None:
            grade = (post.metadata or {}).get("grade") or {}
            action_ok = 0.5 if float(grade.get("action_score", 0.0) or 0.0) > 0.0 else 0.0
        return _clip01(selection + action_ok)

    return TaskSpec(
        task_id="easy_single_urgent_first",
        difficulty="easy",
        description="Pick the most urgent email in the first observation and take the correct action.",
        reset_config={"top_n": 5, "seed": 0, "arrivals_enabled": False, "max_new_emails": 0},
        max_steps=1,
        grader=grader,
    )


def task_medium_sla_safe_throughput() -> TaskSpec:
    """
    Medium: clear a partially observable inbox with zero (or few) SLA breaches,
    reasonable correctness, and without excessive escalation.
    """

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        del final_state
        if not trajectory:
            return 0.0

        breaches = 0
        graded_steps = 0
        correct_actions = 0
        action_attempts = 0
        escalations = 0

        for step_idx, (_, action) in enumerate(trajectory):
            if step_idx + 1 >= len(observation_chain):
                break
            post = observation_chain[step_idx + 1]
            grade = (post.metadata or {}).get("grade") or {}
            if grade:
                graded_steps += 1
                if bool(grade.get("sla_breach", False)):
                    breaches += 1
            if action is not None:
                action_attempts += 1
                if str(action.action_type) == "escalate":
                    escalations += 1
                if float(grade.get("action_score", 0.0) or 0.0) > 0.0:
                    correct_actions += 1

        breach_rate = (breaches / graded_steps) if graded_steps else 1.0
        sla_score = 0.5 * (1.0 - breach_rate)

        acc = (correct_actions / action_attempts) if action_attempts else 0.0
        action_score = 0.3 * acc

        if escalations <= 1:
            cost_score = 0.2
        elif escalations >= 4:
            cost_score = 0.0
        else:
            cost_score = 0.2 * (1.0 - ((escalations - 1) / 3.0))

        return _clip01(sla_score + action_score + cost_score)

    return TaskSpec(
        task_id="medium_sla_safe_throughput",
        difficulty="medium",
        description="Clear the inbox with partial observability while avoiding SLA breaches and over-escalation.",
        reset_config={"top_n": 3, "seed": 1, "arrivals_enabled": False, "max_new_emails": 0},
        max_steps=30,
        grader=grader,
    )


def task_hard_dynamic_arrivals_backlog() -> TaskSpec:
    """
    Hard: handle dynamic arrivals while maintaining SLA and responding quickly to urgent new emails.
    """

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        del final_state
        if not trajectory:
            return 0.0

        breaches = 0
        graded_steps = 0
        invalid = 0

        arrival_first_seen_step: Dict[str, int] = {}
        arrival_handled_step: Dict[str, int] = {}

        for step_idx, (_, action) in enumerate(trajectory):
            if step_idx + 1 >= len(observation_chain):
                break
            post = observation_chain[step_idx + 1]
            md = post.metadata or {}
            grade = md.get("grade") or {}
            if grade:
                graded_steps += 1
                if bool(grade.get("sla_breach", False)):
                    breaches += 1

            if md.get("error") == "invalid_email_id_or_not_pending":
                invalid += 1

            for eid in (md.get("new_emails") or []):
                arrival_first_seen_step.setdefault(str(eid), step_idx + 1)

            if action is not None:
                arrival_handled_step.setdefault(str(action.email_id), step_idx + 1)

        breach_rate = (breaches / graded_steps) if graded_steps else 1.0
        sla_component = 0.4 * (1.0 - breach_rate)

        delays: List[int] = []
        for eid, seen_step in arrival_first_seen_step.items():
            handled = arrival_handled_step.get(eid)
            if handled is None:
                delays.append(999)
            else:
                delays.append(max(0, handled - seen_step))

        if not delays:
            resp_component = 0.15
        else:
            avg_delay = sum(delays) / len(delays)
            if avg_delay <= 1.0:
                resp_component = 0.3
            elif avg_delay <= 3.0:
                resp_component = 0.3 * (1.0 - ((avg_delay - 1.0) / 2.0))
            else:
                resp_component = 0.0

        action_score_sum = 0.0
        response_score_sum = 0.0
        for step_idx in range(len(trajectory)):
            if step_idx + 1 >= len(observation_chain):
                break
            post = observation_chain[step_idx + 1]
            grade = (post.metadata or {}).get("grade") or {}
            action_score_sum += float(grade.get("action_score", 0.0) or 0.0)
            response_score_sum += float(grade.get("response_score", 0.0) or 0.0)
        denom = max(1, len(trajectory))
        quality_component = 0.2 * _clip01((action_score_sum / denom) / 0.3) * 0.75 + 0.2 * _clip01(
            (response_score_sum / denom) / 0.3
        ) * 0.25

        invalid_component = max(0.0, 0.1 - 0.05 * float(invalid))

        return _clip01(sla_component + resp_component + quality_component + invalid_component)

    return TaskSpec(
        task_id="hard_dynamic_arrivals_backlog",
        difficulty="hard",
        description="Handle dynamic arrivals under partial observability; respond quickly to new urgent items and avoid SLA breaches.",
        reset_config={"top_n": 3, "seed": 2, "arrivals_enabled": True, "max_new_emails": 3},
        max_steps=30,
        grader=grader,
    )


def all_tasks() -> List[TaskSpec]:
    return [
        task_easy_single_urgent_first(),
        task_medium_sla_safe_throughput(),
        task_hard_dynamic_arrivals_backlog(),
    ]


async def rollout_task(
    *,
    env: Any,
    task: TaskSpec,
    policy: Callable[[MyObservation], Optional[MyAction]],
) -> Tuple[float, List[Tuple[MyObservation, Optional[MyAction]]], MyState, List[MyObservation]]:
    """
    Run one episode for a task using a policy.

    Returns:
        task_score, trajectory[(observation_before_action, action)], final_state,
        observation_chain (initial obs, then one obs per step after each action).
    """

    result: StepResult[MyObservation] = await env.reset(config=task.reset_config)
    obs = result.observation
    observation_chain: List[MyObservation] = [obs]
    trajectory: List[Tuple[MyObservation, Optional[MyAction]]] = []

    for _ in range(task.max_steps):
        if result.done:
            break
        action = policy(obs)
        trajectory.append((obs, action))
        if action is None:
            break
        result = await env.step(action)
        obs = result.observation
        observation_chain.append(obs)

    state: MyState = await env.state()  # type: ignore[assignment]
    score = task.grader(trajectory, state, observation_chain)
    return score, trajectory, state, observation_chain
