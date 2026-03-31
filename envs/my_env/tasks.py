from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openenv.core.client_types import StepResult

from .models import Email, MyAction, MyObservation, MyState


@dataclass(frozen=True)
class TaskSpec:
    """
    A concrete objective for an agent.

    Each task defines:
    - reset-time config (seed, top_n, arrivals settings)
    - a deterministic task-level grader returning a score in [0.0, 1.0]
    """

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    reset_config: Dict[str, Any]
    max_steps: int
    grader: Callable[[List[Tuple[MyObservation, Optional[MyAction]]], MyState], float]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _urgency_key(email: Email, current_time: int) -> Tuple[int, int, int]:
    remaining = int(email.sla_limit) - (int(current_time) - int(email.created_time))
    priority_rank = {"high": 3, "medium": 2, "low": 1}.get(str(email.priority), 0)
    tier_rank = {"vip": 3, "premium": 2, "standard": 1}.get(str(email.customer_tier), 0)
    # Lower is better: remaining SLA ascending, then priority/tier descending
    return (remaining, -priority_rank, -tier_rank)


def _most_urgent(pending: List[Email], current_time: int) -> Optional[Email]:
    if not pending:
        return None
    return sorted(pending, key=lambda e: _urgency_key(e, current_time))[0]


def _pending_from_state(state: MyState) -> List[Email]:
    return [e for e in state.emails if str(e.status) == "pending"]


def task_easy_single_urgent_first() -> TaskSpec:
    """
    Easy: handle the single most urgent visible email correctly in the first step.
    """

    def grader(trajectory: List[Tuple[MyObservation, Optional[MyAction]]], final_state: MyState) -> float:
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

        # Determine ground-truth action for the chosen email by inspecting final state (it includes all emails).
        gt = None
        for e in final_state.emails:
            if str(e.email_id) == str(first_action.email_id):
                gt = str(e.ground_truth_action)
                break
        action_ok = 0.5 if (gt is not None and str(first_action.action_type) == gt) else 0.0
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

    def grader(trajectory: List[Tuple[MyObservation, Optional[MyAction]]], final_state: MyState) -> float:
        if not trajectory:
            return 0.0

        # SLA component based on per-step grade metadata.
        breaches = 0
        graded_steps = 0
        correct_actions = 0
        action_attempts = 0
        escalations = 0

        for obs, action in trajectory:
            grade = (obs.metadata or {}).get("grade") or {}
            if grade:
                graded_steps += 1
                if bool(grade.get("sla_breach", False)):
                    breaches += 1
            if action is not None:
                action_attempts += 1
                if str(action.action_type) == "escalate":
                    escalations += 1
                # action correctness from grade breakdown if present
                if float(grade.get("action_score", 0.0) or 0.0) > 0.0:
                    correct_actions += 1

        breach_rate = (breaches / graded_steps) if graded_steps else 1.0
        sla_score = 0.5 * (1.0 - breach_rate)  # 0.5 if no breaches

        acc = (correct_actions / action_attempts) if action_attempts else 0.0
        action_score = 0.3 * acc

        # Cost discipline: allow some escalation, but penalize escalating everything.
        # Full points if escalations <= 1, linearly down to 0 at >= 4.
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

    def grader(trajectory: List[Tuple[MyObservation, Optional[MyAction]]], final_state: MyState) -> float:
        if not trajectory:
            return 0.0

        breaches = 0
        graded_steps = 0
        invalid = 0

        # Track responsiveness to new arrivals: if an arrival becomes visible, how quickly is it acted on?
        arrival_first_seen_step: Dict[str, int] = {}
        arrival_handled_step: Dict[str, int] = {}

        for i, (obs, action) in enumerate(trajectory, start=1):
            md = obs.metadata or {}
            grade = md.get("grade") or {}
            if grade:
                graded_steps += 1
                if bool(grade.get("sla_breach", False)):
                    breaches += 1

            if md.get("error") == "invalid_email_id_or_not_pending":
                invalid += 1

            # Record when newly arrived emails appear.
            for eid in (md.get("new_emails") or []):
                arrival_first_seen_step.setdefault(str(eid), i)

            if action is not None:
                arrival_handled_step.setdefault(str(action.email_id), i)

        breach_rate = (breaches / graded_steps) if graded_steps else 1.0
        sla_component = 0.4 * (1.0 - breach_rate)

        # Responsiveness: for arrived emails we saw, compute delay.
        delays: List[int] = []
        for eid, seen_step in arrival_first_seen_step.items():
            handled = arrival_handled_step.get(eid)
            if handled is None:
                delays.append(999)
            else:
                delays.append(max(0, handled - seen_step))

        if not delays:
            resp_component = 0.15  # neutral small credit if no arrivals observed
        else:
            # Full credit if average delay <=1, partial if <=3, else 0.
            avg_delay = sum(delays) / len(delays)
            if avg_delay <= 1.0:
                resp_component = 0.3
            elif avg_delay <= 3.0:
                resp_component = 0.3 * (1.0 - ((avg_delay - 1.0) / 2.0))
            else:
                resp_component = 0.0

        # Action/response quality: use average action_score + response_score from grade breakdowns.
        action_score_sum = 0.0
        response_score_sum = 0.0
        for obs, _ in trajectory:
            grade = (obs.metadata or {}).get("grade") or {}
            action_score_sum += float(grade.get("action_score", 0.0) or 0.0)
            response_score_sum += float(grade.get("response_score", 0.0) or 0.0)
        denom = max(1, len(trajectory))
        quality_component = 0.2 * _clip01((action_score_sum / denom) / 0.3) * 0.75 + 0.2 * _clip01(
            (response_score_sum / denom) / 0.3
        ) * 0.25

        # Stability: penalize invalid actions and excessive steps.
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
) -> Tuple[float, List[Tuple[MyObservation, Optional[MyAction]]], MyState]:
    """
    Run one episode for a task using a policy.

    Returns:
        task_score, trajectory[(observation, action)], final_state
    """

    result: StepResult[MyObservation] = await env.reset(config=task.reset_config)
    obs = result.observation
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

    # Final observation (if last step ended episode) isn't included with an action; that's fine for deterministic graders.
    state: MyState = await env.state()  # type: ignore[assignment]
    score = task.grader(trajectory, state)
    return score, trajectory, state

