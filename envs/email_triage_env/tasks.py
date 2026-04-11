from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openenv.core.client_types import StepResult

try:
    from .models import MyAction, MyObservation, MyState
except ImportError:
    from models import MyAction, MyObservation, MyState


@dataclass(frozen=True)
class TaskSpec:
    """Episode benchmark: ``reset_config``, ``max_steps``, and ``grader`` (raw score in [0, 1])."""

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    reset_config: Dict[str, Any]
    max_steps: int
    grader: Callable[
        [List[Tuple[MyObservation, Optional[MyAction]]], MyState, List[MyObservation]],
        float,
    ]


TASK_SCORE_HARNESS_MARGIN: float = 1e-3


def harness_task_score(raw: float) -> float:
    """Affine map: raw grader score in [0, 1] -> (margin, 1-margin) for strict open-interval harnesses."""
    m = TASK_SCORE_HARNESS_MARGIN
    x = float(raw)
    if not math.isfinite(x):
        x = 0.0
    x = max(0.0, min(1.0, x))
    out = m + (1.0 - 2.0 * m) * x
    return min(1.0 - m, max(m, out))


_to_open_interval = harness_task_score


def _consequence_signal(obs: Any) -> float:
    try:
        return float((obs.metadata or {}).get("grade", {}).get("consequence_signal", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _new_email_id(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("email_id", "")).strip()
    return str(entry).strip()


def _arrival_responsiveness(
    observation_chain: List[MyObservation],
    trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
) -> float:
    """Mean latency from arrival (metadata new_emails) to handling; maps to [0, 1]."""
    arrivals: Dict[str, int] = {}
    for step_idx, obs in enumerate(observation_chain[1:], start=1):
        new = (obs.metadata or {}).get("new_emails") or []
        for e in new:
            eid = _new_email_id(e)
            if eid and eid not in arrivals:
                arrivals[eid] = step_idx
    if not arrivals:
        return 1.0
    latencies: List[float] = []
    for eid, arrived_step in arrivals.items():
        handled: Optional[int] = None
        for i, (_, act) in enumerate(trajectory[arrived_step:], start=arrived_step):
            if act is not None and str(act.email_id).strip() == eid:
                handled = i
                break
        latencies.append(float(handled - arrived_step) if handled is not None else 10.0)
    return max(0.0, 1.0 - (sum(latencies) / len(latencies)) / 10.0)


def task_easy_single_urgent_first() -> TaskSpec:
    """Easy: first-step pick matches top visible row (urgency-sorted) plus observable reply / archive hygiene."""

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        score = 0.0
        if not observation_chain or not observation_chain[0].inbox:
            return score
        initial_obs = observation_chain[0]
        top_id = str(initial_obs.inbox[0].email_id)

        if len(trajectory) >= 1 and trajectory[0][1] is not None:
            action = trajectory[0][1]
            if str(action.email_id) == top_id:
                score += 0.4
            if action.action_type == "reply":
                r = action.response or ""
                if len(r) > 20 and any(c in r for c in ".!?,"):
                    score += 0.35
            else:
                score += 0.35
            priority = str(getattr(initial_obs.inbox[0], "priority", "")).lower()
            if action.action_type != "archive" or priority not in ("high", "critical"):
                score += 0.25

        return score

    return TaskSpec(
        task_id="easy_single_urgent_first",
        difficulty="easy",
        description=(
            "First step: top-visible pick (0.4), structured reply or non-reply credit (0.35), "
            "no archive on high/critical top row (0.25)."
        ),
        reset_config={
            "top_n": 5,
            "seed": 0,
            "arrivals_enabled": False,
            "max_new_emails": 0,
        },
        max_steps=1,
        grader=grader,
    )


def task_medium_sla_safe_throughput() -> TaskSpec:
    """Medium benchmark task."""

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        score = 0.0
        if not trajectory:
            return score

        breach_count = 0
        total_steps = 0
        for obs in observation_chain[1:]:
            total_steps += 1
            if getattr(obs, "sla_breach", False):
                breach_count += 1
        breach_rate = breach_count / max(1, total_steps)
        if breach_rate < 0.2:
            score += 0.4
        else:
            score += max(0.0, 0.4 * (1.0 - (breach_rate - 0.2) / 0.8))

        initial_inbox_size = len(observation_chain[0].inbox)
        if getattr(final_state.config, "arrivals_enabled", False):
            new_email_count = sum(
                len((obs.metadata or {}).get("new_emails") or []) for obs in observation_chain[1:]
            )
        else:
            new_email_count = 0
        total_to_handle = max(1, initial_inbox_size + new_email_count)
        handled = len([t for t in trajectory if t[1] is not None])
        score += min(0.4, 0.4 * handled / total_to_handle)

        reply_actions = [t[1] for t in trajectory if t[1] is not None and t[1].action_type == "reply"]
        if reply_actions:
            good = sum(
                1
                for a in reply_actions
                if len(a.response or "") > 20 and any(c in (a.response or "") for c in ".!?,")
            )
            score += 0.2 * (good / len(reply_actions))
        else:
            score += 0.1

        return score

    return TaskSpec(
        task_id="medium_sla_safe_throughput",
        difficulty="medium",
        description="Clear the inbox with low SLA breach rate, throughput, and structured replies.",
        reset_config={
            "top_n": 3,
            "seed": 1,
            "arrivals_enabled": False,
            "max_new_emails": 0,
        },
        max_steps=30,
        grader=grader,
    )


def task_hard_dynamic_arrivals_backlog() -> TaskSpec:
    """Hard benchmark task."""

    def grader(
        trajectory: List[Tuple[MyObservation, Optional[MyAction]]],
        final_state: MyState,
        observation_chain: List[MyObservation],
    ) -> float:
        score = 0.0
        if not trajectory:
            return score

        breach_count = 0
        total_steps = 0
        for obs in observation_chain[1:]:
            total_steps += 1
            if getattr(obs, "sla_breach", False):
                breach_count += 1
        breach_rate = breach_count / max(1, total_steps)
        if breach_rate < 0.25:
            score += 0.3
        else:
            score += max(0.0, 0.3 * (1.0 - (breach_rate - 0.25) / 0.75))

        score += 0.3 * _arrival_responsiveness(observation_chain, trajectory)

        pressure_steps = sum(
            1
            for i, (_, act) in enumerate(trajectory)
            if act is not None
            and i < len(observation_chain)
            and int(getattr(observation_chain[i], "hidden_pending_count", 0)) > 0
        )
        total_tr = max(1, len(trajectory))
        score += 0.2 * (pressure_steps / total_tr)

        signals = [_consequence_signal(obs) for obs in observation_chain[1:]]
        if signals:
            mean_cs = sum(signals) / len(signals)
            score += 0.2 * min(1.0, mean_cs / 0.1)

        return score

    return TaskSpec(
        task_id="hard_dynamic_arrivals_backlog",
        difficulty="hard",
        description="Dynamic arrivals, SLA, responsiveness, pressure steps, consequence signal.",
        reset_config={
            "top_n": 3,
            "seed": 2,
            "arrivals_enabled": True,
            "max_new_emails": 3,
        },
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
    raw = float(task.grader(trajectory, state, observation_chain))
    score = harness_task_score(raw)
    return score, trajectory, state, observation_chain
