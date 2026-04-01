"""
Inbox Environment Evaluator
==========================

This script runs one or more evaluation episodes against the MyEnv inbox server.

It is intentionally structured like `sample_inference_script.py`:
- create client
- reset to get first observation
- loop: build prompt from observation -> get action -> step -> log -> repeat

Supports two agent modes:
1) LLM policy (if OPENAI_API_KEY + MODEL_NAME are configured)
2) Deterministic fallback rule policy (no external model required)

Environment dynamics (including incoming email arrivals) are controlled via reset
`config` (seed, arrivals_enabled, max_new_emails, top_n, costs, durations).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from envs.my_env.client import MyEnv
from envs.my_env.models import MyAction, MyObservation, PublicEmail
from envs.my_env.tasks import TaskSpec, all_tasks, rollout_task


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional override
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL")

BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"

MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")
TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "220")

JSON_ACTION_RE = re.compile(r"\{[\s\S]*\}")


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an inbox triage agent.

    You will receive:
    - current_time
    - a list of visible pending emails (top-N)
    - hidden_pending_count
    - step history (recent)

    You must output EXACTLY one JSON object, with keys:
    - "email_id": string (must be one of the visible emails' email_id)
    - "action_type": one of "reply", "escalate", "archive"
    - "response": string (optional, but required for good "reply" actions)

    Rules:
    - Prefer handling urgent items first: VIP/high priority and near SLA.
    - Avoid SLA breaches.
    - Escalate sparingly: it has a cost.
    - If you reply, include concise helpful content.
    - Do NOT include any text outside the JSON object.
    """
).strip()


@dataclass
class EpisodeSummary:
    task_id: str
    difficulty: str
    steps: int
    total_reward: float
    task_score: float
    sla_breaches: int
    invalid_actions: int


def _remaining_sla(email: PublicEmail, current_time: int) -> int:
    return int(email.sla_limit) - (int(current_time) - int(email.created_time))


def _urgency_sort_key(email: PublicEmail, current_time: int) -> Tuple[int, int, int]:
    # Lower key is better after sort:
    # - remaining SLA ascending (smaller remaining is more urgent)
    # - priority descending
    # - tier descending
    rem = _remaining_sla(email, current_time)
    priority_rank = {"high": 3, "medium": 2, "low": 1}.get(str(email.priority), 0)
    tier_rank = {"vip": 3, "premium": 2, "standard": 1}.get(str(email.customer_tier), 0)
    return (rem, -priority_rank, -tier_rank)


def build_user_prompt(step_idx: int, obs: MyObservation, history: List[str]) -> str:
    emails_lines: List[str] = []
    for e in obs.inbox:
        emails_lines.append(
            f"- id={e.email_id} tier={e.customer_tier} priority={e.priority} "
            f"created={e.created_time} sla={e.sla_limit} remaining={_remaining_sla(e, obs.current_time)} "
            f"subject={e.subject!r}"
        )

    history_block = "\n".join(history[-6:]) if history else "None"

    prompt = textwrap.dedent(
        f"""
        Step: {step_idx}
        current_time: {obs.current_time}
        hidden_pending_count: {obs.hidden_pending_count}

        Visible emails:
        {chr(10).join(emails_lines) if emails_lines else "(none)"}

        Recent history:
        {history_block}

        Decide one action now. Output exactly one JSON object.
        """
    ).strip()
    return prompt


def parse_model_action(text: str, visible_ids: List[str]) -> Optional[MyAction]:
    if not text:
        return None
    match = JSON_ACTION_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None

    email_id = str(obj.get("email_id", "")).strip()
    action_type = str(obj.get("action_type", "")).strip().lower()
    response = str(obj.get("response", "") or "")

    if email_id not in visible_ids:
        return None
    if action_type not in ("reply", "escalate", "archive"):
        return None

    return MyAction(email_id=email_id, action_type=action_type, response=response)


def rule_policy(obs: MyObservation) -> MyAction:
    # Greedy baseline: pick most urgent visible email.
    if not obs.inbox:
        # Shouldn't happen; environment should mark done when no pending.
        return MyAction(email_id="0", action_type="archive", response="")

    ranked = sorted(obs.inbox, key=lambda e: _urgency_sort_key(e, obs.current_time))
    chosen = ranked[0]

    rem = _remaining_sla(chosen, obs.current_time)
    priority = str(chosen.priority)
    tier = str(chosen.customer_tier)
    subj = (chosen.subject or "").lower()
    body = (chosen.body or "").lower()
    text = f"{subj} {body}"

    # Simple decision rules
    if rem <= 2 or tier == "vip" or priority == "high" or any(
        k in text for k in ("outage", "down", "failing", "urgent", "asap", "vip")
    ):
        return MyAction(email_id=chosen.email_id, action_type="escalate", response="")

    if priority == "low" and any(k in text for k in ("no action", "fyi", "notes", "newsletter")):
        return MyAction(email_id=chosen.email_id, action_type="archive", response="")

    # Reply template that tries to satisfy keyword-based rubrics
    response = (
        "Thanks for reaching out — we’re looking into this and will share an ETA shortly. "
        "Could you share any relevant details (order ID / screenshots / steps) so we can help faster?"
    )
    return MyAction(email_id=chosen.email_id, action_type="reply", response=response)


async def llm_policy(client: OpenAI, obs: MyObservation, history: List[str], step_idx: int) -> Optional[MyAction]:
    visible_ids = [e.email_id for e in obs.inbox]
    user_prompt = build_user_prompt(step_idx, obs, history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception:
        return None

    return parse_model_action(response_text, visible_ids=visible_ids)


async def run_task(task: TaskSpec) -> EpisodeSummary:
    history: List[str] = []
    total_reward = 0.0
    sla_breaches = 0
    invalid_actions = 0
    trajectory: List[Tuple[MyObservation, Optional[MyAction]]] = []

    llm_client: Optional[OpenAI] = None
    use_llm = bool(MODEL_NAME and OPENAI_API_KEY)
    if use_llm:
        kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        llm_client = OpenAI(**kwargs)

    async with MyEnv(base_url=BASE_URL) as env:
        # Use the task suite runner for deterministic task scores.
        def policy(obs: MyObservation) -> Optional[MyAction]:
            # If LLM enabled, we drive it through the same strict JSON prompt used previously.
            # The rollout helper expects a sync policy, so we only use rule policy there.
            return rule_policy(obs)

        # If LLM is configured, we run a manual loop so we can call the async LLM.
        if use_llm and llm_client is not None:
            result = await env.reset(config=task.reset_config)
            obs = result.observation
            observation_chain: List[MyObservation] = [obs]
            for step_idx in range(1, min(task.max_steps, MAX_STEPS) + 1):
                if result.done:
                    break
                action = await llm_policy(llm_client, obs, history, step_idx) or rule_policy(obs)
                trajectory.append((obs, action))
                result = await env.step(action)
                obs = result.observation
                observation_chain.append(obs)

                reward = float(result.reward or 0.0)
                total_reward += reward

                grade = (obs.metadata or {}).get("grade") or {}
                if bool(grade.get("sla_breach", False)):
                    sla_breaches += 1
                if (obs.metadata or {}).get("error") == "invalid_email_id_or_not_pending":
                    invalid_actions += 1

                history.append(
                    f"t={obs.current_time} action=({action.email_id},{action.action_type}) "
                    f"reward={reward:+.2f} breach={bool(grade.get('sla_breach', False))} "
                    f"new={len((obs.metadata or {}).get('new_emails') or [])}"
                )

            state = await env.state()
            task_score = float(task.grader(trajectory, state, observation_chain))
        else:
            task_score, traj, state, observation_chain = await rollout_task(env=env, task=task, policy=policy)
            trajectory = traj
            # Accumulate reward + breach stats using post-step observations aligned with each action.
            for step_idx, (_, action) in enumerate(traj):
                if step_idx + 1 >= len(observation_chain):
                    break
                post = observation_chain[step_idx + 1]
                grade = (post.metadata or {}).get("grade") or {}
                if bool(grade.get("sla_breach", False)):
                    sla_breaches += 1
                if (post.metadata or {}).get("error") == "invalid_email_id_or_not_pending":
                    invalid_actions += 1
            # Total reward can be reconstituted only from step results; keep 0 for pure task scoring mode.
            total_reward = float(total_reward)

    return EpisodeSummary(
        task_id=task.task_id,
        difficulty=task.difficulty,
        steps=len(history),
        total_reward=total_reward,
        task_score=float(task_score),
        sla_breaches=sla_breaches,
        invalid_actions=invalid_actions,
    )


async def main() -> None:
    results: List[EpisodeSummary] = []
    for task in all_tasks():
        summary = await run_task(task)
        results.append(summary)
        print(
            f"[{summary.task_id} ({summary.difficulty})] steps={summary.steps} "
            f"task_score={summary.task_score:.3f} total_reward={summary.total_reward:.2f} "
            f"sla_breaches={summary.sla_breaches} invalid_actions={summary.invalid_actions}"
        )

    if results:
        avg_task = sum(r.task_score for r in results) / len(results)
        print(f"AVERAGE_TASK_SCORE: {avg_task:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

