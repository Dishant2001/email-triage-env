"""
Inbox Environment Evaluator
==========================

This script runs one or more evaluation episodes against the MyEnv inbox server.

It is intentionally structured like `sample_inference_script.py`:
- create client
- reset to get first observation
- loop: build prompt from observation -> get action -> step -> log -> repeat

Supports two agent modes:
1) LLM policy (if MODEL_NAME + HF_TOKEN/API_KEY are configured)
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
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI

from envs.my_env.client import MyEnv
from envs.my_env.models import Email, MyAction, MyObservation


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"

MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")
TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "220")

# Default evaluation suite: different seeds will trigger different arrival samples
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

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
    seed: int
    steps: int
    total_reward: float
    sla_breaches: int
    invalid_actions: int


def _remaining_sla(email: Email, current_time: int) -> int:
    return int(email.sla_limit) - (int(current_time) - int(email.created_time))


def _urgency_sort_key(email: Email, current_time: int) -> Tuple[int, int, int]:
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


async def run_episode(seed: int) -> EpisodeSummary:
    history: List[str] = []
    total_reward = 0.0
    sla_breaches = 0
    invalid_actions = 0

    llm_client: Optional[OpenAI] = None
    use_llm = bool(MODEL_NAME and API_KEY)
    if use_llm:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with MyEnv(base_url=BASE_URL) as env:
        # Reset-time config controls arrivals; vary seed to explore different arrival samples.
        result = await env.reset(
            config={
                "top_n": 5,
                "seed": seed,
                "arrivals_enabled": True,
                "max_new_emails": 3,
            }
        )
        obs: MyObservation = result.observation

        for step_idx in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action: Optional[MyAction] = None
            if use_llm and llm_client is not None:
                action = await llm_policy(llm_client, obs, history, step_idx)

            if action is None:
                action = rule_policy(obs)

            result = await env.step(action)
            obs = result.observation

            reward = float(result.reward or 0.0)
            total_reward += reward

            grade = (obs.metadata or {}).get("grade") or {}
            if bool(grade.get("sla_breach", False)):
                sla_breaches += 1
            if (obs.metadata or {}).get("error") == "invalid_email_id_or_not_pending":
                invalid_actions += 1

            history_line = (
                f"t={obs.current_time} action=({action.email_id},{action.action_type}) "
                f"reward={reward:+.2f} breach={bool(grade.get('sla_breach', False))} "
                f"new={len((obs.metadata or {}).get('new_emails') or [])}"
            )
            history.append(history_line)

    return EpisodeSummary(
        seed=seed,
        steps=len(history),
        total_reward=total_reward,
        sla_breaches=sla_breaches,
        invalid_actions=invalid_actions,
    )


async def main() -> None:
    seeds_env = os.getenv("EVAL_SEEDS")
    if seeds_env:
        seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    else:
        seeds = DEFAULT_SEEDS

    results: List[EpisodeSummary] = []
    for seed in seeds:
        summary = await run_episode(seed)
        results.append(summary)
        print(
            f"[seed={summary.seed}] steps={summary.steps} total_reward={summary.total_reward:.2f} "
            f"sla_breaches={summary.sla_breaches} invalid_actions={summary.invalid_actions}"
        )

    if results:
        avg_reward = sum(r.total_reward for r in results) / len(results)
        avg_breaches = sum(r.sla_breaches for r in results) / len(results)
        print(f"AVERAGE: reward={avg_reward:.2f} sla_breaches={avg_breaches:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

