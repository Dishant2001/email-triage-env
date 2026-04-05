"""
Baseline inference runner for the MyEnv OpenEnv environment.

Requirements satisfied:
- Uses OpenAI Python client
- Reads credentials from OPENAI_API_KEY
- Produces a reproducible baseline score on 3 tasks (easy/medium/hard) via fixed seeds
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from envs.my_env.client import MyEnv
from envs.my_env.models import MyAction, MyObservation
from envs.my_env.tasks import all_tasks

from dotenv import load_dotenv
load_dotenv()


ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional

TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "220")
MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")


def _rule_policy(obs: MyObservation) -> Optional[MyAction]:
    # Simple deterministic baseline: pick first visible email and follow a crude heuristic.
    if not obs.inbox:
        return None
    chosen = obs.inbox[0]
    text = f"{chosen.subject} {chosen.body}".lower()
    if str(chosen.customer_tier) == "vip" or str(chosen.priority) == "high" or any(
        k in text for k in ("outage", "down", "failing", "urgent", "asap")
    ):
        return MyAction(email_id=chosen.email_id, action_type="escalate", response="")
    if str(chosen.priority) == "low" and any(k in text for k in ("no action", "fyi", "newsletter", "notes")):
        return MyAction(email_id=chosen.email_id, action_type="archive", response="")
    return MyAction(
        email_id=chosen.email_id,
        action_type="reply",
        response="Thanks — we’re looking into this. Could you share relevant details so we can help quickly?",
    )


def _build_client() -> Optional[OpenAI]:
    if not (OPENAI_API_KEY and MODEL_NAME):
        return None
    kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


async def _llm_action(client: OpenAI, obs: MyObservation) -> Optional[MyAction]:
    # Strict JSON action schema prompt (kept compact for reproducibility).
    visible = [
        {
            "email_id": e.email_id,
            "tier": str(e.customer_tier),
            "priority": str(e.priority),
            "created_time": int(e.created_time),
            "sla_limit": int(e.sla_limit),
            "subject": e.subject,
        }
        for e in obs.inbox
    ]
    if not visible:
        return None

    system = (
        "You are an inbox triage agent. Output exactly one JSON object with keys "
        '{"email_id": string, "action_type": "reply"|"escalate"|"archive", "response": string}. '
        "email_id must be one of the visible emails."
    )
    user = {"current_time": int(obs.current_time), "visible_emails": visible}

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    # Very strict parse: expect a JSON-ish object and fall back to rule policy if anything is off.
    import json
    import re

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    email_id = str(obj.get("email_id", "")).strip()
    action_type = str(obj.get("action_type", "")).strip().lower()
    response = str(obj.get("response", "") or "")
    if email_id not in {e["email_id"] for e in visible}:
        return None
    if action_type not in ("reply", "escalate", "archive"):
        return None
    return MyAction(email_id=email_id, action_type=action_type, response=response)


async def run() -> None:
    llm_client = _build_client()
    use_llm = llm_client is not None

    results: List[Dict[str, float]] = []
    async with MyEnv(base_url=ENV_BASE_URL) as env:
        for task in all_tasks():
            result = await env.reset(config=task.reset_config)
            obs = result.observation
            observation_chain: List[MyObservation] = [obs]
            total_reward = 0.0
            trajectory = []

            for _ in range(min(task.max_steps, MAX_STEPS)):
                if result.done:
                    break
                if use_llm and llm_client is not None:
                    action = await _llm_action(llm_client, obs) or _rule_policy(obs)
                else:
                    action = _rule_policy(obs)
                if action is None:
                    break
                trajectory.append((obs, action))
                result = await env.step(action)
                obs = result.observation
                observation_chain.append(obs)
                total_reward += float(result.reward or 0.0)

            final_state = await env.state()
            task_score = float(task.grader(trajectory, final_state, observation_chain))
            results.append({"task_score": task_score, "total_reward": total_reward})
            print(
                f"[{task.task_id} ({task.difficulty})] task_score={task_score:.3f} total_reward={total_reward:.2f}"
            )

    avg = sum(r["task_score"] for r in results) / max(1, len(results))
    print(f"AVERAGE_TASK_SCORE: {avg:.3f}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

