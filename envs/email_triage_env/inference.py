from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from email_triage_env.client import EmailTriageEnv
    from email_triage_env.models import MyAction, MyObservation
    from email_triage_env.tasks import TaskSpec, all_tasks
except ImportError:
    from client import EmailTriageEnv
    from models import MyAction, MyObservation
    from tasks import TaskSpec, all_tasks

# OpenAI-compatible chat completions (default: Hugging Face Inference / router).
DEFAULT_LLM_BASE_URL = "https://router.huggingface.co/v1"

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b:groq"
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN
API_BASE_URL = (
    os.getenv("API_BASE_URL") or DEFAULT_LLM_BASE_URL
)
BENCHMARK_ENV = os.getenv("BENCHMARK_ENV") or "email_triage_env"

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") or "openenv-email_triage:latest"

TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
# JSON + reply body can exceed small limits; truncation yields unclosed strings and parse failure.
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "1024")
MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")


def _model_display() -> str:
    return str(MODEL_NAME or "unknown").replace(" ", "_")


def _fmt_reward(x: float) -> str:
    return f"{float(x):.2f}"


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def _fmt_error(obs: MyObservation) -> str:
    md = obs.metadata or {}
    err = md.get("error")
    if err is None:
        return "null"
    s = str(err).replace("\n", " ").replace("\r", " ")
    if '"' in s:
        s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _action_str(action: MyAction) -> str:
    eid = str(action.email_id).replace("\\", "\\\\").replace("'", "\\'")
    at = action.action_type
    if at == "reply":
        r = (action.response or "").replace("\\", "\\\\").replace("'", "\\'")
        if len(r) > 80:
            r = r[:77] + "..."
        return f"{at}('{eid}','{r}')"
    return f"{at}('{eid}')"


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _build_client() -> Optional[OpenAI]:
    if not (API_KEY and MODEL_NAME):
        return None
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def _visible_id_set(visible: List[Dict[str, Any]]) -> set[str]:
    return {str(e["email_id"]).strip() for e in visible}


async def _llm_action(client: OpenAI, obs: MyObservation) -> Optional[MyAction]:
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
        "You are an inbox triage agent. Output exactly one compact JSON object with keys "
        '{"email_id": string, "action_type": "reply"|"escalate"|"archive", "response": string}. '
        "email_id must be one of the visible emails (use the exact string ids from the list). "
        "For action_type escalate or archive, set response to \"\" (empty string). "
        "For reply only, keep response to one or two short sentences so the JSON stays complete."
    )
    user = {"current_time": int(obs.current_time), "visible_emails": visible}
    allowed_ids = _visible_id_set(visible)

    try:
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
    except Exception as ex:
        print(f"inference: LLM request failed: {type(ex).__name__}: {ex}", file=sys.stderr)
        return None

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        hint = ""
        if text.lstrip().startswith("{") and text.rstrip().endswith("}") is False:
            hint = " (output looks truncated — raise MAX_TOKENS or shorten the model's response field)"
        print(
            f"inference: model returned no parseable JSON object{hint} (first 500 chars): {text[:500]!r}",
            file=sys.stderr,
        )
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception as ex:
        print(f"inference: JSON parse error: {type(ex).__name__}: {ex}", file=sys.stderr)
        return None
    email_id = str(obj.get("email_id", "")).strip()
    action_type = str(obj.get("action_type", "")).strip().lower()
    response = str(obj.get("response", "") or "")
    if email_id not in allowed_ids:
        print(
            f"inference: model email_id {email_id!r} not in visible ids {sorted(allowed_ids)}",
            file=sys.stderr,
        )
        return None
    if action_type not in ("reply", "escalate", "archive"):
        print(f"inference: invalid action_type {action_type!r}", file=sys.stderr)
        return None
    return MyAction(email_id=email_id, action_type=action_type, response=response)


async def _run_one_task(
    env: EmailTriageEnv,
    task: TaskSpec,
    llm_client: OpenAI,
) -> tuple[List[float], int, float, bool]:
    """Run one episode; caller prints [END] after env.close() (context exit)."""
    rewards: List[float] = []
    step_num = 0
    score = 0.0
    success = False
    trajectory: List[tuple] = []
    observation_chain: List[MyObservation] = []

    print(
        f"[START] task={task.task_id} env={BENCHMARK_ENV} model={_model_display()}",
        flush=True,
    )

    try:
        result = await env.reset(config=task.reset_config)
        obs = result.observation
        observation_chain = [obs]

        for _ in range(min(task.max_steps, MAX_STEPS)):
            if result.done:
                break
            action = await _llm_action(llm_client, obs)
            if action is None:
                break

            trajectory.append((obs, action))
            result = await env.step(action)
            obs = result.observation
            observation_chain.append(obs)
            step_num += 1

            r = float(result.reward or 0.0)
            rewards.append(r)
            print(
                f"[STEP]  step={step_num} action={_action_str(action)} "
                f"reward={_fmt_reward(r)} done={_fmt_bool(bool(result.done))} "
                f"error={_fmt_error(obs)}",
                flush=True,
            )

        final_state = await env.state()
        score = _clip01(float(task.grader(trajectory, final_state, observation_chain)))
        success = score >= 1.0 - 1e-9
    except Exception as ex:
        print(f"inference: task={task.task_id} {type(ex).__name__}: {ex}", file=sys.stderr)
        score = 0.0
        success = False

    return rewards, step_num, score, success


def _print_end(rewards: List[float], step_num: int, score: float, success: bool) -> None:
    rewards_csv = ",".join(_fmt_reward(x) for x in rewards)
    print(
        f"[END]   success={_fmt_bool(success)} steps={step_num} "
        f"score={_fmt_reward(score)} rewards={rewards_csv}",
        flush=True,
    )


async def run() -> None:
    llm_client = _build_client()
    if llm_client is None:
        print(
            "inference.py requires HF_TOKEN or OPENAI_API_KEY, and MODEL_NAME or OPENAI_MODEL. "
            f"Default LLM base URL is {DEFAULT_LLM_BASE_URL}; set OPENAI_BASE_URL or API_BASE_URL to override.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for task in all_tasks():
        rewards: List[float] = []
        step_num = 0
        score = 0.0
        success = False
        try:
            try:
                async with EmailTriageEnv(base_url=ENV_BASE_URL) as env:
                    rewards, step_num, score, success = await _run_one_task(env, task, llm_client)
            except:
                env = await EmailTriageEnv.from_docker_image(IMAGE_NAME)
                rewards, step_num, score, success = await _run_one_task(env, task, llm_client)
        except Exception as ex:
            print(f"inference: env connection/session: {type(ex).__name__}: {ex}", file=sys.stderr)
            rewards, step_num, score, success = [], 0, 0.0, False
        finally:
            _print_end(rewards, step_num, score, success)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
