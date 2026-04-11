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
    from email_triage_env.tasks import TaskSpec, all_tasks, harness_task_score
except ImportError:
    from client import EmailTriageEnv
    from models import MyAction, MyObservation
    from tasks import TaskSpec, all_tasks, harness_task_score

# OpenAI-compatible chat completions (default: Hugging Face Inference / router).
DEFAULT_LLM_BASE_URL = "https://router.huggingface.co/v1"

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MODEL_NAME = (
    os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-120b:groq"
)
HF_TOKEN = os.getenv("HF_TOKEN") or "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"

API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
API_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL") or DEFAULT_LLM_BASE_URL
BENCHMARK_ENV = os.getenv("BENCHMARK_ENV") or "email_triage_env"

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") or "openenv-email_triage:latest"

TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
# JSON + reply body can exceed small limits; truncation yields unclosed strings and parse failure.
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "1024")
MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")
# Truncate email body in the LLM context to limit tokens (full body is still on the server).
BODY_MAX_CHARS = int(os.getenv("INFERENCE_BODY_MAX_CHARS") or "800")
VERBOSE_STEPS = os.getenv("INFERENCE_VERBOSE_STEPS", "").lower() in ("1", "true", "yes")


def _model_display() -> str:
    return str(MODEL_NAME or "unknown").replace(" ", "_")


def _fmt_reward(x: float) -> str:
    return f"{float(x):.2f}"


def _fmt_task_score(x: float) -> str:
    """Task scores are strictly inside (0, 1); avoid two-decimal rounding to 0.00 / 1.00."""
    return f"{float(x):.6f}"


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


def _build_client() -> Optional[OpenAI]:
    if not (API_KEY and MODEL_NAME):
        return None
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _merge_reset_config(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task reset_config first; optional env overrides for ad-hoc runs (training / ablations).

    Supported env vars: SCENARIO_PROFILE, SEED, ENTANGLEMENT_ENABLED,
    THREAD_FOLLOWUPS_ENABLED, ESCALATION_ECHO_ENABLED, ARRIVALS_ENABLED, MAX_NEW_EMAILS, TOP_N.
    """
    out: Dict[str, Any] = dict(base)
    sp = os.getenv("SCENARIO_PROFILE")
    if sp is not None and str(sp).strip() != "":
        try:
            out["scenario_profile"] = int(sp)
        except ValueError:
            print(f"inference: ignoring invalid SCENARIO_PROFILE={sp!r}", file=sys.stderr)
    seed_override = os.getenv("SEED")
    if seed_override is not None and str(seed_override).strip() != "":
        try:
            out["seed"] = int(seed_override)
        except ValueError:
            print(f"inference: ignoring invalid SEED={seed_override!r}", file=sys.stderr)
    for key, env_name in (
        ("entanglement_enabled", "ENTANGLEMENT_ENABLED"),
        ("thread_followups_enabled", "THREAD_FOLLOWUPS_ENABLED"),
        ("escalation_echo_enabled", "ESCALATION_ECHO_ENABLED"),
        ("arrivals_enabled", "ARRIVALS_ENABLED"),
    ):
        v = _env_bool(env_name)
        if v is not None:
            out[key] = v
    mne = os.getenv("MAX_NEW_EMAILS")
    if mne is not None and str(mne).strip() != "":
        try:
            out["max_new_emails"] = int(mne)
        except ValueError:
            pass
    tn = os.getenv("TOP_N")
    if tn is not None and str(tn).strip() != "":
        try:
            out["top_n"] = int(tn)
        except ValueError:
            pass
    return out


def _truncate_body(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)] + "..."


def _llm_system_prompt() -> str:
    """System prompt aligned with emergent step rewards (no oracle / keyword gaming)."""
    return (
        "You are an inbox triage agent. Output exactly one compact JSON object with keys "
        '{"email_id": string, "action_type": "reply"|"escalate"|"archive", "response": string}. '
        "email_id must be one of the visible emails (use the exact string ids from the list). "
        "For action_type escalate or archive, set response to \"\" (empty string). "
        "Priority urgency order is: critical > high > medium > low. "
        "thread_reply_excerpt (when non-empty) is your prior reply in that thread—stay consistent. "
        "For reply, write at least one clear sentence (>20 characters) and include "
        "punctuation such as . , ! or ? (plain professional tone; do not try to guess hidden rubric keywords). "
        "Keep the response field short enough that the JSON stays complete."
    )


def _step_verbose_suffix(obs: MyObservation) -> str:
    if not VERBOSE_STEPS:
        return ""
    md = obs.metadata or {}
    grade = md.get("grade") or {}
    parts: List[str] = []
    if "consequence_signal" in grade:
        try:
            parts.append(f"cs={float(grade.get('consequence_signal', 0.0)):.2f}")
        except (TypeError, ValueError):
            pass
    stats = md.get("episode_stats")
    if isinstance(stats, dict):
        bc = stats.get("sla_breach_count")
        po = stats.get("sla_pressure_offset")
        if bc is not None:
            parts.append(f"breaches={bc}")
        if po is not None:
            parts.append(f"pressure={po}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def _visible_id_set(visible: List[Dict[str, Any]]) -> set[str]:
    return {str(e["email_id"]).strip() for e in visible}


async def _llm_action(client: OpenAI, obs: MyObservation) -> Optional[MyAction]:
    visible = [
        {
            "email_id": e.email_id,
            "thread_id": e.thread_id or "",
            "tier": str(e.customer_tier),
            "priority": str(e.priority),
            "created_time": int(e.created_time),
            "sla_limit": int(e.sla_limit),
            "subject": e.subject,
            "body": _truncate_body(e.body, BODY_MAX_CHARS),
            "thread_reply_excerpt": (e.thread_reply_excerpt or "").strip(),
        }
        for e in obs.inbox
    ]
    if not visible:
        return None

    system = _llm_system_prompt()
    user = {
        "current_time": int(obs.current_time),
        "hidden_pending_count": int(obs.hidden_pending_count),
        "visible_emails": visible,
    }
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
        reset_cfg = _merge_reset_config(task.reset_config)

        result = await env.reset(config=reset_cfg)
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
                f"error={_fmt_error(obs)}{_step_verbose_suffix(obs)}",
                flush=True,
            )

        final_state = await env.state()
        raw = float(task.grader(trajectory, final_state, observation_chain))
        score = harness_task_score(raw)
        success = 0.0 < score < 1.0
    except Exception as ex:
        print(f"inference: task={task.task_id} {type(ex).__name__}: {ex}", file=sys.stderr)
        score = harness_task_score(0.0)
        success = False

    return rewards, step_num, score, success


def _print_end(rewards: List[float], step_num: int, score: float, success: bool) -> None:
    rewards_csv = ",".join(_fmt_reward(x) for x in rewards)
    print(
        f"[END]   success={_fmt_bool(success)} steps={step_num} "
        f"score={_fmt_task_score(score)} rewards={rewards_csv}",
        flush=True,
    )


async def run() -> None:
    llm_client = _build_client()
    if llm_client is None:
        print(
            "inference.py requires OPENAI_API_KEY or HF_TOKEN, and MODEL_NAME or OPENAI_MODEL. "
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
            except Exception as http_ex:
                print(
                    f"inference: HTTP env at {ENV_BASE_URL!r} failed ({type(http_ex).__name__}: {http_ex}); "
                    f"trying Docker image {IMAGE_NAME!r}",
                    file=sys.stderr,
                )
                docker_env: Optional[EmailTriageEnv] = None
                try:
                    docker_env = await EmailTriageEnv.from_docker_image(IMAGE_NAME)
                    rewards, step_num, score, success = await _run_one_task(
                        docker_env, task, llm_client
                    )
                finally:
                    if docker_env is not None:
                        close_fn = getattr(docker_env, "close", None)
                        if callable(close_fn):
                            maybe = close_fn()
                            if asyncio.iscoroutine(maybe):
                                await maybe
        except Exception as ex:
            print(f"inference: env connection/session: {type(ex).__name__}: {ex}", file=sys.stderr)
            rewards, step_num, score, success = [], 0, harness_task_score(0.0), False
        finally:
            _print_end(rewards, step_num, score, success)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
