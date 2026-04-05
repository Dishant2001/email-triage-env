---
title: EmailTriageEnv Server
emoji: 🏀
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# EmailTriageEnv

**Train and evaluate agents on a realistic support-inbox triage loop**—SLAs, partial visibility, action costs, and dynamic arrivals—inside a proper **OpenEnv** server with HTTP, WebSocket, Docker, and optional **Hugging Face Spaces** deployment.

> *One pending email per step: **reply**, **escalate**, or **archive**. The world keeps ticking. You only see the top of the queue.*

---

## In this README

| Section | You’ll find |
|---------|-------------|
| [Why this matters](#why-this-matters) | Who it’s for and what gap it fills |
| [What you get](#what-you-get-out-of-the-box) | Features and design choices at a glance |
| [How it works](#how-it-works) | Lifecycle, observability, scoring |
| [Benchmarks](#benchmarks) | Three tasks, task graders |
| [Quick start & deploy](#quick-start) | Code, Docker, Hugging Face |
| [Reference](#reference-action--observation--reward) | Schemas, modules, logging |

---

## Why this matters

**Real operations don’t look like static classification.** A support or copilot agent must prioritize under deadlines, avoid burning the escalation path, write acceptable customer-facing text, and keep working while **new work arrives**. Most toy benchmarks fix the input set and hide none of the tradeoffs.

**EmailTriageEnv** compresses that story into a **deterministic, reproducible** simulator:

- **Virtual time** and **SLA breaches** punish procrastination and wrong ordering.
- **Top‑N visibility** forces policies to reason under **partial observability** (urgent mail can sit outside the visible slice until it surfaces).
- **Per-action costs and durations** make “always escalate” and “always reply” both suboptimal in different ways.
- **Optional seeded arrivals** turn the inbox into a **non-stationary** stream for harder evaluation.

That combination is **useful for research and product-facing benchmarks**: RL, preference optimization, and **LLM-driven** agents can all target the same environment and **compare on the same three graded tasks** without leaking answer keys into observations.

---

## What you get out of the box

| Capability | Detail |
|------------|--------|
| **OpenEnv-native** | `EmailTriageEnv` client + `EmailTriageEnvironment` server; `reset` / `step` / `state`; `POST` + `WS /ws`. |
| **Fair evaluation** | `PublicEmail` in observations; **`ground_truth_action`** and **keyword rubrics** stay server-side unless you explicitly expose them in state. |
| **Dense, interpretable signal** | Per-step **normalized reward** in **[0, 1]** plus **`metadata["grade"]`** breakdown (SLA, prioritization, action match, reply rubric, costs). |
| **Episode benchmarks** | Three **`TaskSpec`** graders in **`tasks.py`** (easy → medium → hard), each scoring **[0, 1]** independently of the mean step reward. |
| **LLM runner** | Repo **`inference.py`**: structured **`[START]` / `[STEP]` / `[END]`** logs for leaderboard-style runs. |
| **Shipping** | `uv` + lockfile, **Dockerfile**, **`openenv push`** path for HF Spaces, web UI at `/web`. |

**Honest scope:** `thread_id` is carried for future thread logic but **does not** yet drive dynamics or grading. The keyword rubric is **deterministic substring coverage** (plus a length fallback)—good for stable benchmarks, not a substitute for human judgment.

---

## How it works

### Architecture

```mermaid
flowchart LR
    subgraph client
        A[Agent / script]
        B[EmailTriageEnv client]
    end
    subgraph server
        C[OpenEnv HTTP / WebSocket]
        D[EmailTriageEnvironment]
        E[scenarios / dynamics / grader]
    end
    A --> B
    B -->|reset, step, state| C
    C --> D
    D --> E
```

| Layer | Role |
|--------|------|
| **`EmailTriageEnv`** (`client.py`) | Serializes `MyAction`, parses `StepResult[MyObservation]`, one WebSocket session per server env instance. |
| **`server/app.py`** | `create_app(EmailTriageEnvironment, …)` → `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `WS /ws`. |
| **`EmailTriageEnvironment`** | RNG, full inbox, virtual time, config; `reset` / `step`. |
| **`scenarios.py`** | `starter_inbox()`, `arrival_templates()`. |
| **`dynamics.py`** | Urgency ranking, top‑N slice, time advance, optional arrivals. |
| **`grader.py`** | `grade_step` → breakdown + normalized **`reward`**. |
| **`tasks.py`** | Episode-level **`TaskSpec.grader`** (harness), separate from per-step reward. |

State changes only through **`reset`** and **`step`**. **`state()`** is for inspection; by default grader labels are stripped so RL runs don’t see the answer key—set **`expose_grader_labels_in_state: true`** in **`EnvConfig`** when you need full rows.

### Episode lifecycle

**Reset:** Load **`EnvConfig`** (e.g. `top_n`, `seed`, arrivals budget), optionally seed RNG, deep-copy **`starter_inbox()`**, set **`current_time = 0`**, return top‑N pending by urgency plus **`hidden_pending_count`**.

**Valid step:** Grade at **current** time (before advance), mark email processed, advance clock by **action duration**, maybe inject **one** templated arrival, return new top‑N and **`metadata.grade`** / **`metadata.new_emails`**.

**Invalid step** (bad or stale `email_id`): Time still moves; strong negative reward; **`metadata.error`** = `invalid_email_id_or_not_pending`.

**SLA** is evaluated at **step start**; new mail gets **`created_time`** equal to the clock **after** the step’s time advance.

**Arrivals:** Seeded draws; probability a bit higher early in the episode; at most one new email per valid step when enabled.

### Observability and urgency

Only **pending** mail appears in **`inbox`**. Urgency blends **priority**, **customer tier**, and **remaining SLA**. The agent sees the **top `top_n`** by that score—not FIFO—so hidden mail can breach SLA if the policy ignores the global picture.

### Per-step reward (summary)

Multi-factor **deterministic** score: SLA, “did you pick the most urgent pending mail?”, **ground-truth action** match, **reply** keyword (or length) rubric, plus **action** and **idle** costs. Scalar **`reward`** is **normalized to [0, 1]**; **`metadata["grade"]`** keeps the breakdown for analysis and task graders.

| Component | Role |
|-----------|------|
| **SLA** | Breach vs on-time at step start |
| **Prioritization** | Match to best pending by urgency |
| **Action** | `reply` / `escalate` / `archive` vs label |
| **Response** | Keywords in reply text (or length proxy) |
| **Costs** | Discourage spam escalate and useless steps |

### Task score vs mean step reward

**Per-step reward** shapes learning signal. **Task score** (in **`tasks.py`**) scores the **whole episode**—e.g. first-step urgency on easy, SLA + escalation discipline on medium, **responsiveness to new arrivals** on hard. They **need not** agree; a run can have high mean **`reward`** but modest **`task_score`**.

---

## Benchmarks

Three tasks, **[0, 1]** scores, **deterministic** under fixed seeds:

| Task ID | Difficulty | Max steps | What it tests |
|---------|------------|-----------|----------------|
| `easy_single_urgent_first` | Easy | 1 | Pick the **most urgent visible** mail and the **right** action first try. |
| `medium_sla_safe_throughput` | Medium | 30 | Clear the queue with **few SLA breaches**, **correct actions**, **limited escalation**. |
| `hard_dynamic_arrivals_backlog` | Hard | 30 | Same pressures **plus** **new mail**—latency to first handle of arrivals matters. |


Repo-root **`inference.py`** drives an **LLM** against a live server and prints **`[START]` / `[STEP]` / `[END]`** lines. **`success=true`** only when the **task** grader returns a perfect **1.0**—strong step rewards alone are not enough.

---

## Quick start

**Python 3.10+**, from this package root (e.g. `envs/email_triage_env`):

```bash
uv sync
uv run server
# or: uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

Default URL: **`http://localhost:8000`** (`openenv.yaml`).

The client is **async**:

```python
import asyncio
from email_triage_env import EmailTriageEnv, MyAction


async def main():
    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(
            config={"top_n": 3, "seed": 1, "arrivals_enabled": True, "max_new_emails": 2}
        )
        print(f"current_time={result.observation.current_time} visible={len(result.observation.inbox)}")

        first = result.observation.inbox[0]
        result = await env.step(
            MyAction(
                email_id=first.email_id,
                action_type="reply",
                response="Thanks — we’ll investigate and share an ETA.",
            )
        )
        print(f"reward={result.reward} done={result.done}")


asyncio.run(main())
```

Use **`EmailTriageEnv.from_docker_image(...)`** (async) to start a container, connect, and **`close()`** when done.

### Docker

From **this directory** (`pyproject.toml` + `server/Dockerfile`):

```bash
docker build -t email-triage-env:latest -f server/Dockerfile .
```

From **monorepo root** (context = whole repo):

```bash
docker build -f envs/email_triage_env/server/Dockerfile --build-arg SOURCE_DIR=envs/email_triage_env -t email-triage-env:latest .
```

Use a repo-root **`.dockerignore`** that excludes **`**/.venv/`** so host virtualenvs are not copied into the image.

### Hugging Face Spaces

```bash
openenv push
# openenv push --repo-id username/space-name --private
```

Requires HF auth. After deploy: Space URL, **`/web`**, **`/docs`**, **`/health`**, **`/ws`**.

**Options:** `--directory`, `-d` · `--repo-id`, `-r` · `--base-image`, `-b` · `--private`

---

## Reference: action, observation, reward

### `MyAction` (`models.py`)

| Field | Type | Notes |
|--------|------|--------|
| `email_id` | `str` | Must be **pending**; invalid IDs waste time and hurt reward. |
| `action_type` | `reply` \| `escalate` \| `archive` | |
| `response` | `str` | Scored on **keywords** (server-side) when replying. |

### `MyObservation`

| Field | Notes |
|--------|--------|
| `current_time` | After a step, reflects time **after** advance. |
| `inbox` | Up to **`top_n`** **`PublicEmail`**, urgency-sorted; **no** grader labels. |
| `hidden_pending_count` | Pending mail not shown. |
| `last_email_id`, `last_action_type`, `sla_breach`, `time_advance`, `action_cost` | Describe the **last** step. |
| `done` | No pending mail left. |
| `reward` | Normalized **[0, 1]**. |
| `metadata` | **`grade`**, **`new_emails`**, or **`error`**. |

**`EnvConfig`** (at reset): `top_n`, `seed`, `arrivals_enabled`, `max_new_emails`, `action_costs`, `per_step_idle_cost`, `action_durations`, `expose_grader_labels_in_state`.

**`PublicEmail` vs `Email`:** Observations use **`PublicEmail`** (+ **`thread_id`** for grouping, not scored). **`Email`** adds **`ground_truth_action`** and **`required_response_keywords`**. **`GET /state`** strips labels unless **`expose_grader_labels_in_state: true`**.

**`MyState`:** Full clock, all emails, config, arrival counter.

**`training_utils.py`:** **`slot_action_to_my_action`**, **`flat_index_to_slots`**, **`ACTION_KINDS`** for discrete RL spaces.

### Server modules

| File | Role |
|------|------|
| `email_triage_environment.py` | Orchestrates reset / step / state. |
| `scenarios.py` | Starter + arrival templates. |
| `dynamics.py` | Urgency, top‑N, time, arrivals. |
| `grader.py` | **`grade_step`** → **`GradeBreakdown`**. |

### Structured stdout (`inference.py`)

Repo-root **`inference.py`** runs an **LLM policy** only: each step the model sees the visible inbox and must return parseable JSON for **`MyAction`**. Configure **`OPENAI_API_KEY`** or **`HF_TOKEN`**, **`MODEL_NAME`** / **`OPENAI_MODEL`**, **`OPENAI_BASE_URL`** / **`ENV_BASE_URL`** as needed.

1. **`[START]`** — `task`, `env`, `model`  
2. **`[STEP]`** — `step`, `action`, `reward`, `done`, `error`  
3. **`[END]`** — `success` (perfect task score only), `steps`, `score`, `rewards` CSV  

Example:

```text
[START] task=easy_single_urgent_first env=email_triage_env model=openai/gpt-oss-120b:groq
[STEP]  step=1 action=reply('1','…') reward=0.61 done=false error=null
[END]   success=false steps=1 score=0.50 rewards=0.61
```

### Extending

Ideas: thread **follow-ups** after reply; **open/read** actions; **assignee** on escalate; richer reply checklists.

### OpenEnv rubric hook

If a **`rubric`** is installed on the environment, it can override **`observation.reward`** after **`grade_step`**. Default: built-in grader only.

### Advanced: existing server & concurrency

```python
from email_triage_env import EmailTriageEnv, MyAction

async def run():
    async with EmailTriageEnv(base_url="<ENV_HTTP_URL_HERE>") as client:
        result = await client.reset()
        result = await client.step(MyAction(email_id="1", action_type="reply", response="Hello!"))
```

For multiple concurrent WebSocket sessions, use **`EmailTriageEnvironment`** as a **class** in **`create_app`** and raise **`max_concurrent_envs`** in **`server/app.py`**.

### Development

```bash
python -c "from server.email_triage_environment import EmailTriageEnvironment; EmailTriageEnvironment()"
openenv validate
```

### Monorepo imports

```python
from envs.email_triage_env.client import EmailTriageEnv
from envs.email_triage_env.models import MyAction
```

---

## Project structure

```
email_triage_env/
├── .dockerignore
├── __init__.py
├── README.md
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── training_utils.py
├── client.py
├── models.py
├── documentation.md      # stub → see README
├── explanation.md        # stub → see README
└── server/
    ├── app.py
    ├── email_triage_environment.py
    ├── scenarios.py
    ├── dynamics.py
    ├── grader.py
    └── Dockerfile
```

---

*EmailTriageEnv: a compact, reproducible inbox for agents that need to do more than classify a fixed list—prioritize, commit, and keep up when the queue moves.*
