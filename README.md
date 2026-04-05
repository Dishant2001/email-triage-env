---
title: EmailTriageEnv Server
emoji: 🏀
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# EmailTriageEnv

**EmailTriageEnv** is the HTTP/WebSocket **client**; the server implements **EmailTriageEnvironment** (OpenEnv `Environment`). Each step, an agent selects one **pending** email and applies **`reply`**, **`escalate`**, or **`archive`**.

The environment uses a **virtual clock**, **SLA deadlines**, **partial observability** (only the top‑N pending emails are visible), and optional **seeded new arrivals** during an episode.

## Description and motivation

This is a compact **support-inbox triage** simulator: agents must prioritize under time pressure, avoid SLA breaches, balance **escalation cost** against urgency, and (when replying) satisfy a deterministic **keyword rubric**—without ever seeing grader-only labels in observations.

**Why it exists:** It supports research and evaluation on prioritization, long-horizon policies, non-stationary inboxes (arrivals), and constrained visibility, while keeping scoring deterministic and reproducible via seeds and explicit task definitions in `tasks.py`.

## Setup

- **Python** 3.10+ (see `pyproject.toml`).
- From **this directory** (the environment package root, e.g. `envs/email_triage_env` in the repo):

  ```bash
  uv sync
  ```

- **Run the server** (after sync, `.venv` on your `PATH` or `uv run`):

  ```bash
  uv run server
  # or: uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000
  ```

Default HTTP URL: `http://localhost:8000` (see `openenv.yaml`).

## Quick Start

The simplest way to connect is the **`EmailTriageEnv`** client class:

```python
from email_triage_env import EmailTriageEnv, MyAction

try:
    # Create environment from Docker image
    env = EmailTriageEnv.from_docker_image("email-triage-env:latest")

    # Reset
    result = env.reset(config={"top_n": 3, "seed": 1, "arrivals_enabled": True, "max_new_emails": 2})
    print(f"current_time={result.observation.current_time} visible={len(result.observation.inbox)}")

    # Take one action
    first = result.observation.inbox[0]
    result = env.step(MyAction(email_id=first.email_id, action_type="reply", response="Thanks — we’ll investigate and share an ETA."))
    print(f"reward={result.reward} done={result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `EmailTriageEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From this directory (where pyproject.toml and server/Dockerfile live)
docker build -t email-triage-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Action space

Actions are **`MyAction`** (`models.py`).

| Field | Type | Description |
|--------|------|-------------|
| `email_id` | `str` | Target mail; must be **pending**. For valid grading on a normal step it should be one of the IDs in the current `inbox` (invalid or stale IDs still consume time and yield a low reward). |
| `action_type` | `reply` \| `escalate` \| `archive` | Triage decision for that email. |
| `response` | `str` | Free text; used when `action_type == "reply"` for keyword / length-based scoring (`required_response_keywords` on the server row). |

## Observation space

### `MyObservation`

| Field | Description |
|--------|-------------|
| `current_time` | Virtual time (integer). |
| `inbox` | List of **`PublicEmail`** rows (up to **`top_n`** pending emails, urgency-ranked). No grader-only fields. |
| `hidden_pending_count` | Number of pending emails **not** shown (partial observability). |
| `last_email_id`, `last_action_type` | Last step’s target and action (if any). |
| `sla_breach` | Whether the **last** handled email was acted on after its SLA deadline. |
| `time_advance` | Virtual time consumed by the last action. |
| `action_cost` | Configured monetary-style cost for the last action type. |
| `done` | `True` when there are no pending emails. |
| `reward` | Scalar step reward in **[0, 1]** (normalized shaping signal from the server). |
| `metadata` | After a valid step: `grade` (component breakdown), `new_emails` (ids arriving that step). On invalid selection: `error` (e.g. `invalid_email_id_or_not_pending`). |

Reset-time knobs are passed via `reset(config={...})` using **`EnvConfig`**: e.g. `top_n`, `seed`, `arrivals_enabled`, `max_new_emails`, `action_costs`, `per_step_idle_cost`, `action_durations` (`models.py`).

### `PublicEmail` (each visible inbox row)

| Field | Description |
|--------|-------------|
| `email_id`, `thread_id` | Identity / thread grouping. |
| `subject`, `body` | Content visible to the agent. |
| `priority` | `low` \| `medium` \| `high`. |
| `customer_tier` | `standard` \| `premium` \| `vip`. |
| `created_time`, `sla_limit` | For SLA reasoning. |
| `status` | `pending` \| `processed` (inbox is normally pending-only). |

**Not in observations:** `ground_truth_action`, `required_response_keywords` (server-only for scoring).

## Step reward and `metadata["grade"]`

Each step, the server computes a **multi-factor** score (`server/grader.py`): SLA, prioritization vs most urgent pending mail, match to `ground_truth_action`, reply rubric, and action/idle costs. The scalar **`reward`** returned to the client is **normalized to [0, 1]** for a stable learning signal. The raw-style breakdown remains in **`metadata["grade"]`** for analysis and for episode task graders in `tasks.py`.

## Tasks (benchmark suite)

Episode-level scores come from **`TaskSpec.grader`** in `tasks.py`. Each **`task_score`** is in **[0, 1]** and need not equal the mean of step rewards.

| Task ID | Difficulty | Max steps | Reset (summary) | What the grader rewards |
|---------|------------|-----------|-----------------|-------------------------|
| `easy_single_urgent_first` | **Easy** | 1 | `top_n=5`, `seed=0`, no arrivals | First step: pick **most urgent visible** email + **correct** action (from step metadata). |
| `medium_sla_safe_throughput` | **Medium** | 30 | `top_n=3`, `seed=1`, no arrivals | Low **SLA breach** rate, **correct actions**, **limited escalation** (penalizes heavy escalate use). |
| `hard_dynamic_arrivals_backlog` | **Hard** | 30 | `top_n=3`, `seed=2`, arrivals on, `max_new_emails=3` | **SLA**, **short delay** from first sighting to handling **new** mail, action/reply **quality**, few **invalid** steps. |

**Expected difficulty for agents:** easy → medium → hard (single decision vs long horizon vs dynamic inbox).

## Baseline scores

Two metrics: repo-root **`inference.py`** runs the **LLM** policy against a live server (structured `[START]`/`[STEP]`/`[END]` logs). **`evaluator.py`** can still run a **rule** policy when no LLM is configured.

| Metric | Range | Meaning |
|--------|--------|---------|
| `task_score` | [0, 1] | Task-specific grader for that benchmark. |
| `mean_step_reward` | [0, 1] | Mean of per-step **`reward`** over the episode. |

**Rule-policy baseline** (deterministic heuristics aligned with **`evaluator.rule_policy`**), measured **in-process** on `EmailTriageEnvironment` with the shipped scenarios/graders:

| Task | `task_score` | `mean_step_reward` |
|------|----------------|---------------------|
| `easy_single_urgent_first` | 0.500 | 0.837 |
| `medium_sla_safe_throughput` | 0.858 | 0.856 |
| `hard_dynamic_arrivals_backlog` | 0.930 | 0.859 |
| **Average `task_score`** | **0.763** | — |

Figures drift if scenarios, grader weights, or normalization change; recompute in-process with the rule policy, or run **`evaluator.py`** / **`inference.py`** (LLM + server) for live numbers.

## Structured stdout (`inference.py`)

Repo-root **`inference.py`** runs each **`tasks.all_tasks()`** episode against a live server using an **OpenAI** model (`OPENAI_API_KEY` plus `MODEL_NAME` or `OPENAI_MODEL`). It prints **only** these line types to **stdout**, in order, **per task**:

1. **`[START]`** — Once at the beginning of the episode.  
   `task=<task_id> env=<benchmark> model=<model>`  
   - `env` defaults to `email_triage_env`; override with **`BENCHMARK_ENV`**.  
   - `model` uses the configured model id (spaces replaced with `_`).

2. **`[STEP]`** — Once after **each** successful `env.step()`.  
   `step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`  
   - **`reward`**: per-step shaped signal from the server, **two decimal places**, in **[0, 1]**.  
   - **`done`**: lowercase `true` / `false` — inbox has no pending mail after this step.  
   - **`error`**: `null`, or a quoted string from `observation.metadata["error"]` (e.g. invalid email / not pending).  
   - **`action_str`**: compact encoding, e.g. `escalate('4')`, `archive('2')`, `reply('3','…')` (long replies truncated in the log).

3. **`[END]`** — Always emitted **after** the client session for that task closes (including on failure), so every episode ends with exactly one **`[END]`**.  
   `success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>`  
   - **`steps`**: count of **`[STEP]`** lines for that episode.  
   - **`score`**: episode **`task_score`** from **`TaskSpec.grader`** in **`tasks.py`**, **two decimal places**, clipped to **[0, 1]**. This is **not** the same as averaging step rewards.  
   - **`rewards`**: comma-separated step rewards, each **two decimal places** (empty if no steps).  
   - **`success`**: `true` only if **`score`** is effectively **1.0** (perfect on the task grader); otherwise `false` even when step rewards look good.

**How to read a run:** Strong **`reward`** values mean the **environment grader** liked individual steps; a low **`score`** still means the **benchmark grader** (SLA breaches, escalation count, first-step urgency, handling new arrivals, etc.) penalized the trajectory. **`success=false`** with a non-zero score is therefore normal.

Example (illustrative shape only):

```text
[START] task=easy_single_urgent_first env=email_triage_env model=gpt-4o-mini
[STEP]  step=1 action=reply('1','…') reward=0.61 done=false error=null
[END]   success=false steps=1 score=0.50 rewards=0.61
```

Point **`ENV_BASE_URL`** at your server (default `http://localhost:8000`).

## Advanced Usage

### Connecting to an Existing Server

If you already have an EmailTriageEnv server running, you can connect directly:

```python
from email_triage_env import EmailTriageEnv, MyAction

client = EmailTriageEnv(base_url="<ENV_HTTP_URL_HERE>")
result = client.reset()
result = client.step(MyAction(email_id="1", action_type="reply", response="Hello!"))
```

Note: When connecting to an existing server, `client.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from email_triage_env import EmailTriageEnv, MyAction

with EmailTriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: visible={len(result.observation.inbox)}")
    if result.observation.inbox:
        e = result.observation.inbox[0]
        result = env.step(MyAction(email_id=e.email_id, action_type="reply", response="Thanks."))
        print(f"reward={result.reward}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
# (EmailTriageEnvironment is defined in server/email_triage_environment.py)
app = create_app(
    EmailTriageEnvironment,  # server Environment class, not instance
    MyAction,
    MyObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from email_triage_env import EmailTriageEnv, MyAction
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            if not result.observation.inbox:
                break
            eid = result.observation.inbox[0].email_id
            result = env.step(
                MyAction(email_id=eid, action_type="reply", response=f"Client {client_id}, step {i}")
            )
        return client_id, int(result.observation.current_time)

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Smoke-test the server class without HTTP:

```bash
# From envs/email_triage_env (same directory as pyproject.toml)
python -c "from server.email_triage_environment import EmailTriageEnvironment; EmailTriageEnvironment()"
```

This checks that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
# From this directory, after `uv sync` (package on PYTHONPATH or installed in .venv):
uvicorn email_triage_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

### Validate as an OpenEnv environment

From the environment directory (where `openenv.yaml` is located):

```bash
openenv validate
```

## Usage from the meta-openenv monorepo

If you run tools from the **repository root** with `envs` on `PYTHONPATH`, use:

```python
from envs.email_triage_env.client import EmailTriageEnv
from envs.email_triage_env.models import MyAction
```

Then point the client at `ENV_BASE_URL` (e.g. `http://localhost:8000`) as in repo-root `inference.py` / `evaluator.py`.

## See also

- [`documentation.md`](documentation.md) — schemas, API surfaces, agent design notes  
- [`explanation.md`](explanation.md) — end-to-end flow, rewards, task graders vs step reward  

## Project Structure

```
email_triage_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Exports EmailTriageEnv client
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # EmailTriageEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Exports EmailTriageEnvironment
    ├── email_triage_environment.py  # EmailTriageEnvironment (OpenEnv server class)
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
