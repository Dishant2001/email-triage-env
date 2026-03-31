# Inbox Management Environment Documentation

This environment simulates **multi-email inbox triage with SLA awareness**. An agent repeatedly selects one pending email and takes an action (`reply`, `escalate`, `archive`) while a **virtual clock** advances and new emails may arrive. Rewards are computed deterministically from a rubric-like breakdown.

The implementation follows the standard OpenEnv server/client structure:

- **Server**: `envs/my_env/server/app.py` exposes HTTP + WebSocket endpoints via OpenEnv.
- **Environment logic**: `envs/my_env/server/my_env_environment.py` (orchestrator).
- **Models**: `envs/my_env/models.py` (Action/Observation/State schemas).
- **Client**: `envs/my_env/client.py` (typed parsing + payload shaping).

## Core Concepts

### Episode flow (high level)

- **Reset**
  - Creates a starter inbox (deterministic scenario).
  - Sets `current_time = 0`, `step_count = 0`.
  - Initializes configuration (`EnvConfig`), including `top_n` and seeding for deterministic dynamics.
  - Returns an initial `MyObservation` with only the **top‑N pending emails**.

- **Step**
  - Agent picks an `email_id` and an `action_type`, optionally includes `response`.
  - Environment validates selection (must be pending).
  - Computes reward via a deterministic grader (SLA + prioritization + correctness + response rubric + costs).
  - Marks the selected email as processed.
  - Advances `current_time` by an **action-specific duration**.
  - Optionally injects **new arriving emails** (seeded stochasticity).
  - Returns next `MyObservation` (again top‑N) with grading breakdown attached in metadata.

### Virtual time & SLA

- `current_time` is a **virtual clock** (an integer).
- Each email has:
  - `created_time`
  - `sla_limit` (maximum allowed elapsed time)
- SLA breach check:
  - A breach occurs if `(current_time - created_time) > sla_limit`.

### Partial observability (Top‑N)

Observations show only the top‑N pending emails ranked by urgency score:

- `MyObservation.inbox`: visible pending emails
- `MyObservation.hidden_pending_count`: how many pending emails exist but were hidden

This forces agents to operate under limited bandwidth like a real inbox.

## Data Model Reference (Schemas)

All schemas live in `envs/my_env/models.py`.

### Email

`Email` is the environment’s inbox item. Key fields:

- **Identity & grouping**
  - `email_id`: unique identifier (string)
  - `thread_id`: groups related emails into threads
- **Content**
  - `subject`, `body`
- **Priority signals**
  - `priority`: `low | medium | high`
  - `customer_tier`: `standard | premium | vip`
- **Timing**
  - `created_time`, `sla_limit`
- **Lifecycle**
  - `status`: `pending | processed`
- **Deterministic grading labels**
  - `ground_truth_action`: the ideal action for this email (`reply | escalate | archive`)
  - `required_response_keywords`: keyword checklist for deterministic reply grading

### Action (`MyAction`)

Agent’s move per step:

- `email_id`: which email to handle
- `action_type`: `reply | escalate | archive`
- `response`: optional text (primarily meaningful for `reply`)

### Observation (`MyObservation`)

What the agent receives each step:

- **Inbox snapshot**
  - `current_time`
  - `inbox`: top‑N pending emails
  - `hidden_pending_count`
- **Last step outcome fields**
  - `last_email_id`, `last_action_type`
  - `sla_breach`
  - `time_advance`: how much virtual time the last action consumed
  - `action_cost`: direct penalty cost applied for last action
- **OpenEnv base fields**
  - `reward`, `done`, `metadata`

### State (`MyState`)

Full environment state (typically for debugging/inspection):

- `current_time`
- `emails`: full inbox list (pending + processed)
- `config`: `EnvConfig` used for this episode
- `new_emails_added`: how many arrivals have been injected so far

### Reset configuration (`EnvConfig`)

`EnvConfig` controls difficulty and realism:

- **Observability**
  - `top_n`: number of pending emails shown
- **Arrivals**
  - `arrivals_enabled`
  - `max_new_emails`
  - `seed` (for deterministic arrivals)
- **Time & costs**
  - `action_durations`: e.g. `reply:2`, `escalate:1`, `archive:1`
  - `action_costs`: penalize actions (e.g., escalation fatigue)
  - `per_step_idle_cost`: small constant penalty each step

How to pass it at reset time:

```python
env.reset(config={"top_n": 3, "seed": 123, "arrivals_enabled": True, "max_new_emails": 2})
```

## Modular Server Architecture

The server is split into small modules for clarity and extensibility.

### `server/my_env_environment.py` (Orchestrator)

Responsibilities:

- Holds the authoritative `MyState`
- Implements OpenEnv lifecycle methods:
  - `reset(seed, episode_id, **kwargs)`
  - `step(action)`
  - `state` property
- Calls into:
  - dynamics: top‑N selection, arrivals, action duration
  - grader: deterministic reward breakdown
  - scenarios: starter inbox and arrival templates

### `server/scenarios.py` (Scenarios)

Defines:

- `starter_inbox()`: deterministic initial emails (with ground truth labels)
- `arrival_templates()`: templates used to generate new incoming emails

Extend this module to add:

- harder scenarios
- more diverse domains (billing, incidents, refunds, account issues)
- multi-turn threads (follow-up emails)

### `server/dynamics.py` (Dynamics)

Defines:

- **Urgency scoring** combining:
  - priority
  - customer tier
  - remaining SLA time
- `select_top_n_pending(...)` for partial observation
- variable time advancement via `apply_action_dynamics(...)`
- seeded arrivals via `maybe_generate_arrivals(...)`

### `server/grader.py` (Deterministic grading)

Computes a `GradeBreakdown` with:

- `sla_score`: +0.5 if within SLA, −1.0 if breached
- `prioritization_score`: +0.4 if agent chose the most urgent pending email
- `action_score`: +0.3 if action matches `ground_truth_action`
- `response_score`: up to +0.3 based on keyword checklist (or length proxy if none)
- `cost_penalty`: negative cost from `action_costs`
- `idle_penalty`: negative constant penalty each step
- `total`: sum of the above

The breakdown is attached to `MyObservation.metadata["grade"]` for transparency.

## Reward Design (Why it works for agents)

This environment’s reward is intentionally **multi-factor**:

- encourages **SLA compliance**
- encourages **good prioritization**
- encourages **correct action selection**
- provides a simple, deterministic **response rubric**
- discourages degenerate strategies through **costs**

This creates meaningful trade-offs:

- escalating everything is expensive
- replying well takes longer (time) but can score higher
- missing urgent/VIP emails hurts via SLA + prioritization penalties

## Agent Design (Everything you need)

This environment is designed to support multiple agent styles: rule-based baselines, LLM agents, and hybrids.

### Agent I/O contract

At each step, an agent receives a `MyObservation` and must output a `MyAction`:

- Select `email_id` from `observation.inbox`
- Choose `action_type`
- Provide `response` when replying (optional otherwise)

### Baseline 1: Greedy rule-based agent (recommended baseline)

**Goal**: produce a strong deterministic baseline that is cheap and explainable.

Typical behavior:

- score each visible email by urgency (SLA remaining + priority + tier)
- choose the max-scoring email
- choose `action_type`:
  - `escalate` if urgent/high/VIP or near SLA
  - `archive` if low priority / “no action required”
  - else `reply`
- craft a minimal reply template when replying that includes likely required keywords

Why it’s useful:

- demonstrates the environment is learnable
- provides a floor for evaluation
- doubles as the “filtering policy” in a hybrid agent

### Baseline 2: LLM-only agent

**Goal**: maximize correctness and response quality via language understanding.

Approach:

- prompt the LLM with the visible emails, current time, and instructions
- ask it to output structured JSON matching `MyAction`

Trade-offs:

- higher latency/cost
- risk of inconsistent formatting unless you enforce strict JSON schema

### Best-in-practice: Hybrid agent (Filtering + Decision)

This matches the hackathon design best:

- **Filtering policy (rule-based)**
  - shortlist top‑K emails (K ≤ top_n) using urgency score
  - optionally compress/normalize fields for prompting (subject/body snippets, SLA remaining)
- **Decision policy (LLM)**
  - choose final email + action
  - generate response only when replying

Why it’s stronger:

- reduces LLM context + cost
- keeps a deterministic “safety net” for urgency handling
- improves reliability and demo clarity (“why did we pick these 3?”)

### Execution loop (agent-runner pseudocode)

```python
obs = env.reset(config={"top_n": 5, "seed": 123})
done = False
while not done:
    action = agent.act(obs)  # returns MyAction
    step = env.step(action)
    obs = step.observation
    done = step.done
```

### How to evaluate an agent

Use episode aggregates:

- **Total reward** (sum of `reward`)
- **SLA breach count** (`metadata["grade"]["sla_breach"]`)
- **Correct action rate** (`action_score`)
- **Prioritization hit rate** (`prioritization_score`)
- **Cost spent** (sum of `action_cost`)

Because grading is deterministic, you can run the same episode multiple times under a fixed seed and compare agents fairly.

## Extending the Environment (Common next upgrades)

- **Thread follow-ups**: after a `reply`, inject a follow-up email in the same `thread_id`.
- **“Read/open” actions**: make bodies hidden until the agent spends an action to open an email.
- **Team routing**: add an “assignee” dimension to escalations for richer action space.
- **Better response rubric**: replace keywords with deterministic checklists (fields, required questions, tone).

