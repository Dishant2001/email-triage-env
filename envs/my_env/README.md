---
title: My Env Environment Server
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

# My Env Environment

An **inbox triage** environment where an agent manages multiple emails under **SLA deadlines**. Each step, the agent selects a pending email and chooses one action:
- `reply` (optionally with a response)
- `escalate` (costly but appropriate for VIP/high urgency)
- `archive` (for low-priority/no-action items)

The environment supports **partial observability** (top‑N emails only) and optional **seeded arrivals**.

## Quick Start

The simplest way to use the My Env environment is through the `MyEnv` class:

```python
from my_env import MyAction, MyEnv

try:
    # Create environment from Docker image
    env = MyEnv.from_docker_image("my_env-env:latest")

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

That's it! The `MyEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t my_env-env:latest -f server/Dockerfile .
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

## Environment Details

### Action
**MyAction**
- `email_id` (str) - target email id (must be pending & visible)
- `action_type` (`reply | escalate | archive`)
- `response` (str) - optional; used for deterministic reply grading

### Observation
**MyObservation**
- `current_time` (int)
- `inbox` (list[Email]) - top‑N visible pending emails
- `hidden_pending_count` (int)
- `reward` (float) - scalar shaped reward for the last step
- `done` (bool) - True when no pending emails remain
- `metadata["grade"]` (dict) - deterministic breakdown (SLA/prioritization/correctness/response/costs)

### Reward
Reward is a **multi-factor shaped signal** computed each step (see `server/grader.py`):
- SLA compliance / breach penalty
- prioritization (did you pick the most urgent pending email?)
- action correctness vs deterministic `ground_truth_action`
- reply rubric (keyword checklist)
- action costs + small per-step penalty

## Advanced Usage

### Connecting to an Existing Server

If you already have a My Env environment server running, you can connect directly:

```python
from my_env import MyEnv

# Connect to existing server
my_envenv = MyEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = my_envenv.reset()
result = my_envenv.step(MyAction(message="Hello!"))
```

Note: When connecting to an existing server, `my_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from my_env import MyAction, MyEnv

# Connect with context manager (auto-connects and closes)
with MyEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(MyAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
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
app = create_app(
    MyEnvironment,  # Pass class, not instance
    MyAction,
    MyObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from my_env import MyAction, MyEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with MyEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(MyAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/my_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

### Validate as an OpenEnv environment

From the environment directory (where `openenv.yaml` is located):

```bash
openenv validate
```

## Project Structure

```
my_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # MyEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── my_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
