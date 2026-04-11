# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for EmailTriageEnvironment (HTTP + WebSocket via openenv)."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MyAction, MyObservation
    from .email_triage_environment import EmailTriageEnvironment
except ImportError:  # pragma: no cover
    from models import MyAction, MyObservation
    from server.email_triage_environment import EmailTriageEnvironment


app = create_app(
    EmailTriageEnvironment,
    MyAction,
    MyObservation,
    env_name="email_triage_env",
    max_concurrent_envs=1,
)


def main() -> None:
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
