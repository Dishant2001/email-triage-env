# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment Client."""

from typing import Dict, List, Optional, cast

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Email, MyAction, MyObservation, MyState


class MyEnv(
    EnvClient[MyAction, MyObservation, State]
):
    """
    Client for the My Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_time)
        ...
        ...     result = client.step(MyAction(email_id="1", action_type="escalate"))
        ...     print(result.observation.inbox)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyAction) -> Dict:
        """
        Convert MyAction to JSON payload for step message.

        Args:
            action: MyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "email_id": action.email_id,
            "action_type": action.action_type,
            "response": action.response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        """
        Parse server response into StepResult[MyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyObservation
        """
        obs_data = payload.get("observation", {})
        inbox_data = cast(List[Dict], obs_data.get("inbox", []))
        inbox = [Email.model_validate(e) for e in inbox_data]
        observation = MyObservation(
            current_time=obs_data.get("current_time", 0),
            inbox=inbox,
            hidden_pending_count=int(obs_data.get("hidden_pending_count", 0)),
            last_email_id=cast(Optional[str], obs_data.get("last_email_id")),
            last_action_type=cast(Optional[str], obs_data.get("last_action_type")),
            sla_breach=bool(obs_data.get("sla_breach", False)),
            time_advance=int(obs_data.get("time_advance", 0)),
            action_cost=float(obs_data.get("action_cost", 0.0)),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        # The server includes extra fields (current_time, emails). Parse into our
        # typed state model first, but keep return type compatible with EnvClient.
        state = MyState.model_validate(payload)
        return state
