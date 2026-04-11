# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EmailTriageEnv client (WebSocket session to the inbox triage server)."""

from typing import Dict, List, Optional, cast

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import Email, MyAction, MyObservation, MyState, PublicEmail
except ImportError:
    from models import Email, MyAction, MyObservation, MyState, PublicEmail


class EmailTriageEnv(
    EnvClient[MyAction, MyObservation, State]
):
    """WebSocket client to ``EmailTriageEnvironment`` (one server session per instance)."""

    def _step_payload(self, action: MyAction) -> Dict:
        return {
            "email_id": action.email_id,
            "action_type": action.action_type,
            "response": action.response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        inbox_data = cast(List[Dict], obs_data.get("inbox", []))
        inbox = [PublicEmail.model_validate(e) for e in inbox_data]
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
        data = dict(payload)
        raw_emails = cast(List[Dict], data.pop("emails", []))
        emails: List[PublicEmail] = []
        for e in raw_emails:
            if isinstance(e, dict) and e.get("ground_truth_action") is not None:
                emails.append(Email.model_validate(e))
            else:
                emails.append(PublicEmail.model_validate(e))
        data["emails"] = emails
        return MyState.model_validate(data)
