# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EmailTriageEnv package (import as ``email_triage_env``)."""

from .client import EmailTriageEnv
from .models import Email, MyAction, MyObservation, MyReward, PublicEmail, to_public_email
from .tasks import (
    TASK_SCORE_HARNESS_MARGIN,
    TaskSpec,
    all_tasks,
    harness_task_score,
)
from .training_utils import (
    ACTION_KINDS,
    flat_discrete_dimensions,
    flat_index_to_slots,
    slot_action_to_my_action,
)

__all__ = [
    "ACTION_KINDS",
    "TASK_SCORE_HARNESS_MARGIN",
    "Email",
    "MyAction",
    "MyObservation",
    "MyReward",
    "PublicEmail",
    "EmailTriageEnv",
    "TaskSpec",
    "all_tasks",
    "harness_task_score",
    "flat_discrete_dimensions",
    "flat_index_to_slots",
    "slot_action_to_my_action",
    "to_public_email",
]
