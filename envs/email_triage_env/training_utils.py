"""Discrete action helpers: map (email slot, action kind) to ``MyAction`` for visible inbox rows."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

from .models import MyAction, PublicEmail

ActionKind = Literal["reply", "escalate", "archive"]

ACTION_KINDS: Tuple[ActionKind, ...] = ("reply", "escalate", "archive")
ACTION_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(ACTION_KINDS)}


def slot_action_to_my_action(
    *,
    email_slot: int,
    action_kind_index: int,
    visible_emails: Sequence[PublicEmail],
    response: str = "",
) -> Optional[MyAction]:
    """``None`` if slot or action index is out of range."""
    if email_slot < 0 or email_slot >= len(visible_emails):
        return None
    if action_kind_index < 0 or action_kind_index >= len(ACTION_KINDS):
        return None
    e = visible_emails[email_slot]
    return MyAction(
        email_id=e.email_id,
        action_type=ACTION_KINDS[action_kind_index],
        response=response,
    )


def flat_discrete_dimensions(top_n: int) -> int:
    """Size of ``email_slot * len(ACTION_KINDS)`` when using a flattened joint discrete space."""
    return max(0, int(top_n)) * len(ACTION_KINDS)


def flat_index_to_slots(flat_index: int, top_n: int) -> Tuple[int, int]:
    """Split a flattened index into ``(email_slot, action_kind_index)``."""
    n = len(ACTION_KINDS)
    if top_n <= 0 or flat_index < 0:
        return (0, 0)
    return (flat_index // n, flat_index % n)
