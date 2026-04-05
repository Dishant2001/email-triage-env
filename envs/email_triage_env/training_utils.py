"""
Helpers for RL / policy optimization integrations.

The environment exposes variable ``email_id`` strings and a structured :class:`MyAction`.
Classic RL libraries often expect a fixed discrete space; use these helpers to map
**slot indices** over the current **visible** inbox (``observation.inbox``) plus an
action-type index.

This module is optional and dependency-light (no Gymnasium required). For Gymnasium
or PettingZoo wrappers, keep a single async client per worker or run the environment
in-process if your stack allows it.
"""

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
    """
    Build a :class:`MyAction` from a visible email slot and action-kind index.

    ``email_slot`` is in ``range(len(visible_emails))`` (typically ``0 .. top_n-1``).
    ``action_kind_index`` is ``0=reply, 1=escalate, 2=archive`` (see :data:`ACTION_KINDS`).
    Returns ``None`` if indices are out of range (caller may treat as invalid action).
    """
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
