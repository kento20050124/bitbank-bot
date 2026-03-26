"""Position state machine: manages lifecycle transitions."""

from __future__ import annotations

import logging
from datetime import datetime

from bitbank_bot.data.models import Position, PositionState

logger = logging.getLogger(__name__)

# Valid state transitions
TRANSITIONS: dict[PositionState, set[PositionState]] = {
    PositionState.FLAT: {PositionState.PENDING_ENTRY},
    PositionState.PENDING_ENTRY: {PositionState.OPEN, PositionState.FLAT},
    PositionState.OPEN: {
        PositionState.SCALING_OUT,
        PositionState.PENDING_EXIT,
        PositionState.EMERGENCY_EXIT,
    },
    PositionState.SCALING_OUT: {PositionState.TRAILING},
    PositionState.TRAILING: {
        PositionState.PENDING_EXIT,
        PositionState.EMERGENCY_EXIT,
    },
    PositionState.PENDING_EXIT: {PositionState.CLOSED, PositionState.EMERGENCY_EXIT},
    PositionState.EMERGENCY_EXIT: {PositionState.CLOSED},
    PositionState.CLOSED: set(),
}


def transition(position: Position, new_state: PositionState, reason: str = "") -> Position:
    """Attempt to transition a position to a new state.

    Raises ValueError if the transition is invalid.
    """
    allowed = TRANSITIONS.get(position.state, set())
    if new_state not in allowed:
        raise ValueError(
            f"Invalid transition: {position.state.value} -> {new_state.value}. "
            f"Allowed: {[s.value for s in allowed]}"
        )

    old_state = position.state
    position.state = new_state

    if new_state == PositionState.CLOSED:
        position.closed_at = datetime.now()

    if reason:
        position.reasoning = reason

    logger.info(
        "Position %s: %s -> %s (%s)",
        position.id or "new",
        old_state.value,
        new_state.value,
        reason,
    )

    return position
