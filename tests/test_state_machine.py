"""Tests for position state machine transitions."""

import pytest

from bitbank_bot.data.models import Position, PositionState, Side
from bitbank_bot.engine.state_machine import transition


def _make_position(state: PositionState) -> Position:
    return Position(
        id=1,
        symbol="XRP/JPY",
        side=Side.BUY,
        entry_price=100.0,
        amount=10,
        current_amount=10,
        state=state,
    )


def test_valid_transitions():
    # FLAT -> PENDING_ENTRY
    p = _make_position(PositionState.FLAT)
    transition(p, PositionState.PENDING_ENTRY, "test")
    assert p.state == PositionState.PENDING_ENTRY

    # PENDING_ENTRY -> OPEN
    transition(p, PositionState.OPEN, "filled")
    assert p.state == PositionState.OPEN

    # OPEN -> SCALING_OUT
    transition(p, PositionState.SCALING_OUT, "target hit")
    assert p.state == PositionState.SCALING_OUT

    # SCALING_OUT -> TRAILING
    transition(p, PositionState.TRAILING, "trailing")
    assert p.state == PositionState.TRAILING

    # TRAILING -> PENDING_EXIT
    transition(p, PositionState.PENDING_EXIT, "chandelier")
    assert p.state == PositionState.PENDING_EXIT

    # PENDING_EXIT -> CLOSED
    transition(p, PositionState.CLOSED, "filled")
    assert p.state == PositionState.CLOSED
    assert p.closed_at is not None


def test_invalid_transition_raises():
    p = _make_position(PositionState.FLAT)
    with pytest.raises(ValueError, match="Invalid transition"):
        transition(p, PositionState.OPEN, "skip pending")


def test_closed_cannot_transition():
    p = _make_position(PositionState.CLOSED)
    with pytest.raises(ValueError):
        transition(p, PositionState.FLAT, "reopen")


def test_emergency_exit_from_open():
    p = _make_position(PositionState.OPEN)
    transition(p, PositionState.EMERGENCY_EXIT, "stop breach")
    assert p.state == PositionState.EMERGENCY_EXIT
    transition(p, PositionState.CLOSED, "market filled")
    assert p.state == PositionState.CLOSED
