"""Tests for circuit breaker safety limits."""

import pytest

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.store import DataStore
from bitbank_bot.engine.circuit_breaker import CircuitBreaker


@pytest.fixture
def store(tmp_path):
    db = DataStore(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
def breaker(store):
    cfg = StrategyConfig(
        max_daily_trades=5,
        max_daily_loss_pct=3.0,
        max_consecutive_losses=3,
        max_concurrent_positions=2,
    )
    return CircuitBreaker(store, cfg)


def test_can_trade_initially(breaker):
    assert breaker.check_can_trade("XRP/JPY", 1_000_000)


def test_consecutive_losses_halt(breaker):
    breaker.record_trade_result(-100)
    breaker.record_trade_result(-100)
    assert breaker.check_can_trade("XRP/JPY", 1_000_000)
    breaker.record_trade_result(-100)
    assert not breaker.check_can_trade("XRP/JPY", 1_000_000)
    assert "CONSECUTIVE_LOSSES" in breaker.halt_reason


def test_win_resets_streak(breaker):
    breaker.record_trade_result(-100)
    breaker.record_trade_result(-100)
    breaker.record_trade_result(200)  # Win resets streak
    breaker.record_trade_result(-100)
    assert breaker.check_can_trade("XRP/JPY", 1_000_000)


def test_reset_halted(breaker):
    for _ in range(3):
        breaker.record_trade_result(-100)
    assert not breaker.check_can_trade("XRP/JPY", 1_000_000)
    breaker.reset()
    assert breaker.check_can_trade("XRP/JPY", 1_000_000)


def test_state_persists_across_restart(store):
    """Circuit breaker state should survive creating a new instance."""
    cfg = StrategyConfig(max_consecutive_losses=3)

    # First instance: record 2 losses
    cb1 = CircuitBreaker(store, cfg)
    cb1.record_trade_result(-100)
    cb1.record_trade_result(-100)

    # Second instance (simulates restart): should restore 2 losses
    cb2 = CircuitBreaker(store, cfg)
    assert cb2._consecutive_losses == 2

    # One more loss should halt
    cb2.record_trade_result(-100)
    assert not cb2.check_can_trade("XRP/JPY", 1_000_000)
    assert "CONSECUTIVE_LOSSES" in cb2.halt_reason


def test_halt_persists_across_restart(store):
    """Halted state should survive restart."""
    cfg = StrategyConfig(max_consecutive_losses=2)

    cb1 = CircuitBreaker(store, cfg)
    cb1.record_trade_result(-100)
    cb1.record_trade_result(-100)
    cb1.check_can_trade("XRP/JPY", 1_000_000)  # triggers halt
    assert cb1.is_halted

    # New instance should still be halted
    cb2 = CircuitBreaker(store, cfg)
    assert cb2.is_halted
    assert "CONSECUTIVE_LOSSES" in cb2.halt_reason
