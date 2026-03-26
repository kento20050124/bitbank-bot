"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import Position, PositionState, Side


@pytest.fixture
def strategy_config():
    """Default strategy configuration for tests."""
    return StrategyConfig()


@pytest.fixture
def sample_ohlcv_df():
    """Generate sample OHLCV data for testing.

    Creates 300 H1 candles with a synthetic uptrend followed by a downtrend.
    """
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h")

    # Generate price with trend + noise
    base_price = 100.0
    trend = np.concatenate([
        np.linspace(0, 30, n // 2),      # Uptrend
        np.linspace(30, 10, n // 2),      # Downtrend
    ])
    noise = np.random.randn(n) * 1.5
    close = base_price + trend + noise

    # Generate OHLCV from close
    high = close + np.abs(np.random.randn(n)) * 0.8
    low = close - np.abs(np.random.randn(n)) * 0.8
    open_price = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_long_position():
    """Sample long position for testing."""
    return Position(
        id=1,
        symbol="XRP/JPY",
        side=Side.BUY,
        entry_price=100.0,
        amount=100.0,
        current_amount=100.0,
        stop_price=95.0,
        target_price=110.0,
        highest_price=100.0,
        state=PositionState.OPEN,
    )


@pytest.fixture
def sample_short_position():
    """Sample short position for testing."""
    return Position(
        id=2,
        symbol="XRP/JPY",
        side=Side.SELL,
        entry_price=100.0,
        amount=100.0,
        current_amount=100.0,
        stop_price=105.0,
        target_price=90.0,
        highest_price=0.0,
        lowest_price=100.0,
        state=PositionState.OPEN,
    )
