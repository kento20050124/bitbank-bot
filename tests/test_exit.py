"""Tests for exit signal generation."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import Position, PositionState, Side
from bitbank_bot.strategy.exit import (
    ChandelierExit,
    OverboughtExit,
    ScalingOut,
    check_exit_conditions,
)


def _make_df(prices, atr_val=2.0, rsi_val=50.0, disparity_val=0.0):
    """Create a DataFrame with pre-computed indicator columns.

    Must have at least atr_period+1 rows (default 15) to pass length checks.
    """
    n = len(prices)
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000] * n,
        "atr_14": [atr_val] * n,
        "rsi_14": [rsi_val] if isinstance(rsi_val, (int, float)) and n == 1 else
                  [rsi_val] * n if isinstance(rsi_val, (int, float)) else rsi_val,
        "disparity_20": [disparity_val] * n,
    })
    return df


def _pad_prices(prices, pad_to=20, base=100.0):
    """Pad a price list to at least pad_to length with base values at the start."""
    needed = pad_to - len(prices)
    if needed > 0:
        return [base] * needed + prices
    return prices


def test_chandelier_exit_triggers():
    """Chandelier exit triggers when price drops below trailing stop."""
    cfg = StrategyConfig(chandelier_multiplier=2.0)
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        highest_price=110.0,
        stop_price=90.0,
        state=PositionState.TRAILING,
        current_amount=10,
    )
    # Chandelier stop = 110 - 2*2.0 = 106
    # Last price = 100, which is below 106
    prices = _pad_prices(list(range(110, 99, -1)), pad_to=20, base=110)
    df = _make_df(prices)

    exit_rule = ChandelierExit()
    signal = exit_rule.check(position, df, cfg)
    assert signal is not None
    assert signal.exit_type == "chandelier"


def test_chandelier_exit_no_trigger():
    """Chandelier exit should not trigger when price is above stop."""
    cfg = StrategyConfig(chandelier_multiplier=2.0)
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        highest_price=105.0,
        stop_price=90.0,
        state=PositionState.OPEN,
        current_amount=10,
    )
    prices = _pad_prices([104, 105, 106, 107, 108], pad_to=20, base=104)
    df = _make_df(prices)

    exit_rule = ChandelierExit()
    signal = exit_rule.check(position, df, cfg)
    assert signal is None


def test_scaling_out_triggers():
    """Scaling out triggers at RR target."""
    cfg = StrategyConfig(scaling_rr_target=2.0, scaling_close_ratio=0.5)
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        state=PositionState.OPEN,
        current_amount=10,
    )
    # Target: 100 + 2.0 * 2.0 = 104, last price = 105
    prices = _pad_prices([101, 102, 103, 104, 105], pad_to=20, base=101)
    df = _make_df(prices, atr_val=2.0)

    rule = ScalingOut()
    signal = rule.check(position, df, cfg)
    assert signal is not None
    assert signal.exit_type == "scaling"
    assert signal.close_ratio == 0.5


def test_scaling_out_only_when_open():
    """Scaling should not trigger if already trailing."""
    cfg = StrategyConfig()
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        state=PositionState.TRAILING,
        current_amount=5,
    )
    prices = _pad_prices([110, 115, 120], pad_to=20, base=110)
    df = _make_df(prices)

    rule = ScalingOut()
    assert rule.check(position, df, cfg) is None


def test_overbought_exit_disparity():
    """Overbought exit triggers on high disparity."""
    cfg = StrategyConfig(disparity_threshold=3.0)
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        state=PositionState.OPEN,
        current_amount=10,
    )
    prices = _pad_prices([105, 106, 107], pad_to=25, base=105)
    n = len(prices)
    df = _make_df(prices, disparity_val=4.0, rsi_val=65)

    rule = OverboughtExit()
    signal = rule.check(position, df, cfg)
    assert signal is not None
    assert signal.exit_type == "overbought"


def test_overbought_exit_rsi_drop():
    """Overbought exit triggers on RSI dropping from overbought zone."""
    cfg = StrategyConfig(rsi_overbought_entry=75, rsi_overbought_exit=70)
    position = Position(
        side=Side.BUY,
        entry_price=100.0,
        state=PositionState.OPEN,
        current_amount=10,
    )
    # Need 25+ rows. RSI drops from 76 to 68 at the end.
    n = 25
    prices = [105.0] * n
    rsi_values = [76.0] * (n - 1) + [68.0]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000] * n,
        "atr_14": [2.0] * n,
        "rsi_14": rsi_values,
        "disparity_20": [1.0] * n,
    })

    rule = OverboughtExit()
    signal = rule.check(position, df, cfg)
    assert signal is not None
    assert signal.exit_type == "overbought"
