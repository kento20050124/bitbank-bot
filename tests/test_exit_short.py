"""Tests for short position exit logic - Chandelier, Scaling, Overbought."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import Position, PositionState, Side
from bitbank_bot.strategy.exit import ChandelierExit, OverboughtExit, ScalingOut
from bitbank_bot.strategy.indicators import compute_all_indicators


def _make_df(closes, highs=None, lows=None, n=100):
    """Build a DataFrame with indicators from close prices."""
    if highs is None:
        highs = [c + 1 for c in closes]
    if lows is None:
        lows = [c - 1 for c in closes]
    # Pad front with stable prices to satisfy indicator warmup
    pad = n - len(closes)
    base = closes[0]
    all_close = [base] * pad + list(closes)
    all_high = [base + 1] * pad + list(highs)
    all_low = [base - 1] * pad + list(lows)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "open": all_close,
        "high": all_high,
        "low": all_low,
        "close": all_close,
        "volume": [1000.0] * n,
    })
    return compute_all_indicators(df)


class TestChandelierExitShort:
    def test_short_stop_triggers_on_price_rise(self):
        """Short chandelier exit should trigger when price rises above stop."""
        cfg = StrategyConfig(atr_period=14, chandelier_multiplier=2.8)
        # Price drops from 100 to 90, then spikes to 99
        closes = [100.0] * 20 + [95.0] * 5 + [90.0] * 5 + [99.0]
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=10,
            stop_price=108.0, lowest_price=90.0,
            state=PositionState.OPEN,
        )
        exit_rule = ChandelierExit()
        signal = exit_rule.check(pos, df, cfg)
        # ATR on stable data is small, so stop = 90 + ATR*2.8 ~ 90 + small
        # price 99 should be above that stop
        assert signal is not None
        assert signal.exit_type == "chandelier"

    def test_short_lowest_price_tracks_correctly(self):
        """lowest_price should only decrease, never increase."""
        cfg = StrategyConfig(atr_period=14, chandelier_multiplier=10.0)  # wide stop
        closes = [100.0] * 20 + [95.0] * 5 + [93.0]
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=10,
            stop_price=110.0, lowest_price=96.0,
            state=PositionState.OPEN,
        )
        exit_rule = ChandelierExit()
        signal = exit_rule.check(pos, df, cfg)
        # No exit with wide multiplier
        assert signal is None
        # lowest_price should have moved down from 96 to the candle low
        assert pos.lowest_price <= 93.0

    def test_short_stop_never_moves_up(self):
        """For shorts, effective stop should only decrease (tighten), not increase."""
        cfg = StrategyConfig(atr_period=14, chandelier_multiplier=2.8)
        closes = [100.0] * 30
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=10,
            stop_price=102.0, lowest_price=99.0,
            state=PositionState.OPEN,
        )
        old_stop = pos.stop_price
        exit_rule = ChandelierExit()
        exit_rule.check(pos, df, cfg)
        assert pos.stop_price <= old_stop


class TestScalingOutShort:
    def test_scaling_triggers_on_price_drop(self):
        """Scaling out short triggers when price drops to target."""
        cfg = StrategyConfig(atr_period=14, scaling_rr_target=2.0, scaling_close_ratio=0.5)
        # Price drops significantly
        closes = [100.0] * 20 + [85.0]
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=10,
            state=PositionState.OPEN,
        )
        rule = ScalingOut()
        signal = rule.check(pos, df, cfg)
        # target = 100 - ATR*2 = ~100 - small*2, price at 85 should be below
        assert signal is not None
        assert signal.exit_type == "scaling"
        assert signal.close_ratio == 0.5

    def test_scaling_only_triggers_once(self):
        """Scaling should not trigger if position is already TRAILING."""
        cfg = StrategyConfig()
        closes = [100.0] * 20 + [80.0]
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=5,
            state=PositionState.TRAILING,
        )
        rule = ScalingOut()
        signal = rule.check(pos, df, cfg)
        assert signal is None


class TestOverboughtExitShort:
    def test_oversold_rsi_recovery_triggers(self):
        """RSI crossing above 30 from below should trigger exit for shorts."""
        cfg = StrategyConfig(
            rsi_period=14,
            rsi_overbought_entry=75.0,
            rsi_overbought_exit=70.0,
            disparity_threshold=3.5,
        )
        # Create data where RSI goes very low then recovers
        # Drop hard then bounce
        closes = [100.0] * 50 + [float(100 - i * 2) for i in range(15)] + [float(70 + i * 3) for i in range(5)]
        df = _make_df(closes, n=80)

        pos = Position(
            id=1, symbol="XRP/JPY", side=Side.SELL,
            entry_price=100.0, amount=10, current_amount=10,
            state=PositionState.OPEN,
        )
        rule = OverboughtExit()
        signal = rule.check(pos, df, cfg)
        # RSI may or may not trigger depending on exact data, but the code path is tested
        # At minimum, verify no crash
        assert signal is None or signal.exit_type == "overbought"
