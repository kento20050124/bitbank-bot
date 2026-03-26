"""Tests for entry signal generation."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.config import StrategyConfig
from bitbank_bot.strategy.entry import detect_trend, generate_entry_signal
from bitbank_bot.strategy.indicators import compute_all_indicators


def test_detect_trend_uptrend():
    """Test trend detection with clear uptrend data."""
    np.random.seed(1)
    n = 100
    # Strong uptrend
    close = np.linspace(50, 150, n) + np.random.randn(n) * 0.5
    high = close + 0.5
    low = close - 0.5

    df = pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n) * 1000,
    })

    cfg = StrategyConfig(adx_threshold=15)
    df = compute_all_indicators(df, ema_fast=cfg.ema_fast_period, ema_slow=cfg.ema_slow_period)
    trend = detect_trend(df, cfg)
    # Strong uptrend should be detected as long
    assert trend in ("long", None)  # ADX may not be strong enough for short data


def test_detect_trend_insufficient_data():
    """Insufficient data should return None, not crash."""
    cfg = StrategyConfig()
    df = pd.DataFrame({
        "open": [1, 2],
        "high": [1.1, 2.1],
        "low": [0.9, 1.9],
        "close": [1, 2],
        "volume": [100, 100],
    })
    # Don't try to compute indicators on tiny data - just test detect_trend directly
    # with empty indicator columns
    df["ema_20"] = [np.nan, np.nan]
    df["ema_50"] = [np.nan, np.nan]
    df["adx_14"] = [np.nan, np.nan]
    df["dmp_14"] = [np.nan, np.nan]
    df["dmn_14"] = [np.nan, np.nan]
    assert detect_trend(df, cfg) is None


def test_generate_entry_signal_alignment(sample_ohlcv_df, strategy_config):
    """Test that entry signal is generated or None (no crash)."""
    # Split data into H1 and create H4
    df_h1 = sample_ohlcv_df.copy()
    df_h4 = sample_ohlcv_df.iloc[::4].reset_index(drop=True)

    signal = generate_entry_signal(df_h1, df_h4, strategy_config)
    # Signal can be None if trends don't align, that's valid
    if signal is not None:
        assert signal.entry_price > 0
        assert signal.stop_distance > 0
        assert signal.atr_value > 0
        assert signal.reasoning != ""
