"""Tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.strategy.indicators import (
    compute_all_indicators,
    compute_atr,
    compute_disparity,
    compute_ema,
    compute_rsi,
)


def test_compute_ema(sample_ohlcv_df):
    ema = compute_ema(sample_ohlcv_df["close"], 20)
    assert len(ema) == len(sample_ohlcv_df)
    # After warmup should have valid values
    valid = ema.dropna()
    assert len(valid) > 0


def test_compute_rsi(sample_ohlcv_df):
    rsi = compute_rsi(sample_ohlcv_df["close"], 14)
    valid = rsi.dropna()
    assert len(valid) > 0
    # RSI should be between 0 and 100
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_compute_atr(sample_ohlcv_df):
    atr = compute_atr(
        sample_ohlcv_df["high"],
        sample_ohlcv_df["low"],
        sample_ohlcv_df["close"],
        14,
    )
    # The ta library may return 0 for initial rows, which is not NaN
    # Filter for values after warmup period
    valid = atr.iloc[14:]
    non_zero = valid[valid > 0]
    assert len(non_zero) > 0
    # All non-zero ATR values should be positive
    assert (non_zero > 0).all()


def test_compute_disparity(sample_ohlcv_df):
    disp = compute_disparity(sample_ohlcv_df["close"], 20)
    valid = disp.dropna()
    assert len(valid) > 0


def test_compute_all_indicators(sample_ohlcv_df):
    result = compute_all_indicators(sample_ohlcv_df)
    # Check all expected columns exist
    assert "ema_20" in result.columns
    assert "ema_50" in result.columns
    assert "adx_14" in result.columns
    assert "atr_14" in result.columns
    assert "rsi_14" in result.columns
    assert "disparity_20" in result.columns
    assert len(result) == len(sample_ohlcv_df)
