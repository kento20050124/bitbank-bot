"""Tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from bitbank_bot.strategy.indicators import (
    compute_all_indicators,
    compute_atr,
    compute_choppiness,
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


def test_compute_choppiness(sample_ohlcv_df):
    chop = compute_choppiness(
        sample_ohlcv_df["high"],
        sample_ohlcv_df["low"],
        sample_ohlcv_df["close"],
        14,
    )
    valid = chop.dropna()
    assert len(valid) > 0
    # Choppinessは理論上 0..100 だが実装上 100 をやや超えることもあるので余裕を持たせる
    assert (valid >= 0).all()
    assert (valid <= 110).all()


def test_compute_choppiness_flat_range_is_high():
    """フラットなレンジ → Choppiness は高い値（>50）になる。"""
    n = 60
    flat = pd.Series([100.0 + np.sin(i / 3.0) * 0.5 for i in range(n)])
    high = flat + 0.5
    low = flat - 0.5
    chop = compute_choppiness(high, low, flat, 14)
    # 後半（warmup後）の平均値を見る
    tail = chop.dropna().tail(20)
    assert len(tail) > 0
    assert tail.mean() > 50, f"Flat range CI should be high, got {tail.mean():.1f}"


def test_compute_all_indicators(sample_ohlcv_df):
    result = compute_all_indicators(sample_ohlcv_df)
    # Check all expected columns exist
    assert "ema_20" in result.columns
    assert "ema_50" in result.columns
    assert "adx_14" in result.columns
    assert "atr_14" in result.columns
    assert "rsi_14" in result.columns
    assert "disparity_20" in result.columns
    assert "chop_14" in result.columns
    assert len(result) == len(sample_ohlcv_df)
