"""Entry signal generation using multi-timeframe trend alignment."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import Signal, SignalDirection

logger = logging.getLogger(__name__)


def detect_trend(df: pd.DataFrame, cfg: StrategyConfig) -> str | None:
    """Detect trend direction from a DataFrame with pre-computed indicators.

    Returns "long", "short", or None.

    Conditions for LONG:
    - Price > EMA_fast > EMA_slow
    - ADX > threshold (trend has strength)
    - DI+ > DI- (bullish directional movement)

    Conditions for SHORT:
    - Price < EMA_fast < EMA_slow
    - ADX > threshold
    - DI- > DI+ (bearish directional movement)
    """
    if df.empty or len(df) < cfg.ema_slow_period + 10:
        return None

    last = df.iloc[-1]
    ema_fast_col = f"ema_{cfg.ema_fast_period}"
    ema_slow_col = f"ema_{cfg.ema_slow_period}"
    adx_col = f"adx_{cfg.adx_period}"
    dmp_col = f"dmp_{cfg.adx_period}"
    dmn_col = f"dmn_{cfg.adx_period}"

    # Check required columns exist
    for col in [ema_fast_col, ema_slow_col, adx_col, dmp_col, dmn_col]:
        if col not in df.columns or pd.isna(last.get(col)):
            return None

    price = last["close"]
    ema_fast = last[ema_fast_col]
    ema_slow = last[ema_slow_col]
    adx = last[adx_col]
    dmp = last[dmp_col]
    dmn = last[dmn_col]

    # ADX must show trend strength
    if adx < cfg.adx_threshold:
        return None

    # Long conditions
    if price > ema_fast > ema_slow and dmp > dmn:
        return "long"

    # Short conditions
    if price < ema_fast < ema_slow and dmn > dmp:
        return "short"

    return None


def generate_entry_signal(
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    cfg: StrategyConfig,
) -> Signal | None:
    """Generate an entry signal based on multi-timeframe alignment.

    Args:
        df_h1: DataFrame with H1 candles + pre-computed indicators.
        df_h4: DataFrame with H4 candles + pre-computed indicators.
        cfg: Strategy configuration.

    Returns:
        Signal if entry conditions are met, None otherwise.
    """
    # Detect trends on both timeframes (indicators already computed by caller)
    h4_trend = detect_trend(df_h4, cfg)
    h1_trend = detect_trend(df_h1, cfg)

    if h4_trend is None or h1_trend is None:
        return None

    # Trends must align
    if h4_trend != h1_trend:
        logger.debug("Trend mismatch: H4=%s, H1=%s", h4_trend, h1_trend)
        return None

    direction = SignalDirection.LONG if h1_trend == "long" else SignalDirection.SHORT

    # Get current price and ATR for stop calculation
    last_h1 = df_h1.iloc[-1]
    atr_col = f"atr_{cfg.atr_period}"
    atr_value = last_h1[atr_col]

    if pd.isna(atr_value) or atr_value <= 0:
        logger.warning("Invalid ATR value: %s", atr_value)
        return None

    entry_price = last_h1["close"]
    stop_distance = atr_value * cfg.chandelier_multiplier

    # Build reasoning
    adx_h1 = last_h1[f"adx_{cfg.adx_period}"]
    adx_h4 = df_h4.iloc[-1][f"adx_{cfg.adx_period}"]
    reasoning = (
        f"ENTRY_{direction.value.upper()}: "
        f"H4 trend={h4_trend} (ADX={adx_h4:.1f}), "
        f"H1 trend={h1_trend} (ADX={adx_h1:.1f}), "
        f"price={entry_price:.4f}, "
        f"ATR={atr_value:.4f}, "
        f"stop_distance={stop_distance:.4f}"
    )

    logger.info(reasoning)

    return Signal(
        direction=direction,
        entry_price=entry_price,
        stop_distance=stop_distance,
        atr_value=atr_value,
        timestamp=last_h1["timestamp"] if "timestamp" in df_h1.columns else datetime.now(),
        reasoning=reasoning,
    )
