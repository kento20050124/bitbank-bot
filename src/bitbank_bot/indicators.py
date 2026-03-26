"""Technical indicator calculations using the 'ta' library.

All functions are pure: they take DataFrames/Series and return Series.
Shared between live trading and backtesting for consistency.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return EMAIndicator(close=close, window=period).ema_indicator()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Average Directional Index.

    Returns DataFrame with columns: ADX_{period}, DMP_{period}, DMN_{period}.
    """
    indicator = ADXIndicator(high=high, low=low, close=close, window=period)
    result = pd.DataFrame({
        f"ADX_{period}": indicator.adx(),
        f"DMP_{period}": indicator.adx_pos(),
        f"DMN_{period}": indicator.adx_neg(),
    })
    return result


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return RSIIndicator(close=close, window=period).rsi()


def compute_disparity(close: pd.Series, ema_period: int = 20) -> pd.Series:
    """Disparity Index: percentage deviation of price from EMA.

    Formula: ((close - EMA) / EMA) * 100
    Positive = overbought territory, Negative = oversold.
    """
    ema = compute_ema(close, ema_period)
    return ((close - ema) / ema) * 100


def compute_all_indicators(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    adx_period: int = 14,
    atr_period: int = 14,
    rsi_period: int = 14,
    disparity_ema_period: int = 20,
) -> pd.DataFrame:
    """Compute all indicators and add them as columns to the DataFrame.

    Expects df to have columns: open, high, low, close, volume.
    Returns a copy with indicator columns added.
    """
    result = df.copy()

    # EMAs
    result[f"ema_{ema_fast}"] = compute_ema(result["close"], ema_fast)
    result[f"ema_{ema_slow}"] = compute_ema(result["close"], ema_slow)

    # ADX
    adx_df = compute_adx(result["high"], result["low"], result["close"], adx_period)
    if adx_df is not None:
        result[f"adx_{adx_period}"] = adx_df[f"ADX_{adx_period}"]
        result[f"dmp_{adx_period}"] = adx_df[f"DMP_{adx_period}"]
        result[f"dmn_{adx_period}"] = adx_df[f"DMN_{adx_period}"]

    # ATR
    result[f"atr_{atr_period}"] = compute_atr(
        result["high"], result["low"], result["close"], atr_period
    )

    # RSI
    result[f"rsi_{rsi_period}"] = compute_rsi(result["close"], rsi_period)

    # Disparity
    result[f"disparity_{disparity_ema_period}"] = compute_disparity(
        result["close"], disparity_ema_period
    )

    return result
