"""Backtesting engine using pandas-based simulation.

Simulates the trend-following strategy with Chandelier Exit, Scaling Out,
and Overbought detection on historical data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bitbank_bot.config import StrategyConfig
from bitbank_bot.strategy.indicators import compute_all_indicators

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_bars: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: pd.Series | None = None


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate H1 candles into H4 candles."""
    df = df_1h.copy()
    df = df.set_index("timestamp")
    df_4h = df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    df_4h = df_4h.dropna().reset_index()
    return df_4h


def run_backtest(
    df_1h: pd.DataFrame,
    cfg: StrategyConfig,
    initial_equity: float = 1_000_000.0,
    use_maker_fee: bool = True,
) -> BacktestResult:
    """Run a full backtest of the trend-following strategy.

    Args:
        df_1h: DataFrame with H1 OHLCV data. Must have columns:
               timestamp, open, high, low, close, volume.
        cfg: Strategy configuration.
        initial_equity: Starting equity in JPY.
        use_maker_fee: If True, use maker fee; otherwise taker fee.

    Returns:
        BacktestResult with performance metrics and trade list.
    """
    fee = cfg.maker_fee if use_maker_fee else cfg.taker_fee

    # Compute H4 data from H1
    df_4h = _resample_to_4h(df_1h)

    # Compute indicators
    df_h1 = compute_all_indicators(
        df_1h,
        ema_fast=cfg.ema_fast_period,
        ema_slow=cfg.ema_slow_period,
        adx_period=cfg.adx_period,
        atr_period=cfg.atr_period,
        rsi_period=cfg.rsi_period,
        disparity_ema_period=cfg.disparity_ema_period,
    )
    df_h4 = compute_all_indicators(
        df_4h,
        ema_fast=cfg.ema_fast_period,
        ema_slow=cfg.ema_slow_period,
        adx_period=cfg.adx_period,
        atr_period=cfg.atr_period,
        rsi_period=cfg.rsi_period,
        disparity_ema_period=cfg.disparity_ema_period,
    )

    # Column names
    ema_fast_col = f"ema_{cfg.ema_fast_period}"
    ema_slow_col = f"ema_{cfg.ema_slow_period}"
    adx_col = f"adx_{cfg.adx_period}"
    dmp_col = f"dmp_{cfg.adx_period}"
    dmn_col = f"dmn_{cfg.adx_period}"
    atr_col = f"atr_{cfg.atr_period}"
    rsi_col = f"rsi_{cfg.rsi_period}"
    disp_col = f"disparity_{cfg.disparity_ema_period}"

    # Ensure H4 data is aligned with H1 timestamps (use int64 for searchsorted compat)
    h4_timestamps = pd.to_datetime(df_h4["timestamp"]).values.astype("int64")
    h4_adx = df_h4[adx_col].values if adx_col in df_h4.columns else np.full(len(df_h4), np.nan)
    h4_ema_fast = df_h4[ema_fast_col].values if ema_fast_col in df_h4.columns else np.full(len(df_h4), np.nan)
    h4_ema_slow = df_h4[ema_slow_col].values if ema_slow_col in df_h4.columns else np.full(len(df_h4), np.nan)
    h4_dmp = df_h4[dmp_col].values if dmp_col in df_h4.columns else np.full(len(df_h4), np.nan)
    h4_dmn = df_h4[dmn_col].values if dmn_col in df_h4.columns else np.full(len(df_h4), np.nan)
    h4_close = df_h4["close"].values

    # State variables
    equity = initial_equity
    position_side = None  # "long" or "short" or None
    entry_price = 0.0
    position_size = 0.0
    current_size = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0  # For longs
    lowest_since_entry = float("inf")  # For shorts
    entry_bar = 0
    scaled_out = False
    last_exit_bar = -999  # Re-entry cooldown tracker

    trades = []
    equity_history = []

    # Warmup period: EMA needs ~3x period to stabilize, ADX needs ~2x
    warmup = max(cfg.ema_slow_period * 3, cfg.adx_period * 2, cfg.atr_period, cfg.rsi_period) + 10

    for i in range(warmup, len(df_h1)):
        row = df_h1.iloc[i]
        prev_row = df_h1.iloc[i - 1]
        ts = row["timestamp"]
        price = row["close"]
        high = row["high"]
        low = row["low"]

        # Get corresponding H4 values
        ts_int = pd.Timestamp(ts).value
        h4_idx = np.searchsorted(h4_timestamps, ts_int, side="right") - 1
        if h4_idx < warmup // 4:
            equity_history.append(equity)
            continue

        # H4 trend detection
        h4_trend = None
        if h4_idx >= 0 and h4_idx < len(h4_close):
            h4_p = h4_close[h4_idx]
            h4_ef = h4_ema_fast[h4_idx]
            h4_es = h4_ema_slow[h4_idx]
            h4_a = h4_adx[h4_idx]
            h4_dp = h4_dmp[h4_idx]
            h4_dn = h4_dmn[h4_idx]
            if not any(np.isnan([h4_a, h4_ef, h4_es, h4_dp, h4_dn])):
                if h4_a >= cfg.adx_threshold:
                    if h4_p > h4_ef > h4_es and h4_dp > h4_dn:
                        h4_trend = "long"
                    elif h4_p < h4_ef < h4_es and h4_dn > h4_dp:
                        h4_trend = "short"

        # H1 trend detection
        h1_trend = None
        h1_ef = row.get(ema_fast_col)
        h1_es = row.get(ema_slow_col)
        h1_a = row.get(adx_col)
        h1_dp = row.get(dmp_col)
        h1_dn = row.get(dmn_col)
        if not any(pd.isna(v) for v in [h1_a, h1_ef, h1_es, h1_dp, h1_dn]):
            if h1_a >= cfg.adx_threshold:
                if price > h1_ef > h1_es and h1_dp > h1_dn:
                    h1_trend = "long"
                elif price < h1_ef < h1_es and h1_dn > h1_dp:
                    h1_trend = "short"

        atr = row.get(atr_col, 0)
        rsi = row.get(rsi_col)
        rsi_prev = prev_row.get(rsi_col)
        disparity = row.get(disp_col)

        # --- Exit logic ---
        if position_side is not None and not pd.isna(atr) and atr > 0:
            exit_reason = None
            exit_ratio = 1.0

            if position_side == "long":
                highest_since_entry = max(highest_since_entry, high)

                # 1. Scaling out check (before chandelier)
                if not scaled_out:
                    target = entry_price + atr * cfg.scaling_rr_target
                    if price >= target:
                        exit_reason = "scaling"
                        exit_ratio = cfg.scaling_close_ratio

                # 2. Overbought check
                if exit_reason is None and not pd.isna(disparity) and not pd.isna(rsi) and not pd.isna(rsi_prev):
                    if disparity > cfg.disparity_threshold:
                        exit_reason = "overbought_disparity"
                    elif rsi_prev >= cfg.rsi_overbought_entry and rsi < cfg.rsi_overbought_exit:
                        exit_reason = "overbought_rsi"

                # 3. Chandelier exit
                if exit_reason is None:
                    chandelier_stop = highest_since_entry - atr * cfg.chandelier_multiplier
                    effective_stop = max(chandelier_stop, stop_price)
                    stop_price = effective_stop
                    if price <= effective_stop:
                        exit_reason = "chandelier"

            elif position_side == "short":
                lowest_since_entry = min(lowest_since_entry, low)

                if not scaled_out:
                    target = entry_price - atr * cfg.scaling_rr_target
                    if price <= target:
                        exit_reason = "scaling"
                        exit_ratio = cfg.scaling_close_ratio

                if exit_reason is None and not pd.isna(disparity) and not pd.isna(rsi) and not pd.isna(rsi_prev):
                    recovery_level = 100 - cfg.rsi_overbought_exit  # 30
                    if disparity < -cfg.disparity_threshold:
                        exit_reason = "oversold_disparity"
                    elif rsi_prev <= recovery_level and rsi > recovery_level:
                        exit_reason = "oversold_rsi"

                if exit_reason is None:
                    chandelier_stop = lowest_since_entry + atr * cfg.chandelier_multiplier
                    # For shorts, stop only moves DOWN (tighter)
                    effective_stop = min(chandelier_stop, stop_price) if stop_price > 0 else chandelier_stop
                    stop_price = effective_stop
                    if price >= effective_stop:
                        exit_reason = "chandelier"

            # Execute exit
            if exit_reason is not None:
                close_size = current_size * exit_ratio

                # Fee: negative = maker rebate, positive = taker cost
                # Only charge EXIT fee here (entry fee already deducted at entry)
                exit_cost = close_size * price * fee
                if position_side == "long":
                    pnl = (price - entry_price) * close_size - exit_cost
                else:
                    pnl = (entry_price - price) * close_size - exit_cost

                equity += pnl
                current_size -= close_size

                if exit_reason == "scaling":
                    scaled_out = True
                    stop_price = entry_price  # Move stop to breakeven
                    trades.append({
                        "entry_time": df_h1.iloc[entry_bar]["timestamp"],
                        "exit_time": ts,
                        "side": position_side,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "size": close_size,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "partial": True,
                        "bars_held": i - entry_bar,
                    })
                else:
                    # Full exit
                    trades.append({
                        "entry_time": df_h1.iloc[entry_bar]["timestamp"],
                        "exit_time": ts,
                        "side": position_side,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "size": current_size + close_size,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "partial": False,
                        "bars_held": i - entry_bar,
                    })
                    position_side = None
                    current_size = 0.0
                    scaled_out = False
                    stop_price = 0.0
                    last_exit_bar = i

                if current_size <= 0:
                    position_side = None
                    current_size = 0.0
                    scaled_out = False
                    stop_price = 0.0
                    last_exit_bar = i

        # --- Entry logic (with 4-bar re-entry cooldown) ---
        bars_since_exit = i - last_exit_bar
        if position_side is None and bars_since_exit >= 4 and h4_trend and h1_trend == h4_trend and not pd.isna(atr) and atr > 0:
            stop_dist = atr * cfg.chandelier_multiplier
            risk_amount = equity * (cfg.risk_per_trade_pct / 100.0)
            pos_size = risk_amount / stop_dist

            max_pos_value = equity * (cfg.max_position_pct / 100.0)
            max_pos_size = max_pos_value / price
            pos_size = min(pos_size, max_pos_size)

            if pos_size > 0:
                # fee is negative for maker (rebate), positive for taker
                # entry_cost: positive = pay, negative = receive
                entry_cost = pos_size * price * fee
                equity -= entry_cost  # Subtract cost (adds money if maker rebate)

                position_side = h1_trend
                entry_price = price
                position_size = pos_size
                current_size = pos_size
                entry_bar = i
                scaled_out = False
                if h1_trend == "long":
                    highest_since_entry = high
                    lowest_since_entry = float("inf")
                else:
                    lowest_since_entry = low
                    highest_since_entry = 0.0

                if h1_trend == "long":
                    stop_price = price - stop_dist
                else:
                    stop_price = price + stop_dist

        equity_history.append(equity)

    # --- Compute metrics ---
    result = BacktestResult()
    result.trades = trades

    if not trades:
        result.equity_curve = pd.Series(equity_history)
        return result

    pnls = [t["pnl"] for t in trades if not t.get("partial")]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    result.total_trades = len(pnls)
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    result.avg_win_pct = (sum(wins) / len(wins) / initial_equity * 100) if wins else 0
    result.avg_loss_pct = (sum(losses) / len(losses) / initial_equity * 100) if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    result.total_return_pct = (equity - initial_equity) / initial_equity * 100

    bars_held = [t["bars_held"] for t in trades if not t.get("partial")]
    result.avg_trade_duration_bars = sum(bars_held) / len(bars_held) if bars_held else 0

    # Equity curve and drawdown
    eq_series = pd.Series(equity_history)
    result.equity_curve = eq_series

    running_max = eq_series.cummax()
    drawdown = (eq_series - running_max) / running_max * 100
    result.max_drawdown_pct = drawdown.min()

    # Annualized return (assuming H1 bars, ~8760 bars/year)
    n_bars = len(equity_history)
    if n_bars > 0:
        total_return = equity / initial_equity
        years = n_bars / 8760.0
        result.annual_return_pct = (total_return ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe ratio (annualized, assuming hourly returns)
    if len(eq_series) > 1:
        returns = eq_series.pct_change().dropna()
        if returns.std() > 0:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760)

    return result


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print("          BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:          {result.total_return_pct:>10.2f}%")
    print(f"  Annualized Return:     {result.annual_return_pct:>10.2f}%")
    print(f"  Sharpe Ratio:          {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:          {result.max_drawdown_pct:>10.2f}%")
    print(f"  Win Rate:              {result.win_rate:>10.1f}%")
    print(f"  Total Trades:          {result.total_trades:>10d}")
    print(f"  Winning Trades:        {result.winning_trades:>10d}")
    print(f"  Losing Trades:         {result.losing_trades:>10d}")
    print(f"  Avg Win:               {result.avg_win_pct:>10.3f}%")
    print(f"  Avg Loss:              {result.avg_loss_pct:>10.3f}%")
    print(f"  Profit Factor:         {result.profit_factor:>10.2f}")
    print(f"  Avg Duration (bars):   {result.avg_trade_duration_bars:>10.1f}")
    print("=" * 60)
