"""Exit signal generation: Chandelier Exit, Scaling Out, Overbought Detection."""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import ExitSignal, Position, PositionState, Side

logger = logging.getLogger(__name__)


class ExitRule(Protocol):
    """Protocol for exit rule implementations."""

    def check(
        self, position: Position, df: pd.DataFrame, cfg: StrategyConfig
    ) -> ExitSignal | None: ...


class ChandelierExit:
    """Trailing stop based on ATR from highest/lowest price since entry.

    For longs:  stop = highest_high - ATR * multiplier
    For shorts: stop = lowest_low + ATR * multiplier

    Stop only moves in favorable direction (never widens).
    """

    def check(
        self, position: Position, df: pd.DataFrame, cfg: StrategyConfig
    ) -> ExitSignal | None:
        if len(df) < cfg.atr_period + 1:
            return None

        last = df.iloc[-1]
        atr_col = f"atr_{cfg.atr_period}"
        atr = last.get(atr_col)
        if pd.isna(atr) or atr <= 0:
            return None

        current_price = last["close"]

        if position.side == Side.BUY:
            # Track highest price since entry
            new_highest = max(position.highest_price, last["high"])

            # Chandelier stop for longs
            chandelier_stop = new_highest - atr * cfg.chandelier_multiplier

            # Stop only moves up, never down
            effective_stop = max(chandelier_stop, position.stop_price)

            if current_price <= effective_stop:
                reasoning = (
                    f"CHANDELIER_EXIT_LONG: price={current_price:.4f} <= "
                    f"stop={effective_stop:.4f} "
                    f"(highest={new_highest:.4f} - ATR*{cfg.chandelier_multiplier}={atr * cfg.chandelier_multiplier:.4f})"
                )
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="chandelier",
                    exit_price=current_price,
                    close_ratio=1.0,
                    reasoning=reasoning,
                )

            # Update highest price (caller is responsible for persisting this)
            position.highest_price = new_highest
            position.stop_price = effective_stop

        elif position.side == Side.SELL:
            # Track lowest price since entry (use lowest_price field, not highest_price)
            current_lowest = position.lowest_price if position.lowest_price > 0 else last["low"]
            new_lowest = min(current_lowest, last["low"])

            chandelier_stop = new_lowest + atr * cfg.chandelier_multiplier
            # For shorts, stop only moves DOWN (tighter = lower), never up
            effective_stop = min(chandelier_stop, position.stop_price) if position.stop_price > 0 else chandelier_stop

            if current_price >= effective_stop:
                reasoning = (
                    f"CHANDELIER_EXIT_SHORT: price={current_price:.4f} >= "
                    f"stop={effective_stop:.4f} "
                    f"(lowest={new_lowest:.4f} + ATR*{cfg.chandelier_multiplier}={atr * cfg.chandelier_multiplier:.4f})"
                )
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="chandelier",
                    exit_price=current_price,
                    close_ratio=1.0,
                    reasoning=reasoning,
                )

            position.lowest_price = new_lowest
            position.stop_price = effective_stop

        return None


class ScalingOut:
    """Partial take-profit when price reaches RR target.

    Closes half the position and moves stop to breakeven.
    Only triggers once (when position is in OPEN state, not already SCALING/TRAILING).
    """

    def check(
        self, position: Position, df: pd.DataFrame, cfg: StrategyConfig
    ) -> ExitSignal | None:
        # Only trigger once: when position is still in OPEN state
        if position.state != PositionState.OPEN:
            return None

        if len(df) < cfg.atr_period + 1:
            return None

        last = df.iloc[-1]
        atr_col = f"atr_{cfg.atr_period}"
        atr = last.get(atr_col)
        if pd.isna(atr) or atr <= 0:
            return None

        current_price = last["close"]
        target_distance = atr * cfg.scaling_rr_target

        if position.side == Side.BUY:
            target_price = position.entry_price + target_distance
            if current_price >= target_price:
                reasoning = (
                    f"SCALING_OUT_LONG: price={current_price:.4f} >= "
                    f"target={target_price:.4f} "
                    f"(entry={position.entry_price:.4f} + ATR*{cfg.scaling_rr_target}={target_distance:.4f}). "
                    f"Closing {cfg.scaling_close_ratio*100:.0f}%, moving stop to breakeven."
                )
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="scaling",
                    exit_price=current_price,
                    close_ratio=cfg.scaling_close_ratio,
                    reasoning=reasoning,
                )

        elif position.side == Side.SELL:
            target_price = position.entry_price - target_distance
            if current_price <= target_price:
                reasoning = (
                    f"SCALING_OUT_SHORT: price={current_price:.4f} <= "
                    f"target={target_price:.4f}. "
                    f"Closing {cfg.scaling_close_ratio*100:.0f}%, moving stop to breakeven."
                )
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="scaling",
                    exit_price=current_price,
                    close_ratio=cfg.scaling_close_ratio,
                    reasoning=reasoning,
                )

        return None


class OverboughtExit:
    """Exit on overbought detection using Disparity Index + RSI.

    Triggers when:
    - Disparity Index > threshold (price far above EMA)
    - RSI was above entry threshold (75) and drops below exit threshold (70)
    """

    def check(
        self, position: Position, df: pd.DataFrame, cfg: StrategyConfig
    ) -> ExitSignal | None:
        if len(df) < max(cfg.rsi_period, cfg.disparity_ema_period) + 2:
            return None

        disp_col = f"disparity_{cfg.disparity_ema_period}"
        rsi_col = f"rsi_{cfg.rsi_period}"

        last = df.iloc[-1]
        prev = df.iloc[-2]

        disparity = last.get(disp_col)
        rsi_now = last.get(rsi_col)
        rsi_prev = prev.get(rsi_col)

        if any(pd.isna(v) for v in [disparity, rsi_now, rsi_prev]):
            return None

        current_price = last["close"]

        if position.side == Side.BUY:
            # Overbought for long positions
            disparity_triggered = disparity > cfg.disparity_threshold
            rsi_dropping = rsi_prev >= cfg.rsi_overbought_entry and rsi_now < cfg.rsi_overbought_exit

            if disparity_triggered or rsi_dropping:
                reasons = []
                if disparity_triggered:
                    reasons.append(f"disparity={disparity:.2f}% > {cfg.disparity_threshold}%")
                if rsi_dropping:
                    reasons.append(
                        f"RSI dropped {rsi_prev:.1f} -> {rsi_now:.1f} "
                        f"(crossed below {cfg.rsi_overbought_exit})"
                    )
                reasoning = f"OVERBOUGHT_EXIT_LONG: {', '.join(reasons)}, price={current_price:.4f}"
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="overbought",
                    exit_price=current_price,
                    close_ratio=1.0,
                    reasoning=reasoning,
                )

        elif position.side == Side.SELL:
            # Oversold for short positions: exit when price is recovering (bouncing up)
            disparity_triggered = disparity < -cfg.disparity_threshold
            # RSI was oversold (below 30) and now rising above 30 = recovery signal
            oversold_level = 100 - cfg.rsi_overbought_entry  # 25
            recovery_level = 100 - cfg.rsi_overbought_exit   # 30
            rsi_rising = rsi_prev <= recovery_level and rsi_now > recovery_level

            if disparity_triggered or rsi_rising:
                reasons = []
                if disparity_triggered:
                    reasons.append(f"disparity={disparity:.2f}% < -{cfg.disparity_threshold}%")
                if rsi_rising:
                    reasons.append(f"RSI rising {rsi_prev:.1f} -> {rsi_now:.1f}")
                reasoning = f"OVERSOLD_EXIT_SHORT: {', '.join(reasons)}, price={current_price:.4f}"
                logger.info(reasoning)
                return ExitSignal(
                    exit_type="overbought",
                    exit_price=current_price,
                    close_ratio=1.0,
                    reasoning=reasoning,
                )

        return None


# Default exit rules in priority order
EXIT_RULES: list[ExitRule] = [
    ScalingOut(),
    OverboughtExit(),
    ChandelierExit(),
]


def check_exit_conditions(
    position: Position,
    df: pd.DataFrame,
    cfg: StrategyConfig,
    rules: list[ExitRule] | None = None,
) -> ExitSignal | None:
    """Check all exit conditions. First rule to fire wins."""
    if rules is None:
        rules = EXIT_RULES

    for rule in rules:
        signal = rule.check(position, df, cfg)
        if signal is not None:
            return signal

    return None
