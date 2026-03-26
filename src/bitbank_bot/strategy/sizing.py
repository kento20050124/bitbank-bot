"""Position sizing based on ATR and risk percentage."""

from __future__ import annotations

import logging

from bitbank_bot.config import StrategyConfig

logger = logging.getLogger(__name__)


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_distance: float,
    cfg: StrategyConfig,
    min_order_size: float = 0.0001,
) -> float:
    """Calculate position size based on risk per trade.

    Uses the formula:
        position_size = (equity * risk_pct) / stop_distance

    Also enforces max_position_pct limit.

    Args:
        equity: Total account equity in quote currency (JPY).
        entry_price: Expected entry price.
        stop_distance: Distance from entry to stop in price units.
        cfg: Strategy configuration.
        min_order_size: Minimum order size for the asset.

    Returns:
        Position size in base currency units.
    """
    if stop_distance <= 0 or entry_price <= 0 or equity <= 0:
        logger.warning(
            "Invalid inputs for sizing: equity=%.2f, entry=%.4f, stop_dist=%.4f",
            equity,
            entry_price,
            stop_distance,
        )
        return 0.0

    # Risk amount in JPY
    risk_amount = equity * (cfg.risk_per_trade_pct / 100.0)

    # Position size = risk / stop distance (in base currency)
    position_size = risk_amount / stop_distance

    # Max position limit
    max_position_value = equity * (cfg.max_position_pct / 100.0)
    max_position_size = max_position_value / entry_price

    if position_size > max_position_size:
        logger.info(
            "Position size %.8f exceeds max %.8f, capping.",
            position_size,
            max_position_size,
        )
        position_size = max_position_size

    # Enforce minimum
    if position_size < min_order_size:
        logger.warning(
            "Calculated size %.8f below minimum %.8f, skipping.",
            position_size,
            min_order_size,
        )
        return 0.0

    logger.info(
        "Position sizing: equity=%.0f JPY, risk=%.0f JPY (%.1f%%), "
        "stop_dist=%.4f, size=%.8f",
        equity,
        risk_amount,
        cfg.risk_per_trade_pct,
        stop_distance,
        position_size,
    )

    return position_size
