"""Position sizing based on ATR and risk percentage."""

from __future__ import annotations

import logging

from bitbank_bot.config import StrategyConfig

logger = logging.getLogger(__name__)


def _resolve_risk_pct(cfg: StrategyConfig, confidence_score: float | None) -> float:
    """確信度スコアからリスク%を決める。

    score が None もしくは confidence_score_min 未満なら base (risk_per_trade_pct)。
    confidence_score_min..max の範囲で confidence_min_risk_pct..max_risk_pct に線形補間。
    """
    if confidence_score is None:
        return cfg.risk_per_trade_pct
    s_min = cfg.confidence_score_min
    s_max = max(cfg.confidence_score_max, s_min + 1e-6)
    if confidence_score < s_min:
        return cfg.risk_per_trade_pct
    clipped = min(confidence_score, s_max)
    t = (clipped - s_min) / (s_max - s_min)
    return cfg.confidence_min_risk_pct + t * (cfg.confidence_max_risk_pct - cfg.confidence_min_risk_pct)


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_distance: float,
    cfg: StrategyConfig,
    min_order_size: float = 0.0001,
    confidence_score: float | None = None,
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
        confidence_score: シグナルの確信度(0-100)。None なら従来挙動 (risk_per_trade_pct)。

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

    # 確信度ベースで risk% を決定
    effective_risk_pct = _resolve_risk_pct(cfg, confidence_score)
    risk_amount = equity * (effective_risk_pct / 100.0)

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
        "Position sizing: equity=%.0f JPY, risk=%.0f JPY (%.2f%%, score=%s), "
        "stop_dist=%.4f, size=%.8f",
        equity,
        risk_amount,
        effective_risk_pct,
        f"{confidence_score:.1f}" if confidence_score is not None else "n/a",
        stop_distance,
        position_size,
    )

    return position_size
