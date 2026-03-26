"""Configuration loader for strategy parameters and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class ExchangeConfig:
    api_key: str
    api_secret: str
    enable_rate_limit: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "XRP/JPY"
    timeframe_entry: str = "1h"
    timeframe_trend: str = "4h"

    # EMA
    ema_fast_period: int = 20
    ema_slow_period: int = 50

    # ADX
    adx_period: int = 14
    adx_threshold: float = 20.0

    # ATR
    atr_period: int = 14

    # Chandelier Exit
    chandelier_multiplier: float = 2.8

    # Scaling Out
    scaling_rr_target: float = 2.0
    scaling_close_ratio: float = 0.5

    # Overbought Detection
    disparity_ema_period: int = 20
    disparity_threshold: float = 3.5
    rsi_period: int = 14
    rsi_overbought_entry: float = 75.0
    rsi_overbought_exit: float = 70.0

    # Position Sizing
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 10.0

    # Circuit Breaker
    max_concurrent_positions: int = 3
    max_daily_trades: int = 10
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 5

    # Order Execution
    maker_timeout_seconds: int = 180
    order_poll_interval: int = 5
    emergency_stop_slippage: float = 0.5

    # Fees
    maker_fee: float = -0.0002
    taker_fee: float = 0.0012


@dataclass(frozen=True)
class NotificationConfig:
    discord_webhook_url: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    exchange: ExchangeConfig
    strategy: StrategyConfig
    notification: NotificationConfig
    db_path: str = "data/candles.db"
    log_level: str = "INFO"


def load_config(env_path: str | None = None, strategy_path: str | None = None) -> AppConfig:
    """Load configuration from .env and strategy.yaml files."""
    # Load .env
    env_file = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    load_dotenv(env_file)

    # Load strategy YAML
    yaml_file = Path(strategy_path) if strategy_path else PROJECT_ROOT / "config" / "strategy.yaml"
    strategy_data = {}
    if yaml_file.exists():
        with open(yaml_file, "r") as f:
            strategy_data = yaml.safe_load(f) or {}

    # Build ExchangeConfig
    api_key = os.getenv("BITBANK_API_KEY", "")
    api_secret = os.getenv("BITBANK_API_SECRET", "")
    exchange_cfg = ExchangeConfig(api_key=api_key, api_secret=api_secret)

    # Build StrategyConfig from YAML
    timeframes = strategy_data.get("timeframes", {})
    strategy_cfg = StrategyConfig(
        symbol=strategy_data.get("symbol", "XRP/JPY"),
        timeframe_entry=timeframes.get("entry", "1h"),
        timeframe_trend=timeframes.get("trend", "4h"),
        ema_fast_period=strategy_data.get("ema_fast_period", 20),
        ema_slow_period=strategy_data.get("ema_slow_period", 50),
        adx_period=strategy_data.get("adx_period", 14),
        adx_threshold=strategy_data.get("adx_threshold", 20.0),
        atr_period=strategy_data.get("atr_period", 14),
        chandelier_multiplier=strategy_data.get("chandelier_multiplier", 2.8),
        scaling_rr_target=strategy_data.get("scaling_rr_target", 2.0),
        scaling_close_ratio=strategy_data.get("scaling_close_ratio", 0.5),
        disparity_ema_period=strategy_data.get("disparity_ema_period", 20),
        disparity_threshold=strategy_data.get("disparity_threshold", 3.5),
        rsi_period=strategy_data.get("rsi_period", 14),
        rsi_overbought_entry=strategy_data.get("rsi_overbought_entry", 75.0),
        rsi_overbought_exit=strategy_data.get("rsi_overbought_exit", 70.0),
        risk_per_trade_pct=strategy_data.get("risk_per_trade_pct", 1.0),
        max_position_pct=strategy_data.get("max_position_pct", 10.0),
        max_concurrent_positions=strategy_data.get("max_concurrent_positions", 3),
        max_daily_trades=strategy_data.get("max_daily_trades", 10),
        max_daily_loss_pct=strategy_data.get("max_daily_loss_pct", 3.0),
        max_consecutive_losses=strategy_data.get("max_consecutive_losses", 5),
        maker_timeout_seconds=strategy_data.get("maker_timeout_seconds", 180),
        order_poll_interval=strategy_data.get("order_poll_interval", 5),
        emergency_stop_slippage=strategy_data.get("emergency_stop_slippage", 0.5),
        maker_fee=strategy_data.get("maker_fee", -0.0002),
        taker_fee=strategy_data.get("taker_fee", 0.0012),
    )

    # Build NotificationConfig
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    notification_cfg = NotificationConfig(
        discord_webhook_url=webhook_url,
        enabled=bool(webhook_url),
    )

    return AppConfig(
        exchange=exchange_cfg,
        strategy=strategy_cfg,
        notification=notification_cfg,
        db_path=os.getenv("DB_PATH", "data/candles.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
