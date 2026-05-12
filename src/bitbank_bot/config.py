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

    # 確信度ベースの動的サイジング: score を 0-100 で受け取り、
    # risk% を [confidence_min_risk_pct, confidence_max_risk_pct] の範囲で線形補間。
    # confidence_score_min 未満のシグナルでも risk_per_trade_pct=base にフォールバック。
    confidence_min_risk_pct: float = 0.5
    confidence_max_risk_pct: float = 1.5
    confidence_score_min: float = 50.0
    confidence_score_max: float = 90.0

    # Breakeven Stop Move: 含み益が breakeven_trigger_r × ATR に達したら SL を建値に移動
    breakeven_trigger_r: float = 1.0

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

    # 想定スリッページ（指値がbest±0.2%で置かれ、即時約定しない場合の予備マージン）
    expected_slippage_pct: float = 0.002

    # Choppiness Indexによるレジームフィルタ
    chop_period: int = 14
    chop_max_threshold: float = 61.8  # これ超ならレンジ判定でエントリースキップ

    # 銘柄プライオリティ: 高価格(min_lot×price)が大きい銘柄は確信度ハードルを上げる
    expensive_min_notional_jpy: float = 5000.0  # min_lot * price がこれ以上なら "高価格" 扱い
    expensive_symbol_min_score: float = 60.0    # 高価格銘柄の最低スコア閾値

    # 期待値ゲート: 想定R(=stop_distance単位)あたり何JPY期待できるかが正でなければスキップ
    expected_value_min_r: float = 0.0  # 0=コスト相殺以上、ポジティブで上乗せ要求


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
        expected_slippage_pct=strategy_data.get("expected_slippage_pct", 0.002),
        chop_period=strategy_data.get("chop_period", 14),
        chop_max_threshold=strategy_data.get("chop_max_threshold", 61.8),
        expensive_min_notional_jpy=strategy_data.get("expensive_min_notional_jpy", 5000.0),
        expensive_symbol_min_score=strategy_data.get("expensive_symbol_min_score", 60.0),
        expected_value_min_r=strategy_data.get("expected_value_min_r", 0.0),
        confidence_min_risk_pct=strategy_data.get("confidence_min_risk_pct", 0.5),
        confidence_max_risk_pct=strategy_data.get("confidence_max_risk_pct", 1.5),
        confidence_score_min=strategy_data.get("confidence_score_min", 50.0),
        confidence_score_max=strategy_data.get("confidence_score_max", 90.0),
        breakeven_trigger_r=strategy_data.get("breakeven_trigger_r", 1.0),
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
