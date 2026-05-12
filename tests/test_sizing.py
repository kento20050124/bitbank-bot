"""Tests for confidence-based position sizing."""

from bitbank_bot.config import StrategyConfig
from bitbank_bot.strategy.sizing import _resolve_risk_pct, calculate_position_size


def test_resolve_risk_pct_no_score_returns_base():
    cfg = StrategyConfig(risk_per_trade_pct=1.0)
    assert _resolve_risk_pct(cfg, None) == 1.0


def test_resolve_risk_pct_below_min_returns_base():
    cfg = StrategyConfig(
        risk_per_trade_pct=1.0,
        confidence_min_risk_pct=0.5,
        confidence_max_risk_pct=1.5,
        confidence_score_min=50.0,
        confidence_score_max=90.0,
    )
    assert _resolve_risk_pct(cfg, 30.0) == 1.0


def test_resolve_risk_pct_linear_interpolation():
    cfg = StrategyConfig(
        confidence_min_risk_pct=0.5,
        confidence_max_risk_pct=1.5,
        confidence_score_min=50.0,
        confidence_score_max=90.0,
    )
    # score=50 -> 0.5%
    assert abs(_resolve_risk_pct(cfg, 50.0) - 0.5) < 1e-6
    # score=70 -> 1.0% (中点)
    assert abs(_resolve_risk_pct(cfg, 70.0) - 1.0) < 1e-6
    # score=90 -> 1.5%
    assert abs(_resolve_risk_pct(cfg, 90.0) - 1.5) < 1e-6
    # score=100 (max超過) -> max でクリップ
    assert abs(_resolve_risk_pct(cfg, 100.0) - 1.5) < 1e-6


def test_calculate_position_size_uses_score():
    cfg = StrategyConfig(
        risk_per_trade_pct=1.0,
        max_position_pct=100.0,
        confidence_min_risk_pct=0.5,
        confidence_max_risk_pct=1.5,
        confidence_score_min=50.0,
        confidence_score_max=90.0,
    )
    # 高スコア = 大きめサイズ
    size_high = calculate_position_size(
        equity=100_000, entry_price=100.0, stop_distance=2.0, cfg=cfg,
        min_order_size=0.0001, confidence_score=90.0
    )
    # 低スコア(<min) = ベースリスク
    size_base = calculate_position_size(
        equity=100_000, entry_price=100.0, stop_distance=2.0, cfg=cfg,
        min_order_size=0.0001, confidence_score=30.0
    )
    # 中スコア
    size_mid = calculate_position_size(
        equity=100_000, entry_price=100.0, stop_distance=2.0, cfg=cfg,
        min_order_size=0.0001, confidence_score=70.0
    )
    # 90% は base(70%相当=1.0%)より大きい
    assert size_high > size_base
    assert size_high > size_mid
    # base(30%, below min) は 1.0% 相当 = mid(70%) と同じ
    assert abs(size_base - size_mid) < 1e-6
