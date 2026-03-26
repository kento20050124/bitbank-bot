"""Parameter optimization for the trading strategy via grid search."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, replace

import pandas as pd

from bitbank_bot.backtest.runner import BacktestResult, run_backtest
from bitbank_bot.config import StrategyConfig

logger = logging.getLogger(__name__)


def optimize_parameters(
    df_1h: pd.DataFrame,
    base_cfg: StrategyConfig,
    param_grid: dict[str, list] | None = None,
    initial_equity: float = 1_000_000.0,
    metric: str = "sharpe_ratio",
) -> list[dict]:
    """Run grid search over strategy parameters.

    Args:
        df_1h: H1 OHLCV DataFrame.
        base_cfg: Base strategy configuration.
        param_grid: Dict of parameter name -> list of values to test.
                    Defaults to a reasonable grid if None.
        initial_equity: Starting equity.
        metric: Metric to optimize ("sharpe_ratio", "total_return_pct",
                "profit_factor", "win_rate").

    Returns:
        List of dicts with parameters and results, sorted by metric descending.
    """
    if param_grid is None:
        param_grid = {
            "chandelier_multiplier": [2.0, 2.5, 2.8, 3.0, 3.5],
            "scaling_rr_target": [1.5, 2.0, 2.5, 3.0],
            "adx_threshold": [15, 20, 25],
            "ema_fast_period": [10, 20, 30],
        }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    logger.info("Running grid search: %d combinations", total)
    results = []

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Create config with overridden parameters
        cfg = replace(base_cfg, **params)

        try:
            bt_result = run_backtest(df_1h, cfg, initial_equity=initial_equity)
        except Exception as e:
            logger.warning("Backtest failed for params %s: %s", params, e)
            continue

        entry = {
            **params,
            "total_return_pct": bt_result.total_return_pct,
            "annual_return_pct": bt_result.annual_return_pct,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown_pct": bt_result.max_drawdown_pct,
            "win_rate": bt_result.win_rate,
            "total_trades": bt_result.total_trades,
            "profit_factor": bt_result.profit_factor,
        }
        results.append(entry)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            logger.info("Progress: %d/%d", idx + 1, total)

    # Sort by target metric
    results.sort(key=lambda x: x.get(metric, 0), reverse=True)

    return results


def print_optimization_report(results: list[dict], top_n: int = 10):
    """Print top N parameter combinations."""
    print("\n" + "=" * 80)
    print("          OPTIMIZATION RESULTS (Top %d)" % top_n)
    print("=" * 80)

    headers = list(results[0].keys()) if results else []
    param_keys = [k for k in headers if k not in {
        "total_return_pct", "annual_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate", "total_trades", "profit_factor",
    }]

    for i, entry in enumerate(results[:top_n]):
        print(f"\n--- Rank #{i+1} ---")
        for k in param_keys:
            print(f"  {k}: {entry[k]}")
        print(f"  Sharpe:     {entry['sharpe_ratio']:.2f}")
        print(f"  Return:     {entry['total_return_pct']:.2f}%")
        print(f"  Drawdown:   {entry['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:   {entry['win_rate']:.1f}%")
        print(f"  PF:         {entry['profit_factor']:.2f}")
        print(f"  Trades:     {entry['total_trades']}")

    print("\n" + "=" * 80)
