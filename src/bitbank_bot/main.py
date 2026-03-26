"""CLI entry point for the Bitbank auto-trading bot."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

from bitbank_bot.config import load_config


def setup_logging(level: str = "INFO"):
    """Configure logging with file and console handlers."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_dir / "bot.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        ),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_live(args):
    """Run the bot in live trading mode."""
    config = load_config()
    setup_logging(config.log_level)

    if not config.exchange.api_key or not config.exchange.api_secret:
        print("ERROR: BITBANK_API_KEY and BITBANK_API_SECRET must be set in .env")
        sys.exit(1)

    from bitbank_bot.engine.loop import TradingLoop

    loop = TradingLoop(config, paper_mode=False)
    loop.run()


def cmd_paper(args):
    """Run the bot in paper trading mode (simulated fills)."""
    config = load_config()
    setup_logging(config.log_level)

    from bitbank_bot.engine.loop import TradingLoop

    loop = TradingLoop(config, paper_mode=True)
    loop.run()


def cmd_backtest(args):
    """Run a backtest on historical data."""
    config = load_config()
    setup_logging(config.log_level)

    from bitbank_bot.backtest.runner import print_backtest_report, run_backtest
    from bitbank_bot.data.store import DataStore

    store = DataStore(config.db_path)
    df_h1 = store.get_candles_df(
        config.strategy.symbol, config.strategy.timeframe_entry, limit=10000
    )
    store.close()

    if df_h1.empty or len(df_h1) < 200:
        print(
            f"ERROR: Insufficient data. Have {len(df_h1)} candles, need at least 200.\n"
            "Run 'bitbank-bot collect' first to download historical data."
        )
        sys.exit(1)

    print(f"Running backtest on {len(df_h1)} candles...")
    result = run_backtest(
        df_h1,
        config.strategy,
        initial_equity=args.equity,
        use_maker_fee=not args.taker,
    )
    print_backtest_report(result)

    if args.optimize:
        from bitbank_bot.backtest.optimizer import optimize_parameters, print_optimization_report

        print("\nRunning parameter optimization...")
        results = optimize_parameters(df_h1, config.strategy, initial_equity=args.equity)
        print_optimization_report(results)


def cmd_collect(args):
    """Collect historical candle data."""
    config = load_config()
    setup_logging(config.log_level)

    from bitbank_bot.data.collector import collect_historical_candles
    from bitbank_bot.data.store import DataStore
    from bitbank_bot.exchange.client import BitbankClient

    store = DataStore(config.db_path)
    client = BitbankClient(config.exchange)

    symbol = args.symbol or config.strategy.symbol
    timeframes = args.timeframes or [
        config.strategy.timeframe_entry,
        config.strategy.timeframe_trend,
    ]

    for tf in timeframes:
        print(f"Collecting {symbol} {tf} candles ({args.days} days)...")
        count = collect_historical_candles(
            client, store, symbol, tf, days_back=args.days
        )
        print(f"  -> {count} candles saved")

    store.close()
    print("Collection complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Bitbank Auto-Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bitbank-bot collect --days 365          # Download 1 year of history
  bitbank-bot backtest --optimize         # Run backtest with optimization
  bitbank-bot paper                       # Start paper trading
  bitbank-bot live                        # Start live trading
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect historical data")
    p_collect.add_argument("--symbol", default=None)
    p_collect.add_argument("--timeframes", nargs="+", default=None)
    p_collect.add_argument("--days", type=int, default=180)
    p_collect.set_defaults(func=cmd_collect)

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Run backtest")
    p_backtest.add_argument("--equity", type=float, default=1_000_000)
    p_backtest.add_argument("--taker", action="store_true", help="Use taker fees")
    p_backtest.add_argument("--optimize", action="store_true", help="Run optimization")
    p_backtest.set_defaults(func=cmd_backtest)

    # paper
    p_paper = subparsers.add_parser("paper", help="Paper trading mode")
    p_paper.set_defaults(func=cmd_paper)

    # live
    p_live = subparsers.add_parser("live", help="Live trading mode")
    p_live.set_defaults(func=cmd_live)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
