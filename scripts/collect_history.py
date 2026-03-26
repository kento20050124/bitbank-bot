#!/usr/bin/env python3
"""Script to collect historical candle data from Bitbank."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bitbank_bot.config import load_config
from bitbank_bot.data.collector import collect_historical_candles
from bitbank_bot.data.store import DataStore
from bitbank_bot.exchange.client import BitbankClient


def main():
    parser = argparse.ArgumentParser(description="Collect historical candle data")
    parser.add_argument("--symbol", default=None, help="Trading pair (e.g. XRP/JPY)")
    parser.add_argument("--timeframes", nargs="+", default=None, help="Timeframes (e.g. 1h 4h)")
    parser.add_argument("--days", type=int, default=180, help="Days of history to fetch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    symbol = args.symbol or config.strategy.symbol
    timeframes = args.timeframes or [
        config.strategy.timeframe_entry,
        config.strategy.timeframe_trend,
    ]

    store = DataStore(config.db_path)
    client = BitbankClient(config.exchange)

    for tf in timeframes:
        logging.info("Collecting %s %s candles (%d days)...", symbol, tf, args.days)
        count = collect_historical_candles(
            client, store, symbol, tf, days_back=args.days
        )
        logging.info("Done: %d candles saved for %s %s", count, symbol, tf)

    store.close()
    logging.info("All collection complete.")


if __name__ == "__main__":
    main()
