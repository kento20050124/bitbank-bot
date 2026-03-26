"""Historical candle data collector for Bitbank."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from bitbank_bot.data.models import Candle
from bitbank_bot.data.store import DataStore
from bitbank_bot.exchange.client import BitbankClient

logger = logging.getLogger(__name__)

# Timeframe to milliseconds mapping
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def collect_historical_candles(
    client: BitbankClient,
    store: DataStore,
    symbol: str,
    timeframe: str,
    days_back: int = 180,
    batch_size: int = 500,
):
    """Download and store historical candle data.

    Fetches candles from `days_back` days ago up to now, in batches.
    Skips data that is already in the database.
    """
    tf_ms = TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Determine start time
    latest = store.get_latest_candle_time(symbol, timeframe)
    if latest:
        since_dt = latest + timedelta(milliseconds=tf_ms)
        logger.info(
            "Resuming from %s for %s %s", since_dt.isoformat(), symbol, timeframe
        )
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
        logger.info(
            "Starting fresh collection from %s for %s %s",
            since_dt.isoformat(),
            symbol,
            timeframe,
        )

    since_ms = int(since_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total_saved = 0

    while since_ms < now_ms:
        try:
            ohlcv = client.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=batch_size
            )
        except Exception as e:
            logger.error("Failed to fetch candles: %s", e)
            break

        if not ohlcv:
            logger.info("No more candle data available.")
            break

        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in ohlcv
        ]

        store.save_candles(candles)
        total_saved += len(candles)

        # Move to next batch
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + tf_ms

        logger.info(
            "Saved %d candles (total: %d), latest: %s",
            len(candles),
            total_saved,
            datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat(),
        )

        # Respect rate limits
        time.sleep(0.2)

    logger.info(
        "Collection complete for %s %s: %d candles saved", symbol, timeframe, total_saved
    )
    return total_saved


def fetch_latest_candles(
    client: BitbankClient,
    store: DataStore,
    symbol: str,
    timeframe: str,
    limit: int = 10,
):
    """Fetch the most recent candles and update the store.

    Used during live trading to keep the database up to date.
    """
    try:
        ohlcv = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.error("Failed to fetch latest candles: %s", e)
        return 0

    if not ohlcv:
        return 0

    candles = [
        Candle(
            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in ohlcv
    ]

    store.save_candles(candles)
    return len(candles)
