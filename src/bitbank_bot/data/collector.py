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
    # bitbankは日付別エンドポイント（/candlestick/{tf}/{YYYYMMDD}）で、
    # 当該日にデータが無い・未公開だと空配列を返す。連続X日空でも諦めず1日ずつ進める。
    consecutive_empty_days = 0
    max_empty_days = 7
    one_day_ms = 86_400_000

    while since_ms < now_ms:
        try:
            ohlcv = client.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=batch_size
            )
        except Exception as e:
            logger.error("Failed to fetch candles: %s", e)
            break

        if not ohlcv:
            # 空でも次の日にskipして続行（bitbankの日別エンドポイント特性に対応）
            consecutive_empty_days += 1
            since_ms += one_day_ms
            if consecutive_empty_days >= max_empty_days:
                logger.info(
                    "%d consecutive empty days, stopping collection for %s.",
                    max_empty_days,
                    symbol,
                )
                break
            time.sleep(0.2)
            continue

        consecutive_empty_days = 0

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

        # 次のバッチ。返却が1件しか無い場合に無限ループを避けるため最低1tf進める。
        last_ts = ohlcv[-1][0]
        next_since = last_ts + tf_ms
        if next_since <= since_ms:
            next_since = since_ms + tf_ms
        since_ms = next_since

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
    lookback_days: int = 2,
):
    """Fetch the most recent candles and update the store.

    bitbankは日別エンドポイントのため、`since=None`では当日分のみ返ることが多い。
    日跨ぎでの取りこぼしを避けるため、直近 `lookback_days` 日分を `since` 指定で取得する。
    保存時は PRIMARY KEY (symbol, timeframe, timestamp) で重複は OR REPLACE される。
    """
    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
    one_day_ms = 86_400_000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (lookback_days * one_day_ms)
    total_saved = 0

    cursor = since_ms
    consecutive_empty = 0
    # 日跨ぎ対応: 最大 lookback_days+1 回ループ（1日1コール想定）
    max_loops = lookback_days + 2
    for _ in range(max_loops):
        if cursor >= now_ms:
            break
        try:
            ohlcv = client.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=limit
            )
        except Exception as e:
            logger.error("Failed to fetch latest candles for %s: %s", symbol, e)
            return total_saved

        if not ohlcv:
            consecutive_empty += 1
            cursor += one_day_ms
            if consecutive_empty >= 2:
                break
            continue

        consecutive_empty = 0
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

        last_ts = ohlcv[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            next_cursor = cursor + one_day_ms
        cursor = next_cursor

    return total_saved
