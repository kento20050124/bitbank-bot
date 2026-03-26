"""Exchange client wrapper around ccxt for Bitbank."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any

import ccxt

from bitbank_bot.config import ExchangeConfig

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for exponential backoff retry on transient exchange errors."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    ccxt.NetworkError,
                    ccxt.ExchangeNotAvailable,
                    ccxt.RequestTimeout,
                ) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        "Transient error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                except ccxt.RateLimitExceeded as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning("Rate limit hit: %s. Waiting %.1fs", e, delay)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

        return wrapper

    return decorator


class BitbankClient:
    """Wrapper around ccxt.bitbank with rate limiting and error handling."""

    def __init__(self, config: ExchangeConfig):
        self.exchange = ccxt.bitbank(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": config.enable_rate_limit,
                "options": {"defaultType": "spot"},
            }
        )
        self._last_update_time = 0.0
        self._update_interval = 1.0 / 6.0  # 6 updates/sec max

    def _throttle_update(self):
        """Ensure update requests (orders) respect 6/sec limit."""
        now = time.time()
        elapsed = now - self._last_update_time
        if elapsed < self._update_interval:
            time.sleep(self._update_interval - elapsed)
        self._last_update_time = time.time()

    @retry_on_error()
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list]:
        """Fetch OHLCV candle data.

        Returns list of [timestamp_ms, open, high, low, close, volume].
        """
        return self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit
        )

    @retry_on_error()
    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data."""
        return self.exchange.fetch_ticker(symbol)

    @retry_on_error()
    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Fetch order book."""
        return self.exchange.fetch_order_book(symbol, limit=limit)

    @retry_on_error()
    def fetch_balance(self) -> dict:
        """Fetch account balance."""
        self._throttle_update()
        return self.exchange.fetch_balance()

    @retry_on_error()
    def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> dict:
        """Place a limit order (maker). Returns order info dict."""
        self._throttle_update()
        logger.info(
            "Placing LIMIT %s order: %s %.8f @ %.4f",
            side.upper(),
            symbol,
            amount,
            price,
        )
        return self.exchange.create_limit_order(symbol, side, amount, price)

    @retry_on_error()
    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Place a market order (taker). Use only for emergency stops."""
        self._throttle_update()
        logger.warning(
            "Placing MARKET %s order (emergency): %s %.8f",
            side.upper(),
            symbol,
            amount,
        )
        return self.exchange.create_market_order(symbol, side, amount)

    @retry_on_error()
    def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Fetch order status."""
        return self.exchange.fetch_order(order_id, symbol)

    @retry_on_error()
    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an order."""
        self._throttle_update()
        logger.info("Canceling order %s for %s", order_id, symbol)
        return self.exchange.cancel_order(order_id, symbol)

    @retry_on_error()
    def fetch_my_trades(
        self, symbol: str, since: int | None = None, limit: int = 50
    ) -> list[dict]:
        """Fetch recent trades for the authenticated user."""
        return self.exchange.fetch_my_trades(symbol, since=since, limit=limit)

    def get_market_info(self, symbol: str) -> dict:
        """Get market info (min order size, price precision, etc.)."""
        self.exchange.load_markets()
        return self.exchange.markets.get(symbol, {})
