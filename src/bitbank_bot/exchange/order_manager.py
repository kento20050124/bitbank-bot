"""Order manager: Maker-first order placement with fill tracking."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.models import Order, OrderStatus, OrderType, Side
from bitbank_bot.data.store import DataStore
from bitbank_bot.exchange.client import BitbankClient

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order placement, fill polling, and emergency stops."""

    def __init__(self, client: BitbankClient, store: DataStore, cfg: StrategyConfig):
        self.client = client
        self.store = store
        self.cfg = cfg

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Order | None:
        """Place a limit (maker) order and save to store."""
        try:
            result = self.client.create_limit_order(symbol, side, amount, price)
            order = Order(
                order_id=str(result["id"]),
                symbol=symbol,
                side=Side(side),
                order_type=OrderType.LIMIT,
                amount=amount,
                price=price,
                filled=result.get("filled", 0),
                status=OrderStatus.OPEN,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.store.save_order(order)
            return order
        except Exception as e:
            logger.error("Failed to place limit order: %s", e)
            return None

    def place_market_order(self, symbol: str, side: str, amount: float) -> Order | None:
        """Place a market (taker) order for emergency exits."""
        try:
            result = self.client.create_market_order(symbol, side, amount)
            order = Order(
                order_id=str(result["id"]),
                symbol=symbol,
                side=Side(side),
                order_type=OrderType.MARKET,
                amount=amount,
                price=result.get("average") or result.get("price"),
                filled=result.get("filled", amount),
                status=OrderStatus.FILLED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.store.save_order(order)
            return order
        except Exception as e:
            logger.error("EMERGENCY market order failed: %s", e)
            return None

    def check_order_fill(self, order: Order) -> Order:
        """Poll exchange for order status update."""
        try:
            result = self.client.fetch_order(order.order_id, order.symbol)
            order.filled = result.get("filled", order.filled)
            order.updated_at = datetime.now()

            status = result.get("status", "")
            if status == "closed":
                order.status = OrderStatus.FILLED
            elif status == "canceled":
                order.status = OrderStatus.CANCELED
            elif order.filled > 0 and order.filled < order.amount:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.OPEN

            self.store.save_order(order)
        except Exception as e:
            logger.error("Failed to check order %s: %s", order.order_id, e)

        return order

    def cancel_order(self, order: Order) -> bool:
        """Cancel an open order."""
        try:
            self.client.cancel_order(order.order_id, order.symbol)
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now()
            self.store.save_order(order)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order.order_id, e)
            return False

    def wait_for_fill(
        self, order: Order, timeout_seconds: int | None = None
    ) -> Order:
        """Poll until the order is filled or timeout.

        Returns the updated order.
        """
        timeout = timeout_seconds or self.cfg.maker_timeout_seconds
        poll_interval = self.cfg.order_poll_interval
        elapsed = 0

        while elapsed < timeout:
            order = self.check_order_fill(order)

            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.FAILED):
                return order

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout: cancel unfilled order
        logger.warning(
            "Order %s timed out after %ds, canceling.", order.order_id, timeout
        )
        self.cancel_order(order)
        return order

    def place_maker_entry(
        self, symbol: str, side: str, amount: float
    ) -> Order | None:
        """Place an entry order using maker strategy.

        Places a limit order inside the spread to ensure maker status.
        """
        try:
            orderbook = self.client.fetch_order_book(symbol)
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            if not bids or not asks:
                logger.error("Empty order book for %s", symbol)
                return None

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid

            if side == "buy":
                # Place slightly above best bid (inside spread)
                price = best_bid + spread * 0.3
            else:
                # Place slightly below best ask (inside spread)
                price = best_ask - spread * 0.3

            # Round to appropriate precision
            market = self.client.get_market_info(symbol)
            precision = market.get("precision", {}).get("price", 2)
            price = round(price, precision)

            return self.place_limit_order(symbol, side, amount, price)

        except Exception as e:
            logger.error("Failed to place maker entry: %s", e)
            return None
