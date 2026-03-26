"""Main trading loop: orchestrates the entire trading cycle."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime

from bitbank_bot.config import AppConfig
from bitbank_bot.data.collector import fetch_latest_candles
from bitbank_bot.data.models import (
    ExitSignal,
    OrderStatus,
    Position,
    PositionState,
    Side,
    TradeLog,
)
from bitbank_bot.data.store import DataStore
from bitbank_bot.engine.circuit_breaker import CircuitBreaker
from bitbank_bot.engine.state_machine import transition
from bitbank_bot.exchange.client import BitbankClient
from bitbank_bot.exchange.order_manager import OrderManager
from bitbank_bot.notifications.discord import DiscordNotifier
from bitbank_bot.strategy.entry import generate_entry_signal
from bitbank_bot.strategy.exit import check_exit_conditions
from bitbank_bot.strategy.indicators import compute_all_indicators
from bitbank_bot.strategy.sizing import calculate_position_size

logger = logging.getLogger(__name__)


class TradingLoop:
    """Main trading loop that runs continuously."""

    def __init__(self, config: AppConfig, paper_mode: bool = False):
        self.config = config
        self.cfg = config.strategy
        self.paper_mode = paper_mode
        self._running = False

        # Initialize components
        self.store = DataStore(config.db_path)
        self.client = BitbankClient(config.exchange)
        self.order_mgr = OrderManager(self.client, self.store, self.cfg)
        self.circuit_breaker = CircuitBreaker(self.store, self.cfg)
        self.notifier = DiscordNotifier(config.notification)

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        self._running = False

    def run(self):
        """Start the main trading loop."""
        self._running = True
        symbol = self.cfg.symbol

        self.notifier.send_message(
            f"Bot started {'(PAPER MODE)' if self.paper_mode else '(LIVE)'} "
            f"for {symbol}"
        )
        logger.info("Trading loop started for %s (paper=%s)", symbol, self.paper_mode)

        while self._running:
            try:
                self._tick(symbol)
            except Exception as e:
                logger.error("Error in trading loop: %s", e, exc_info=True)
                self.notifier.send_error(f"Loop error: {e}")
                time.sleep(30)
                continue

            # Sleep until next check (aligned to minutes)
            self._sleep_until_next_check()

        self._cleanup(symbol)

    def _tick(self, symbol: str):
        """Execute one cycle of the trading loop."""
        # 1. Fetch latest H1 candle data (H4 is resampled from H1)
        fetch_latest_candles(
            self.client, self.store, symbol, self.cfg.timeframe_entry, limit=10
        )

        # 2. Load H1 candles and resample to H4
        df_h1 = self.store.get_candles_df(symbol, self.cfg.timeframe_entry, limit=800)

        if df_h1.empty or len(df_h1) < 200:
            logger.warning("Insufficient candle data (%d bars), skipping tick.", len(df_h1))
            return

        # Resample H1 -> H4 (Bitbank doesn't support 4h candles directly)
        df_h4 = df_h1.copy().set_index("timestamp")
        df_h4 = df_h4.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna().reset_index()

        df_h1 = compute_all_indicators(
            df_h1,
            ema_fast=self.cfg.ema_fast_period,
            ema_slow=self.cfg.ema_slow_period,
            adx_period=self.cfg.adx_period,
            atr_period=self.cfg.atr_period,
            rsi_period=self.cfg.rsi_period,
            disparity_ema_period=self.cfg.disparity_ema_period,
        )
        df_h4 = compute_all_indicators(
            df_h4,
            ema_fast=self.cfg.ema_fast_period,
            ema_slow=self.cfg.ema_slow_period,
            adx_period=self.cfg.adx_period,
            atr_period=self.cfg.atr_period,
            rsi_period=self.cfg.rsi_period,
            disparity_ema_period=self.cfg.disparity_ema_period,
        )

        # 3. Check pending orders
        self._check_pending_orders(symbol)

        # 4. Check exits on open positions
        open_positions = self.store.get_open_positions(symbol)
        for pos in open_positions:
            self._check_position_exit(pos, df_h1)

        # 5. Check for new entry
        if not self.circuit_breaker.is_halted:
            equity = self._get_equity()
            if self.circuit_breaker.check_can_trade(symbol, equity):
                self._check_entry(symbol, df_h1, df_h4, equity)

    def _check_position_exit(self, position: Position, df):
        """Check exit conditions for an open position."""
        exit_signal = check_exit_conditions(position, df, self.cfg)
        if exit_signal is None:
            # Update highest price tracking
            self.store.save_position(position)
            return

        self._execute_exit(position, exit_signal)

    def _execute_exit(self, position: Position, exit_signal: ExitSignal):
        """Execute an exit based on the signal."""
        close_amount = position.current_amount * exit_signal.close_ratio
        exit_side = "sell" if position.side == Side.BUY else "buy"

        if self.paper_mode:
            # Simulate fill
            logger.info("PAPER EXIT: %s %.8f @ %.4f (%s)",
                        exit_side, close_amount, exit_signal.exit_price,
                        exit_signal.exit_type)
        else:
            if exit_signal.exit_type == "emergency":
                order = self.order_mgr.place_market_order(
                    position.symbol, exit_side, close_amount
                )
            else:
                order = self.order_mgr.place_limit_order(
                    position.symbol, exit_side, close_amount, exit_signal.exit_price
                )

            if order:
                position.exit_order_id = order.order_id

        # Update position state
        if exit_signal.exit_type == "scaling":
            position.current_amount -= close_amount
            position.stop_price = position.entry_price  # Move to breakeven
            transition(position, PositionState.SCALING_OUT, exit_signal.reasoning)
            transition(position, PositionState.TRAILING, "Trailing after scaling out")
        else:
            # Calculate PnL
            if position.side == Side.BUY:
                pnl = (exit_signal.exit_price - position.entry_price) * close_amount
            else:
                pnl = (position.entry_price - exit_signal.exit_price) * close_amount

            position.realized_pnl += pnl
            position.current_amount -= close_amount

            if position.current_amount <= 0:
                transition(position, PositionState.PENDING_EXIT, exit_signal.reasoning)
                transition(position, PositionState.CLOSED, f"PnL: {pnl:.2f}")

        # Save position BEFORE updating circuit breaker (consistency on crash)
        self.store.save_position(position)

        if position.state == PositionState.CLOSED:
            self.circuit_breaker.record_trade_result(position.realized_pnl)
            # Update paper equity
            if self.paper_mode:
                equity = float(self.store.get_state("paper_equity", "1000000"))
                equity += position.realized_pnl
                self.store.set_state("paper_equity", str(equity))

        # Log and notify
        self.store.log_trade(TradeLog(
            timestamp=datetime.now(),
            action="exit",
            symbol=position.symbol,
            side=exit_side,
            amount=close_amount,
            price=exit_signal.exit_price,
            reasoning=exit_signal.reasoning,
            pnl=position.realized_pnl if position.state == PositionState.CLOSED else None,
        ))

        self.notifier.send_trade(
            action="EXIT",
            symbol=position.symbol,
            side=exit_side,
            amount=close_amount,
            price=exit_signal.exit_price,
            reason=exit_signal.reasoning,
            pnl=position.realized_pnl if position.state == PositionState.CLOSED else None,
        )

    def _check_entry(self, symbol: str, df_h1, df_h4, equity: float):
        """Check for entry signal and execute if found."""
        entry_signal = generate_entry_signal(df_h1, df_h4, self.cfg)
        if entry_signal is None:
            return

        # Get minimum order size
        market_info = self.client.get_market_info(symbol)
        min_size = market_info.get("limits", {}).get("amount", {}).get("min", 0.0001)

        # Calculate position size
        pos_size = calculate_position_size(
            equity=equity,
            entry_price=entry_signal.entry_price,
            stop_distance=entry_signal.stop_distance,
            cfg=self.cfg,
            min_order_size=min_size,
        )

        if pos_size <= 0:
            return

        side = "buy" if entry_signal.direction.value == "long" else "sell"

        # Create position record
        position = Position(
            symbol=symbol,
            side=Side(side),
            entry_price=entry_signal.entry_price,
            amount=pos_size,
            current_amount=pos_size,
            highest_price=entry_signal.entry_price,
            state=PositionState.FLAT,
            reasoning=entry_signal.reasoning,
        )

        if entry_signal.direction.value == "long":
            position.stop_price = entry_signal.entry_price - entry_signal.stop_distance
            position.target_price = entry_signal.entry_price + entry_signal.atr_value * self.cfg.scaling_rr_target
            position.highest_price = entry_signal.entry_price
        else:
            position.stop_price = entry_signal.entry_price + entry_signal.stop_distance
            position.target_price = entry_signal.entry_price - entry_signal.atr_value * self.cfg.scaling_rr_target
            position.lowest_price = entry_signal.entry_price

        transition(position, PositionState.PENDING_ENTRY, entry_signal.reasoning)

        if self.paper_mode:
            logger.info("PAPER ENTRY: %s %.8f @ %.4f", side, pos_size, entry_signal.entry_price)
            transition(position, PositionState.OPEN, "Paper fill")
        else:
            order = self.order_mgr.place_maker_entry(symbol, side, pos_size)
            if order:
                position.entry_order_id = order.order_id
            else:
                logger.error("Failed to place entry order")
                return

        pos_id = self.store.save_position(position)
        position.id = pos_id

        self.store.log_trade(TradeLog(
            timestamp=datetime.now(),
            action="entry",
            symbol=symbol,
            side=side,
            amount=pos_size,
            price=entry_signal.entry_price,
            reasoning=entry_signal.reasoning,
        ))

        self.notifier.send_trade(
            action="ENTRY",
            symbol=symbol,
            side=side,
            amount=pos_size,
            price=entry_signal.entry_price,
            reason=entry_signal.reasoning,
        )

    def _check_pending_orders(self, symbol: str):
        """Check and update status of pending orders."""
        pending = self.store.get_pending_orders(symbol)
        for order in pending:
            updated = self.order_mgr.check_order_fill(order)
            if updated.status == OrderStatus.FILLED:
                logger.info("Order %s filled", updated.order_id)
                # Find associated position and update state
                positions = self.store.get_open_positions(symbol)
                for pos in positions:
                    if pos.entry_order_id == updated.order_id and pos.state == PositionState.PENDING_ENTRY:
                        transition(pos, PositionState.OPEN, f"Order {updated.order_id} filled")
                        pos.entry_price = updated.price or pos.entry_price
                        self.store.save_position(pos)

    def _get_equity(self) -> float:
        """Get current account equity in JPY."""
        if self.paper_mode:
            equity_str = self.store.get_state("paper_equity", "1000000")
            return float(equity_str)
        try:
            balance = self.client.fetch_balance()
            jpy = balance.get("JPY", {}).get("total", 0)
            return float(jpy) if jpy else 0.0
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)
            return 0.0

    def _sleep_until_next_check(self):
        """Sleep until the next minute mark."""
        now = time.time()
        seconds_until_next_minute = 60 - (now % 60)
        sleep_time = max(5, seconds_until_next_minute + 5)  # 5s after minute mark
        logger.debug("Sleeping %.1fs until next check", sleep_time)
        # Sleep in small chunks for responsive shutdown
        for _ in range(int(sleep_time)):
            if not self._running:
                break
            time.sleep(1)

    def _cleanup(self, symbol: str):
        """Clean up on shutdown: cancel pending orders, save state."""
        logger.info("Cleaning up...")

        if not self.paper_mode:
            pending = self.store.get_pending_orders(symbol)
            for order in pending:
                self.order_mgr.cancel_order(order)
                logger.info("Canceled pending order %s on shutdown", order.order_id)

        self.store.close()
        self.notifier.send_message("Bot stopped. All pending orders canceled.")
        logger.info("Cleanup complete. Bot stopped.")
