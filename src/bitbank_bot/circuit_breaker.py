"""Circuit breaker: safety limits to prevent runaway losses."""

from __future__ import annotations

import logging
from datetime import datetime

from bitbank_bot.config import StrategyConfig
from bitbank_bot.data.store import DataStore

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Enforces trading safety limits. State is persisted via DataStore."""

    def __init__(self, store: DataStore, cfg: StrategyConfig):
        self.store = store
        self.cfg = cfg
        # Restore persisted state
        self._consecutive_losses = int(store.get_state("consecutive_losses", "0"))
        self._halted = store.get_state("circuit_halted", "0") == "1"
        self._halt_reason = store.get_state("circuit_halt_reason", "")

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def reset(self):
        self._halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        self._persist()

    def record_trade_result(self, pnl: float):
        """Record a trade result and update consecutive loss counter."""
        if pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self.store.set_state("consecutive_losses", str(self._consecutive_losses))

    def check_can_trade(self, symbol: str, equity: float) -> bool:
        """Check if trading is allowed. Returns False with reason if halted."""
        if self._halted:
            return False

        # Check max daily trades (count only entries, not exits)
        daily_count = self.store.get_daily_trade_count(symbol)
        if daily_count >= self.cfg.max_daily_trades:
            self._halt("MAX_DAILY_TRADES: %d entries today (limit %d)" % (
                daily_count, self.cfg.max_daily_trades
            ))
            return False

        # Check max daily loss
        daily_pnl = self.store.get_daily_pnl(symbol)
        if equity > 0:
            daily_loss_pct = abs(daily_pnl) / equity * 100 if daily_pnl < 0 else 0
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                self._halt("MAX_DAILY_LOSS: %.2f%% loss today (limit %.1f%%)" % (
                    daily_loss_pct, self.cfg.max_daily_loss_pct
                ))
                return False

        # Check consecutive losses
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            self._halt("CONSECUTIVE_LOSSES: %d in a row (limit %d)" % (
                self._consecutive_losses, self.cfg.max_consecutive_losses
            ))
            return False

        # Check max concurrent positions
        open_positions = self.store.get_open_positions(symbol)
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            logger.debug(
                "Max concurrent positions reached: %d/%d",
                len(open_positions),
                self.cfg.max_concurrent_positions,
            )
            return False

        return True

    def _halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        self._persist()
        logger.warning("CIRCUIT BREAKER TRIGGERED: %s", reason)

    def _persist(self):
        """Persist circuit breaker state to DB."""
        self.store.set_state("circuit_halted", "1" if self._halted else "0")
        self.store.set_state("circuit_halt_reason", self._halt_reason)
        self.store.set_state("consecutive_losses", str(self._consecutive_losses))
