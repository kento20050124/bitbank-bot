"""Data models for candles, orders, positions, and signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    FAILED = "failed"


class PositionState(str, Enum):
    FLAT = "flat"
    PENDING_ENTRY = "pending_entry"
    OPEN = "open"
    SCALING_OUT = "scaling_out"
    TRAILING = "trailing"
    PENDING_EXIT = "pending_exit"
    EMERGENCY_EXIT = "emergency_exit"
    CLOSED = "closed"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Candle:
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    amount: float
    price: float | None
    filled: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    id: int | None = None
    symbol: str = ""
    side: Side = Side.BUY
    entry_price: float = 0.0
    amount: float = 0.0
    current_amount: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0  # For short positions: tracks lowest since entry
    state: PositionState = PositionState.FLAT
    entry_order_id: str = ""
    exit_order_id: str = ""
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: datetime | None = None
    realized_pnl: float = 0.0
    reasoning: str = ""


@dataclass
class Signal:
    direction: SignalDirection
    entry_price: float
    stop_distance: float  # ATR-based distance for stop
    atr_value: float
    timestamp: datetime
    reasoning: str  # Why this signal was generated


@dataclass
class ExitSignal:
    exit_type: str  # "chandelier", "scaling", "overbought", "emergency"
    exit_price: float
    close_ratio: float = 1.0  # 1.0 = full, 0.5 = half
    reasoning: str = ""


@dataclass
class TradeLog:
    timestamp: datetime
    action: str
    symbol: str
    side: str = ""
    amount: float = 0.0
    price: float = 0.0
    reasoning: str = ""
    pnl: float | None = None
