"""SQLite persistence layer for candles, orders, positions, and trade logs."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from bitbank_bot.data.models import (
    Candle,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Side,
    TradeLog,
)


class DataStore:
    """SQLite-based data store for all bot state."""

    def __init__(self, db_path: str = "data/candles.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL,
                filled REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                current_amount REAL NOT NULL,
                stop_price REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                highest_price REAL DEFAULT 0,
                lowest_price REAL DEFAULT 0,
                state TEXT DEFAULT 'flat',
                entry_order_id TEXT DEFAULT '',
                exit_order_id TEXT DEFAULT '',
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                realized_pnl REAL DEFAULT 0,
                reasoning TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT DEFAULT '',
                amount REAL DEFAULT 0,
                price REAL DEFAULT 0,
                reasoning TEXT DEFAULT '',
                pnl REAL
            );

            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_candles_lookup
                ON candles(symbol, timeframe, timestamp);
            CREATE INDEX IF NOT EXISTS idx_positions_state
                ON positions(state);
        """)
        self.conn.commit()

    # --- Candle operations ---

    def save_candles(self, candles: list[Candle]):
        """Insert or replace candles."""
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO candles
               (timestamp, symbol, timeframe, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.timestamp.isoformat(),
                    c.symbol,
                    c.timeframe,
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                )
                for c in candles
            ],
        )
        self.conn.commit()

    def get_candles_df(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Get candles as a pandas DataFrame, sorted by timestamp ascending."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT timestamp, open, high, low, close, volume
               FROM candles
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (symbol, timeframe, limit),
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_latest_candle_time(self, symbol: str, timeframe: str) -> datetime | None:
        """Get the timestamp of the most recent candle."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT MAX(timestamp) as ts FROM candles
               WHERE symbol = ? AND timeframe = ?""",
            (symbol, timeframe),
        )
        row = cur.fetchone()
        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"])
        return None

    # --- Order operations ---

    def save_order(self, order: Order):
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, symbol, side, order_type, amount, price, filled, status,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order.order_id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.amount,
                order.price,
                order.filled,
                order.status.value,
                order.created_at.isoformat(),
                order.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_order(self, order_id: str) -> Order | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Order(
            order_id=row["order_id"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            order_type=OrderType(row["order_type"]),
            amount=row["amount"],
            price=row["price"],
            filled=row["filled"],
            status=OrderStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get_pending_orders(self, symbol: str) -> list[Order]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM orders WHERE symbol = ? AND status IN ('pending', 'open')",
            (symbol,),
        )
        rows = cur.fetchall()
        return [
            Order(
                order_id=r["order_id"],
                symbol=r["symbol"],
                side=Side(r["side"]),
                order_type=OrderType(r["order_type"]),
                amount=r["amount"],
                price=r["price"],
                filled=r["filled"],
                status=OrderStatus(r["status"]),
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    # --- Position operations ---

    def save_position(self, position: Position) -> int:
        cur = self.conn.cursor()
        if position.id is None:
            cur.execute(
                """INSERT INTO positions
                   (symbol, side, entry_price, amount, current_amount, stop_price,
                    target_price, highest_price, lowest_price, state, entry_order_id,
                    exit_order_id, opened_at, closed_at, realized_pnl, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    position.symbol,
                    position.side.value,
                    position.entry_price,
                    position.amount,
                    position.current_amount,
                    position.stop_price,
                    position.target_price,
                    position.highest_price,
                    position.lowest_price,
                    position.state.value,
                    position.entry_order_id,
                    position.exit_order_id,
                    position.opened_at.isoformat(),
                    position.closed_at.isoformat() if position.closed_at else None,
                    position.realized_pnl,
                    position.reasoning,
                ),
            )
            self.conn.commit()
            return cur.lastrowid
        else:
            cur.execute(
                """UPDATE positions SET
                   symbol=?, side=?, entry_price=?, amount=?, current_amount=?,
                   stop_price=?, target_price=?, highest_price=?, lowest_price=?,
                   state=?, entry_order_id=?, exit_order_id=?, opened_at=?,
                   closed_at=?, realized_pnl=?, reasoning=?
                   WHERE id=?""",
                (
                    position.symbol,
                    position.side.value,
                    position.entry_price,
                    position.amount,
                    position.current_amount,
                    position.stop_price,
                    position.target_price,
                    position.highest_price,
                    position.lowest_price,
                    position.state.value,
                    position.entry_order_id,
                    position.exit_order_id,
                    position.opened_at.isoformat(),
                    position.closed_at.isoformat() if position.closed_at else None,
                    position.realized_pnl,
                    position.reasoning,
                    position.id,
                ),
            )
            self.conn.commit()
            return position.id

    def get_open_positions(self, symbol: str | None = None) -> list[Position]:
        cur = self.conn.cursor()
        active_states = ("open", "scaling_out", "trailing", "pending_exit")
        if symbol:
            cur.execute(
                f"SELECT * FROM positions WHERE symbol = ? AND state IN {active_states}",
                (symbol,),
            )
        else:
            cur.execute(
                f"SELECT * FROM positions WHERE state IN {active_states}"
            )
        return [self._row_to_position(r) for r in cur.fetchall()]

    def _row_to_position(self, row) -> Position:
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            entry_price=row["entry_price"],
            amount=row["amount"],
            current_amount=row["current_amount"],
            stop_price=row["stop_price"],
            target_price=row["target_price"],
            highest_price=row["highest_price"],
            lowest_price=row["lowest_price"],
            state=PositionState(row["state"]),
            entry_order_id=row["entry_order_id"],
            exit_order_id=row["exit_order_id"],
            opened_at=datetime.fromisoformat(row["opened_at"]),
            closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
            realized_pnl=row["realized_pnl"],
            reasoning=row["reasoning"],
        )

    # --- Trade Log ---

    def log_trade(self, log: TradeLog):
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO trade_log
               (timestamp, action, symbol, side, amount, price, reasoning, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log.timestamp.isoformat(),
                log.action,
                log.symbol,
                log.side,
                log.amount,
                log.price,
                log.reasoning,
                log.pnl,
            ),
        )
        self.conn.commit()

    # --- State KV store ---

    def get_state(self, key: str, default: str = "") -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str):
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO state (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_daily_trade_count(self, symbol: str) -> int:
        """Count entry trades executed today (not exits)."""
        today = datetime.now().strftime("%Y-%m-%d")
        cur = self.conn.cursor()
        cur.execute(
            """SELECT COUNT(*) as cnt FROM trade_log
               WHERE symbol = ? AND timestamp LIKE ? AND action = 'entry'""",
            (symbol, f"{today}%"),
        )
        return cur.fetchone()["cnt"]

    def get_daily_pnl(self, symbol: str) -> float:
        """Sum of realized PnL today."""
        today = datetime.now().strftime("%Y-%m-%d")
        cur = self.conn.cursor()
        cur.execute(
            """SELECT COALESCE(SUM(pnl), 0) as total FROM trade_log
               WHERE symbol = ? AND timestamp LIKE ? AND pnl IS NOT NULL""",
            (symbol, f"{today}%"),
        )
        return cur.fetchone()["total"]

    def close(self):
        self.conn.close()
