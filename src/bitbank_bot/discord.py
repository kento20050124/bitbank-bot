"""Discord webhook notifications for trade alerts and errors."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from bitbank_bot.config import NotificationConfig

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send notifications to Discord via webhook."""

    def __init__(self, config: NotificationConfig):
        self.webhook_url = config.discord_webhook_url
        self.enabled = config.enabled and bool(self.webhook_url)

    def send_message(self, content: str):
        """Send a plain text message."""
        if not self.enabled:
            return
        self._post({"content": content})

    def send_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reason: str = "",
        pnl: float | None = None,
    ):
        """Send a trade notification with embed."""
        if not self.enabled:
            return

        color = 0x00FF00 if action == "ENTRY" else (0xFF0000 if pnl and pnl < 0 else 0x0099FF)

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side.upper(), "inline": True},
            {"name": "Amount", "value": f"{amount:.8f}", "inline": True},
            {"name": "Price", "value": f"{price:.4f}", "inline": True},
        ]

        if pnl is not None:
            pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            fields.append({"name": "PnL", "value": f"{pnl_str} JPY", "inline": True})

        if reason:
            fields.append({"name": "Reason", "value": reason[:1024], "inline": False})

        embed = {
            "title": f"{action} {side.upper()} {symbol}",
            "color": color,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._post({"embeds": [embed]})

    def send_error(self, message: str):
        """Send an error notification."""
        if not self.enabled:
            return

        embed = {
            "title": "ERROR",
            "description": message[:2048],
            "color": 0xFF0000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._post({"embeds": [embed]})

    def send_daily_summary(
        self,
        symbol: str,
        equity: float,
        daily_pnl: float,
        total_trades: int,
        win_rate: float,
    ):
        """Send a daily performance summary."""
        if not self.enabled:
            return

        pnl_str = f"+{daily_pnl:.2f}" if daily_pnl >= 0 else f"{daily_pnl:.2f}"
        color = 0x00FF00 if daily_pnl >= 0 else 0xFF0000

        embed = {
            "title": f"Daily Summary - {symbol}",
            "color": color,
            "fields": [
                {"name": "Equity", "value": f"{equity:,.0f} JPY", "inline": True},
                {"name": "Daily PnL", "value": f"{pnl_str} JPY", "inline": True},
                {"name": "Trades Today", "value": str(total_trades), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._post({"embeds": [embed]})

    def _post(self, payload: dict):
        """Post to Discord webhook."""
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.warning("Discord webhook returned %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("Failed to send Discord notification: %s", e)
