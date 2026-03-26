#!/usr/bin/env python3
"""Daily trading report - sends email summary at midnight."""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bitbank_bot.config import load_config
from bitbank_bot.exchange.client import BitbankClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_report")

REPORT_EMAIL = "suzukikento@datarein-inc.com"


def generate_report() -> str:
    config = load_config()
    client = BitbankClient(config.exchange)

    today = datetime.now().strftime("%Y-%m-%d")

    try:
        balance = client.fetch_balance()
        ticker = client.exchange.fetch_ticker("DOGE/JPY")
        price = float(ticker["last"])
        jpy = float(balance.get("JPY", {}).get("total", 0) or 0)
        doge = float(balance.get("DOGE", {}).get("total", 0) or 0)
        total_jpy = jpy + doge * price
    except Exception as e:
        logger.error("Balance fetch failed: %s", e)
        jpy, doge, price, total_jpy = 0, 0, 0, 0

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row

    trades_today = conn.execute(
        "SELECT * FROM trade_log WHERE timestamp LIKE ? ORDER BY id",
        (f"{today}%",),
    ).fetchall()

    daily_pnl = sum(float(t["pnl"] or 0) for t in trades_today if t["pnl"] is not None)

    open_positions = conn.execute(
        "SELECT * FROM positions WHERE state NOT IN ('closed', 'flat')"
    ).fetchall()

    conn.close()

    # Trade rows
    trade_rows = ""
    for t in trades_today:
        pnl_str = f"{float(t['pnl']):.0f}" if t["pnl"] else "-"
        pnl_color = "#4CAF50" if t["pnl"] and float(t["pnl"]) > 0 else "#f44336" if t["pnl"] and float(t["pnl"]) < 0 else "#666"
        trade_rows += f"""<tr>
            <td style="padding:6px;border-bottom:1px solid #eee">{t['timestamp'][:16]}</td>
            <td style="padding:6px;border-bottom:1px solid #eee"><b>{t['action'].upper()}</b></td>
            <td style="padding:6px;border-bottom:1px solid #eee">{t['side']}</td>
            <td style="padding:6px;border-bottom:1px solid #eee;text-align:right">{float(t['amount']):.4f}</td>
            <td style="padding:6px;border-bottom:1px solid #eee;text-align:right">{float(t['price']):.3f}</td>
            <td style="padding:6px;border-bottom:1px solid #eee;text-align:right;color:{pnl_color}">{pnl_str}</td>
        </tr>"""
    if not trade_rows:
        trade_rows = '<tr><td colspan="6" style="padding:12px;text-align:center;color:#999">本日の取引なし</td></tr>'

    # Open position rows
    open_rows = ""
    for p in open_positions:
        entry = float(p["entry_price"])
        amt = float(p["current_amount"])
        unrealized = (price - entry) * amt if p["side"] == "buy" else (entry - price) * amt
        color = "#4CAF50" if unrealized >= 0 else "#f44336"
        open_rows += f"""<tr>
            <td style="padding:6px">{p['side'].upper()}</td>
            <td style="padding:6px;text-align:right">{amt:.4f}</td>
            <td style="padding:6px;text-align:right">{entry:.3f}</td>
            <td style="padding:6px;text-align:right;color:{color}">{unrealized:+.0f}円</td>
        </tr>"""
    if not open_rows:
        open_rows = '<tr><td colspan="4" style="padding:8px;text-align:center;color:#999">ポジションなし</td></tr>'

    pnl_color = "#4CAF50" if daily_pnl >= 0 else "#f44336"

    return f"""<html><body style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;background:#f5f5f5">
    <div style="background:white;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.1)">
        <h2 style="margin:0 0 4px;color:#333">Bitbank BOT デイリーレポート</h2>
        <p style="color:#999;margin:0 0 20px;font-size:14px">{today}</p>
        <table style="width:100%;margin-bottom:20px"><tr>
            <td style="background:#f8f9fa;border-radius:8px;padding:16px;width:50%">
                <div style="font-size:12px;color:#999">総資産</div>
                <div style="font-size:24px;font-weight:bold;color:#333">{total_jpy:,.0f}円</div>
            </td>
            <td style="width:12px"></td>
            <td style="background:#f8f9fa;border-radius:8px;padding:16px;width:50%">
                <div style="font-size:12px;color:#999">本日損益</div>
                <div style="font-size:24px;font-weight:bold;color:{pnl_color}">{daily_pnl:+,.0f}円</div>
            </td>
        </tr></table>
        <div style="background:#f8f9fa;border-radius:8px;padding:12px;margin-bottom:20px;font-size:14px">
            DOGE/JPY: <b>{price:.3f}円</b> | JPY: {jpy:,.0f}円 | DOGE: {doge:,.4f}
        </div>
        <h3 style="margin:20px 0 8px;color:#333;font-size:16px">本日の取引</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr style="background:#f0f0f0">
                <th style="padding:8px;text-align:left">時刻</th><th style="padding:8px">種別</th>
                <th style="padding:8px">売/買</th><th style="padding:8px;text-align:right">数量</th>
                <th style="padding:8px;text-align:right">価格</th><th style="padding:8px;text-align:right">損益</th>
            </tr>{trade_rows}
        </table>
        <h3 style="margin:20px 0 8px;color:#333;font-size:16px">保有ポジション</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr style="background:#f0f0f0">
                <th style="padding:8px;text-align:left">方向</th><th style="padding:8px;text-align:right">数量</th>
                <th style="padding:8px;text-align:right">取得価格</th><th style="padding:8px;text-align:right">含み損益</th>
            </tr>{open_rows}
        </table>
        <p style="margin-top:24px;font-size:11px;color:#bbb;text-align:center">Bitbank Auto Trading BOT</p>
    </div></body></html>"""


def send_email(to: str, subject: str, body_html: str) -> bool:
    """Send email using Gmail SMTP via Mail.app account, or osascript fallback."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib

    # Plain text version as fallback
    plain = _html_to_plain(body_html)

    # Write plain text and HTML to temp files
    tmp_html = Path("/tmp/bitbank_report.html")
    tmp_html.write_text(body_html, encoding="utf-8")
    tmp_plain = Path("/tmp/bitbank_report.txt")
    tmp_plain.write_text(plain, encoding="utf-8")

    # Escape for AppleScript
    escaped_plain = plain.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    applescript = 'tell application "Mail"\n'
    applescript += f'    set newMsg to make new outgoing message with properties {{subject:"{subject}", content:"{escaped_plain}", visible:false}}\n'
    applescript += '    tell newMsg\n'
    applescript += f'        make new to recipient at end of to recipients with properties {{address:"{to}"}}\n'
    applescript += '        send\n'
    applescript += '    end tell\n'
    applescript += 'end tell\n'

    script_path = Path("/tmp/bitbank_send_email.scpt")
    script_path.write_text(applescript, encoding="utf-8")

    try:
        result = subprocess.run(
            ["osascript", str(script_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info("Email sent to %s", to)
            return True
        logger.warning("Mail.app failed (rc=%d): %s", result.returncode, result.stderr)
    except Exception as e:
        logger.warning("Mail.app error: %s", e)
    return False


def _html_to_plain(html: str) -> str:
    """Convert report HTML to readable plain text."""
    import re
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'</tr>', '\n', text)
    text = re.sub(r'</td>', ' | ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def main():
    logger.info("Generating daily report...")
    html = generate_report()
    subject = f"Bitbank BOT レポート {datetime.now().strftime('%Y/%m/%d')}"

    sent = send_email(REPORT_EMAIL, subject, html)

    # Always save locally
    report_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"report_{datetime.now().strftime('%Y%m%d')}.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", path)

    if not sent:
        logger.warning("Email not sent. Report saved locally.")


if __name__ == "__main__":
    main()
