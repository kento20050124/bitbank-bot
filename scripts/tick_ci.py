#!/usr/bin/env python3
"""GitHub Actions版: tick.pyのCI/CD対応版。
ローカルファイル依存（Mail.app等）を除去し、環境変数でAPIキーを取得。
通知はDiscord Webhookで行う。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bitbank_bot.config import load_config
from bitbank_bot.data.collector import fetch_latest_candles, collect_historical_candles
from bitbank_bot.data.store import DataStore
from bitbank_bot.exchange.client import BitbankClient
from bitbank_bot.engine.circuit_breaker import CircuitBreaker
from bitbank_bot.engine.state_machine import transition
from bitbank_bot.strategy.indicators import compute_all_indicators
from bitbank_bot.strategy.entry import generate_entry_signal
from bitbank_bot.strategy.exit import check_exit_conditions
from bitbank_bot.strategy.sizing import calculate_position_size
from bitbank_bot.data.models import Position, PositionState, Side, TradeLog

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tick_ci")

SCAN_SYMBOLS = [
    "XRP/JPY", "DOGE/JPY", "XLM/JPY", "ADA/JPY",
    "DOT/JPY", "RENDER/JPY", "AVAX/JPY", "LINK/JPY",
    "SOL/JPY", "LTC/JPY", "ETH/JPY", "BTC/JPY",
]
MIN_TRADE_VALUE_JPY = 500
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
REPORT_EMAIL = os.getenv("REPORT_EMAIL", "suzukikento@datarein-inc.com")


def _notify(title, body):
    """Send notification via Discord webhook."""
    if not WEBHOOK_URL:
        return
    try:
        data = json.dumps({"content": f"**{title}**\n```\n{body}\n```"}).encode()
        req = Request(WEBHOOK_URL, data=data,
                      headers={"Content-Type": "application/json"})
        urlopen(req, timeout=10)
    except Exception as e:
        logger.warning("Webhook failed: %s", e)


def _place_limit_order(client, symbol, side, amount, price):
    prec = client.exchange.markets.get(symbol, {}).get("precision", {})
    ap = prec.get("amount", 0.0001)
    pp = prec.get("price", 0.001)
    ad = max(0, -int(math.floor(math.log10(ap)))) if ap and ap > 0 else 4
    pd_ = max(0, -int(math.floor(math.log10(pp)))) if pp and pp > 0 else 3
    amount = round(amount, ad)
    price = round(price, pd_)
    if amount < (ap or 0.0001):
        return None
    logger.info("LIMIT %s: %.8f %s @ %s", side.upper(), amount, symbol, price)
    order = client.exchange.create_order(symbol, "limit", side, amount, price)
    oid = str(order.get("id", ""))
    for _ in range(6):
        time.sleep(5)
        st = client.exchange.fetch_order(oid, symbol)
        if st["status"] == "closed":
            fp = float(st.get("average") or st.get("price") or price)
            return {"id": oid, "price": fp, "filled": float(st.get("filled", amount))}
    try:
        client.exchange.cancel_order(oid, symbol)
    except Exception:
        pass
    st = client.exchange.fetch_order(oid, symbol)
    filled = float(st.get("filled", 0))
    if filled > 0:
        return {"id": oid, "price": float(st.get("average") or price), "filled": filled}
    return None


def _get_price(client, symbol, side):
    t = client.exchange.fetch_ticker(symbol)
    return float(t["ask"]) * 1.002 if side == "buy" else float(t["bid"]) * 0.998


def _scan(client, store, symbol, cfg, equity):
    try:
        fetch_latest_candles(client, store, symbol, "1h", limit=10)
        time.sleep(0.3)
        df = store.get_candles_df(symbol, "1h", limit=800)
        if len(df) < 200:
            return None
        df4 = df.copy().set_index("timestamp")
        df4 = df4.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna().reset_index()
        if len(df4) < 60:
            return None
        kw = dict(ema_fast=cfg.ema_fast_period, ema_slow=cfg.ema_slow_period,
                  adx_period=cfg.adx_period, atr_period=cfg.atr_period,
                  rsi_period=cfg.rsi_period, disparity_ema_period=cfg.disparity_ema_period)
        df = compute_all_indicators(df, **kw)
        df4 = compute_all_indicators(df4, **kw)
        last = df.iloc[-1]
        price, adx, rsi, atr = (last["close"], last.get(f"adx_{cfg.adx_period}", 0),
                                 last.get(f"rsi_{cfg.rsi_period}", 50),
                                 last.get(f"atr_{cfg.atr_period}", 0))
        if any(pd.isna(v) for v in [adx, rsi, atr]) or atr <= 0 or rsi < 10 or rsi > 90:
            return None
        sig = generate_entry_signal(df, df4, cfg)
        if not sig:
            return None
        if sig.direction.value == "long" and rsi > 70:
            return None
        if sig.direction.value == "short" and rsi < 30:
            return None
        # Score
        adx_s = min(float(adx), 60) / 60 * 40
        rsi_s = max(0, 20 - abs(rsi - (55 if sig.direction.value == "long" else 45)) * 0.5)
        atr_pct = (atr / price) * 100
        vol_s = 20 if 1 <= atr_pct <= 5 else (atr_pct * 20 if atr_pct < 1 else max(0, 20 - (atr_pct - 5) * 4))
        aff_s = min(20, (equity * 0.1 / price) * 2)
        return (sig, adx_s + rsi_s + vol_s + aff_s, price, adx, rsi)
    except Exception:
        return None


def main():
    config = load_config()
    cfg = config.strategy
    logger.info("=" * 50)
    logger.info("TICK_CI START @ %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Ensure data directory
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    store = DataStore(config.db_path)
    client = BitbankClient(config.exchange)
    client.exchange.load_markets()
    cb = CircuitBreaker(store, cfg)

    # Ensure we have enough data for each symbol
    for sym in SCAN_SYMBOLS:
        df = store.get_candles_df(sym, "1h", limit=10000)
        if len(df) < 200:
            logger.info("Collecting data for %s...", sym)
            try:
                collect_historical_candles(client, store, sym, "1h", days_back=30)
            except Exception:
                pass
            time.sleep(0.5)

    try:
        balance = client.fetch_balance()
        jpy = float(balance.get("JPY", {}).get("free", 0) or 0)
        total = jpy
        holdings = {}
        for sym in SCAN_SYMBOLS:
            base = sym.split("/")[0]
            amt = float(balance.get(base, {}).get("free", 0) or 0)
            if amt > 0:
                try:
                    p = float(client.exchange.fetch_ticker(sym)["last"])
                    holdings[sym] = {"amount": amt, "price": p, "value": amt * p}
                    total += amt * p
                    time.sleep(0.1)
                except Exception:
                    pass

        logger.info("Equity: %.0f JPY", total)
        for s, h in holdings.items():
            if h["value"] > 100:
                logger.info("  %s: %.4f (%.0f JPY)", s, h["amount"], h["value"])

        # Exits
        for pos in store.get_open_positions():
            try:
                df = store.get_candles_df(pos.symbol, "1h", limit=800)
                if len(df) < 50:
                    continue
                df = compute_all_indicators(df, ema_fast=cfg.ema_fast_period,
                    ema_slow=cfg.ema_slow_period, adx_period=cfg.adx_period,
                    atr_period=cfg.atr_period, rsi_period=cfg.rsi_period,
                    disparity_ema_period=cfg.disparity_ema_period)
                es = check_exit_conditions(pos, df, cfg)
                if es:
                    logger.info("EXIT [%s]: %s", pos.symbol, es.reasoning)
                    base = pos.symbol.split("/")[0]
                    bf = float(balance.get(base, {}).get("free", 0) or 0)
                    _do_exit(pos, es, client, store, cb, cfg, jpy, bf, total)
                else:
                    store.save_position(pos)
            except Exception as e:
                logger.error("Exit check %s: %s", pos.symbol, e)

        # Entries
        if not cb.is_halted:
            opens = store.get_open_positions()
            open_syms = {p.symbol for p in opens}
            max_pos = 1 if total < 100_000 else cfg.max_concurrent_positions
            if len(opens) < max_pos:
                cands = []
                for sym in SCAN_SYMBOLS:
                    if sym in open_syms:
                        continue
                    r = _scan(client, store, sym, cfg, total)
                    if r:
                        sig, sc, pr, adx, rsi = r
                        cands.append((sym, sig, sc, pr, adx, rsi))
                        logger.info("  %s: %s ADX=%.1f RSI=%.1f score=%.1f",
                                    sym, sig.direction.value.upper(), adx, rsi, sc)
                if cands:
                    cands.sort(key=lambda x: -x[2])
                    bsym, bsig, bsc, bpr, _, _ = cands[0]
                    logger.info("BEST: %s %s (score=%.1f)", bsym, bsig.direction.value.upper(), bsc)
                    base = bsym.split("/")[0]
                    bf = float(balance.get(base, {}).get("free", 0) or 0)
                    if bsig.direction.value == "long" and jpy < MIN_TRADE_VALUE_JPY:
                        logger.info("LONG but JPY=%.0f. Skip.", jpy)
                    elif bsig.direction.value == "short" and bf * bpr < MIN_TRADE_VALUE_JPY:
                        logger.info("SHORT but %s too small. Skip.", base)
                    elif cb.check_can_trade(bsym, total):
                        _do_entry(bsig, bsym, total, client, store, cfg, jpy, bf, bpr)
                else:
                    logger.info("No signals across %d symbols.", len(SCAN_SYMBOLS))
        else:
            logger.warning("Circuit breaker: %s", cb.halt_reason)

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        _notify("BOT ERROR", str(e))
    finally:
        store.close()
        logger.info("TICK_CI END")


def _do_entry(sig, sym, equity, client, store, cfg, jpy, bf, price):
    market = client.get_market_info(sym)
    mn = market.get("limits", {}).get("amount", {}).get("min", 0.0001)
    ps = calculate_position_size(equity=equity, entry_price=sig.entry_price,
                                  stop_distance=sig.stop_distance, cfg=cfg, min_order_size=mn)
    if ps <= 0:
        return
    side = "buy" if sig.direction.value == "long" else "sell"
    if side == "buy":
        ps = min(ps, jpy / price * 0.95)
    else:
        ps = min(ps, bf * 0.95)
    prec = market.get("precision", {}).get("amount", 0.0001)
    dec = max(0, -int(math.floor(math.log10(prec)))) if prec and prec > 0 else 4
    ps = round(ps, dec)
    if ps < mn:
        return

    pos = Position(symbol=sym, side=Side(side), entry_price=sig.entry_price,
                   amount=ps, current_amount=ps, state=PositionState.FLAT, reasoning=sig.reasoning)
    if sig.direction.value == "long":
        pos.stop_price = sig.entry_price - sig.stop_distance
        pos.target_price = sig.entry_price + sig.atr_value * cfg.scaling_rr_target
        pos.highest_price = sig.entry_price
    else:
        pos.stop_price = sig.entry_price + sig.stop_distance
        pos.target_price = sig.entry_price - sig.atr_value * cfg.scaling_rr_target
        pos.lowest_price = sig.entry_price
    transition(pos, PositionState.PENDING_ENTRY, sig.reasoning)

    op = _get_price(client, sym, side)
    result = _place_limit_order(client, sym, side, ps, op)
    if not result:
        _notify("ORDER FAILED", f"{side} {sym} {ps}")
        return
    pos.entry_price = result["price"]
    pos.amount = result["filled"]
    pos.current_amount = result["filled"]
    pos.entry_order_id = result["id"]
    transition(pos, PositionState.OPEN, f"Fill @ {result['price']}")
    logger.info("ENTRY: %s %.8f %s @ %s", side, result["filled"], sym, result["price"])

    store.save_position(pos)
    store.log_trade(TradeLog(timestamp=datetime.now(), action="entry", symbol=sym,
                             side=side, amount=pos.current_amount, price=pos.entry_price,
                             reasoning=sig.reasoning))
    _notify(f"ENTRY {sym}", f"{side.upper()} {pos.current_amount:.4f} @ {pos.entry_price:.4f}\n{sig.reasoning}\nEquity: {equity:,.0f} JPY")


def _do_exit(pos, es, client, store, cb, cfg, jpy, bf, equity):
    ca = pos.current_amount * es.close_ratio
    exit_side = "sell" if pos.side == Side.BUY else "buy"
    if exit_side == "sell" and ca > bf:
        ca = bf * 0.99
    elif exit_side == "buy" and ca * es.exit_price > jpy:
        ca = jpy / es.exit_price * 0.95
    market = client.get_market_info(pos.symbol)
    prec = market.get("precision", {}).get("amount", 0.0001)
    dec = max(0, -int(math.floor(math.log10(prec)))) if prec and prec > 0 else 4
    ca = round(ca, dec)
    mn = market.get("limits", {}).get("amount", {}).get("min", 0.0001)
    if ca < mn:
        return

    op = _get_price(client, pos.symbol, exit_side)
    result = _place_limit_order(client, pos.symbol, exit_side, ca, op)
    if not result:
        _notify("EXIT FAILED", f"{pos.symbol}")
        return
    ap = result["price"]
    ca = result["filled"]

    if es.exit_type == "scaling":
        pos.current_amount -= ca
        pos.stop_price = pos.entry_price
        transition(pos, PositionState.SCALING_OUT, es.reasoning)
        transition(pos, PositionState.TRAILING, "Trailing")
    else:
        pnl = ((ap - pos.entry_price) * ca if pos.side == Side.BUY
               else (pos.entry_price - ap) * ca)
        pos.realized_pnl += pnl
        pos.current_amount -= ca
        if pos.current_amount < mn:
            pos.current_amount = 0
            transition(pos, PositionState.PENDING_EXIT, es.reasoning)
            transition(pos, PositionState.CLOSED, f"PnL: {pnl:.2f}")

    store.save_position(pos)
    if pos.state == PositionState.CLOSED:
        cb.record_trade_result(pos.realized_pnl)
    store.log_trade(TradeLog(timestamp=datetime.now(), action="exit", symbol=pos.symbol,
                             side=exit_side, amount=ca, price=ap, reasoning=es.reasoning,
                             pnl=pos.realized_pnl if pos.state == PositionState.CLOSED else None))
    _notify(f"EXIT {pos.symbol}",
            f"{exit_side.upper()} {ca:.4f} @ {ap:.4f}\n{es.reasoning}\n"
            f"PnL: {pos.realized_pnl:+.0f} JPY" if pos.state == PositionState.CLOSED else "")


if __name__ == "__main__":
    main()
