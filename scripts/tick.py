#!/usr/bin/env python3
"""Multi-symbol tick: scans top Bitbank pairs, picks the strongest signal, trades.
Sends email notification on every trade execution.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bitbank_bot.config import load_config
from bitbank_bot.data.collector import fetch_latest_candles
from bitbank_bot.data.store import DataStore
from bitbank_bot.exchange.client import BitbankClient
from bitbank_bot.engine.circuit_breaker import CircuitBreaker
from bitbank_bot.engine.state_machine import transition
from bitbank_bot.strategy.indicators import compute_all_indicators
from bitbank_bot.strategy.entry import generate_entry_signal
from bitbank_bot.strategy.exit import check_exit_conditions
from bitbank_bot.strategy.sizing import calculate_position_size
from bitbank_bot.notifications.discord import DiscordNotifier
from bitbank_bot.data.models import Position, PositionState, Side, TradeLog

import pandas as pd

REPORT_EMAIL = "suzukikento@datarein-inc.com"

SCAN_SYMBOLS = [
    "XRP/JPY", "DOGE/JPY", "XLM/JPY", "ADA/JPY",
    "DOT/JPY", "RENDER/JPY", "AVAX/JPY", "LINK/JPY",
    "SOL/JPY", "LTC/JPY", "ETH/JPY", "BTC/JPY",
]

MIN_TRADE_VALUE_JPY = 500


def setup_logging():
    log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_dir / "tick.log", maxBytes=5_000_000, backupCount=3),
    ]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=handlers)


# ── Email notification ──────────────────────────────────────────────

def _send_email(subject, body):
    try:
        escaped = body.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        script = (
            'tell application "Mail"\n'
            f'  set m to make new outgoing message with properties '
            f'{{subject:"{subject}", content:"{escaped}", visible:false}}\n'
            f'  tell m\n'
            f'    make new to recipient at end of to recipients with properties '
            f'{{address:"{REPORT_EMAIL}"}}\n'
            f'    send\n'
            f'  end tell\n'
            'end tell\n')
        Path("/tmp/bitbank_notify.scpt").write_text(script, encoding="utf-8")
        subprocess.run(["osascript", "/tmp/bitbank_notify.scpt"],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def _notify_trade(action, symbol, side, amount, price, reason, pnl=None, equity=0):
    pnl_line = f"損益: {pnl:+,.0f}円\n" if pnl is not None else ""
    body = (f"[Bitbank BOT] {action}\n\n"
            f"通貨: {symbol}\n方向: {side.upper()}\n数量: {amount:.4f}\n"
            f"価格: {price:.4f}\n理由: {reason}\n{pnl_line}"
            f"総資産: {equity:,.0f}円\n時刻: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    _send_email(f"[BOT] {action} {symbol} {side.upper()}", body)


# ── Order helpers ───────────────────────────────────────────────────

def _place_limit_order(client, symbol, side, amount, price, logger):
    prec = client.exchange.markets.get(symbol, {}).get("precision", {})
    ap = prec.get("amount", 0.0001)
    pp = prec.get("price", 0.001)
    ad = max(0, -int(math.floor(math.log10(ap)))) if ap > 0 else 4
    pd_ = max(0, -int(math.floor(math.log10(pp)))) if pp > 0 else 3
    amount = round(amount, ad)
    price = round(price, pd_)
    if amount < ap:
        logger.error("Amount %.8f below min for %s", amount, symbol)
        return None
    logger.info("LIMIT %s: %.8f %s @ %s", side.upper(), amount, symbol, price)
    order = client.exchange.create_order(symbol, "limit", side, amount, price)
    oid = str(order.get("id", ""))
    for _ in range(6):
        time.sleep(5)
        st = client.exchange.fetch_order(oid, symbol)
        if st["status"] == "closed":
            fp = float(st.get("average") or st.get("price") or price)
            filled = float(st.get("filled", amount))
            logger.info("FILLED: %.8f @ %s", filled, fp)
            return {"id": oid, "price": fp, "filled": filled}
    try:
        client.exchange.cancel_order(oid, symbol)
    except Exception:
        pass
    st = client.exchange.fetch_order(oid, symbol)
    filled = float(st.get("filled", 0))
    if filled > 0:
        return {"id": oid, "price": float(st.get("average") or price), "filled": filled}
    logger.warning("Order canceled unfilled")
    return None


def _get_price(client, symbol, side):
    t = client.exchange.fetch_ticker(symbol)
    if side == "buy":
        return float(t["ask"]) * 1.002
    return float(t["bid"]) * 0.998


# ── Symbol scanning ────────────────────────────────────────────────

def _scan_symbol(client, store, symbol, cfg, logger, equity):
    try:
        fetch_latest_candles(client, store, symbol, "1h", limit=10)
        time.sleep(0.3)
        df_h1 = store.get_candles_df(symbol, "1h", limit=800)
        if len(df_h1) < 200:
            return None
        df_h4 = df_h1.copy().set_index("timestamp")
        df_h4 = df_h4.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna().reset_index()
        if len(df_h4) < 60:
            return None
        kw = dict(ema_fast=cfg.ema_fast_period, ema_slow=cfg.ema_slow_period,
                  adx_period=cfg.adx_period, atr_period=cfg.atr_period,
                  rsi_period=cfg.rsi_period, disparity_ema_period=cfg.disparity_ema_period)
        df_h1 = compute_all_indicators(df_h1, **kw)
        df_h4 = compute_all_indicators(df_h4, **kw)
        last = df_h1.iloc[-1]
        price = last["close"]
        adx = last.get(f"adx_{cfg.adx_period}", 0)
        rsi = last.get(f"rsi_{cfg.rsi_period}", 50)
        atr = last.get(f"atr_{cfg.atr_period}", 0)
        if pd.isna(adx) or pd.isna(rsi) or pd.isna(atr) or atr <= 0:
            return None
        if rsi < 10 or rsi > 90:
            return None
        signal = generate_entry_signal(df_h1, df_h4, cfg)
        if signal is None:
            return None
        if signal.direction.value == "long" and rsi > 70:
            return None
        if signal.direction.value == "short" and rsi < 30:
            return None
        # Multi-factor score
        adx_score = min(float(adx), 60) / 60 * 40
        if signal.direction.value == "long":
            rsi_score = max(0, 20 - abs(rsi - 55) * 0.5)
        else:
            rsi_score = max(0, 20 - abs(rsi - 45) * 0.5)
        atr_pct = (atr / price) * 100
        vol_score = 20 if 1.0 <= atr_pct <= 5.0 else (
            atr_pct * 20 if atr_pct < 1.0 else max(0, 20 - (atr_pct - 5) * 4))
        units = (equity * 0.1) / price
        afford_score = min(20, units * 2)
        score = adx_score + rsi_score + vol_score + afford_score
        return (signal, score, df_h1, price, adx, rsi)
    except Exception as e:
        logger.debug("Scan %s failed: %s", symbol, e)
        return None


# ── Main tick ───────────────────────────────────────────────────────

def run_tick(mode="live"):
    logger = logging.getLogger("tick")
    config = load_config()
    cfg = config.strategy
    logger.info("=" * 50)
    logger.info("TICK START [%s] MULTI-SYMBOL @ %s", mode.upper(),
                datetime.now().strftime("%Y-%m-%d %H:%M"))
    store = DataStore(config.db_path)
    client = BitbankClient(config.exchange)
    client.exchange.load_markets()
    cb = CircuitBreaker(store, cfg)
    notifier = DiscordNotifier(config.notification)

    tick_actions = []
    tick_signals = []

    try:
        balance = client.fetch_balance()
        jpy_free = float(balance.get("JPY", {}).get("free", 0) or 0)
        total_equity = jpy_free
        holdings = {}
        for sym in SCAN_SYMBOLS:
            base = sym.split("/")[0]
            amt = float(balance.get(base, {}).get("free", 0) or 0)
            if amt > 0:
                try:
                    p = float(client.exchange.fetch_ticker(sym)["last"])
                    v = amt * p
                    holdings[sym] = {"amount": amt, "price": p, "value_jpy": v}
                    total_equity += v
                    time.sleep(0.1)
                except Exception:
                    pass
        logger.info("Equity: %.0f JPY (JPY=%.0f + holdings=%.0f)",
                    total_equity, jpy_free, total_equity - jpy_free)
        for sym, h in holdings.items():
            if h["value_jpy"] > 100:
                logger.info("  %s: %.4f (%.0f JPY)", sym, h["amount"], h["value_jpy"])

        # Check exits
        all_open = store.get_open_positions()
        for pos in all_open:
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
                    _execute_exit(pos, es, client, store, cb, notifier, cfg, mode,
                                  jpy_free=jpy_free, base_free=bf, equity=total_equity)
                    tick_actions.append(f"EXIT {pos.symbol}: {es.exit_type}")
                else:
                    store.save_position(pos)
            except Exception as e:
                logger.error("Exit check %s: %s", pos.symbol, e)

        # Liquidate untracked holdings in bearish market
        all_open = store.get_open_positions()
        pos_syms = {p.symbol for p in all_open}
        pos_amounts = {}
        for p in all_open:
            base = p.symbol.split("/")[0]
            pos_amounts[base] = pos_amounts.get(base, 0) + p.current_amount
        for sym, h in list(holdings.items()):
            if h["value_jpy"] < MIN_TRADE_VALUE_JPY:
                continue
            base = sym.split("/")[0]
            # Amount not covered by open positions
            untracked = h["amount"] - pos_amounts.get(base, 0)
            if untracked * h["price"] < MIN_TRADE_VALUE_JPY:
                continue
            # Check if this symbol has a SHORT trend (bearish)
            try:
                fetch_latest_candles(client, store, sym, "1h", limit=10)
                time.sleep(0.3)
                df_check = store.get_candles_df(sym, "1h", limit=200)
                if len(df_check) < 60:
                    continue
                df_check = compute_all_indicators(df_check,
                    ema_fast=cfg.ema_fast_period, ema_slow=cfg.ema_slow_period,
                    adx_period=cfg.adx_period, atr_period=cfg.atr_period,
                    rsi_period=cfg.rsi_period, disparity_ema_period=cfg.disparity_ema_period)
                last = df_check.iloc[-1]
                ema_f = last.get(f"ema_{cfg.ema_fast_period}")
                ema_s = last.get(f"ema_{cfg.ema_slow_period}")
                adx = last.get(f"adx_{cfg.adx_period}")
                price_now = last["close"]
                if pd.isna(ema_f) or pd.isna(ema_s) or pd.isna(adx):
                    continue
                # Bearish: price < EMA fast < EMA slow AND ADX shows trend strength
                if price_now < ema_f < ema_s and adx >= cfg.adx_threshold:
                    market = client.get_market_info(sym)
                    prec = market.get("precision", {}).get("amount", 0.0001)
                    dec = max(0, -int(math.floor(math.log10(prec)))) if prec > 0 else 4
                    sell_amount = round(untracked * 0.95, dec)
                    mn = market.get("limits", {}).get("amount", {}).get("min", 0.0001)
                    if sell_amount < mn:
                        continue
                    reason = f"BEARISH_LIQUIDATE: {sym} EMA{cfg.ema_fast_period}<EMA{cfg.ema_slow_period}, ADX={adx:.1f}"
                    logger.info(reason)
                    if mode == "dry-run":
                        logger.info("DRY-RUN LIQUIDATE: sell %.4f %s", sell_amount, sym)
                        tick_actions.append(f"DRY-RUN LIQUIDATE {sym}: {sell_amount:.4f}")
                    elif mode == "paper":
                        logger.info("PAPER LIQUIDATE: sell %.4f %s", sell_amount, sym)
                        tick_actions.append(f"PAPER LIQUIDATE {sym}: {sell_amount:.4f}")
                    else:
                        op = _get_price(client, sym, "sell")
                        result = _place_limit_order(client, sym, "sell", sell_amount, op, logger)
                        if result:
                            jpy_free += result["filled"] * result["price"]
                            tick_actions.append(f"LIQUIDATE {sym}: sold {result['filled']:.4f} @ {result['price']:.4f}")
                            _notify_trade("LIQUIDATE", sym, "sell", result["filled"],
                                          result["price"], reason, equity=total_equity)
                            notifier.send_trade(action="LIQUIDATE", symbol=sym, side="sell",
                                                amount=result["filled"], price=result["price"],
                                                reason=reason)
                        else:
                            tick_actions.append(f"LIQUIDATE {sym}: order failed")
            except Exception as e:
                logger.debug("Liquidation check %s failed: %s", sym, e)

        # Scan for entries
        if cb.is_halted:
            logger.warning("Circuit breaker: %s", cb.halt_reason)
            tick_actions.append(f"Circuit breaker: {cb.halt_reason}")
        else:
            all_open = store.get_open_positions()
            open_syms = {p.symbol for p in all_open}
            max_pos = cfg.max_concurrent_positions
            if len(all_open) < max_pos:
                cands = []
                for sym in SCAN_SYMBOLS:
                    if sym in open_syms:
                        continue
                    r = _scan_symbol(client, store, sym, cfg, logger, total_equity)
                    if r:
                        sig, sc, _, pr, adx, rsi = r
                        cands.append((sym, sig, sc, pr, adx, rsi))
                        tick_signals.append(f"{sym} {sig.direction.value.upper()} ADX={adx:.1f} RSI={rsi:.1f} score={sc:.1f}")
                        logger.info("  %s: %s ADX=%.1f RSI=%.1f score=%.1f",
                                    sym, sig.direction.value.upper(), adx, rsi, sc)
                if cands:
                    cands.sort(key=lambda x: -x[2])
                    entered = False
                    for bsym, bsig, bsc, bpr, badx, _ in cands:
                        logger.info("TRY: %s %s (score=%.1f)", bsym,
                                    bsig.direction.value.upper(), bsc)
                        base = bsym.split("/")[0]
                        bf = float(balance.get(base, {}).get("free", 0) or 0)
                        if bsig.direction.value == "long" and jpy_free < MIN_TRADE_VALUE_JPY:
                            logger.info("LONG but JPY=%.0f < %d. Skip.", jpy_free, MIN_TRADE_VALUE_JPY)
                            tick_actions.append(f"SKIP {bsym} LONG: JPY不足")
                            continue
                        if bsig.direction.value == "short" and bf * bpr < MIN_TRADE_VALUE_JPY:
                            logger.info("SHORT but %s too small. Next.", base)
                            tick_actions.append(f"SKIP {bsym} SHORT: 残高不足")
                            continue
                        if cb.check_can_trade(bsym, total_equity):
                            _execute_entry(bsig, bsym, total_equity, client, store,
                                           notifier, cfg, mode, jpy_free=jpy_free,
                                           base_free=bf, price=bpr)
                            tick_actions.append(f"ENTRY {bsym} {bsig.direction.value.upper()} @ {bpr:.4f}")
                            entered = True
                            break
                    if not entered:
                        logger.info("All %d signals skipped (insufficient balance).", len(cands))
                else:
                    logger.info("No signals across %d symbols.", len(SCAN_SYMBOLS))
            else:
                logger.info("Max positions reached (%d).", max_pos)
                tick_actions.append(f"Max positions ({max_pos})")

    except Exception as e:
        logger.error("Tick error: %s", e, exc_info=True)
        notifier.send_error(f"Tick error: {e}")
    finally:
        store.close()
        logger.info("TICK END")


# ── Entry execution ─────────────────────────────────────────────────

def _execute_entry(signal, symbol, equity, client, store, notifier, cfg, mode,
                   jpy_free=0, base_free=0, price=0):
    logger = logging.getLogger("tick.entry")
    market = client.get_market_info(symbol)
    min_size = market.get("limits", {}).get("amount", {}).get("min", 0.0001)
    pos_size = calculate_position_size(equity=equity, entry_price=signal.entry_price,
                                       stop_distance=signal.stop_distance, cfg=cfg,
                                       min_order_size=min_size)
    if pos_size <= 0:
        return
    side_str = "buy" if signal.direction.value == "long" else "sell"
    if side_str == "buy":
        cap = jpy_free / price * 0.95
        pos_size = min(pos_size, cap)
    else:
        pos_size = min(pos_size, base_free * 0.95)
    prec = market.get("precision", {}).get("amount", 0.0001)
    dec = max(0, -int(math.floor(math.log10(prec)))) if prec > 0 else 4
    pos_size = round(pos_size, dec)
    if pos_size < min_size:
        return

    position = Position(symbol=symbol, side=Side(side_str),
                        entry_price=signal.entry_price, amount=pos_size,
                        current_amount=pos_size, state=PositionState.FLAT,
                        reasoning=signal.reasoning)
    if signal.direction.value == "long":
        position.stop_price = signal.entry_price - signal.stop_distance
        position.target_price = signal.entry_price + signal.atr_value * cfg.scaling_rr_target
        position.highest_price = signal.entry_price
    else:
        position.stop_price = signal.entry_price + signal.stop_distance
        position.target_price = signal.entry_price - signal.atr_value * cfg.scaling_rr_target
        position.lowest_price = signal.entry_price
    transition(position, PositionState.PENDING_ENTRY, signal.reasoning)

    if mode == "dry-run":
        logger.info("DRY-RUN: %s %.8f %s @ %.4f", side_str, pos_size, symbol, signal.entry_price)
        return
    if mode == "paper":
        logger.info("PAPER ENTRY: %s %.8f %s @ %.4f", side_str, pos_size, symbol, signal.entry_price)
        transition(position, PositionState.OPEN, "Paper fill")
    else:
        op = _get_price(client, symbol, side_str)
        result = _place_limit_order(client, symbol, side_str, pos_size, op, logger)
        if not result:
            notifier.send_error(f"Entry {side_str} {symbol} failed")
            return
        position.entry_price = result["price"]
        position.amount = result["filled"]
        position.current_amount = result["filled"]
        position.entry_order_id = result["id"]
        transition(position, PositionState.OPEN, f"Fill @ {result['price']}")
        logger.info("LIVE ENTRY: %s %.8f %s @ %s", side_str, result["filled"], symbol, result["price"])

    pid = store.save_position(position)
    position.id = pid
    store.log_trade(TradeLog(timestamp=datetime.now(), action="entry", symbol=symbol,
                             side=side_str, amount=position.current_amount,
                             price=position.entry_price, reasoning=signal.reasoning))
    notifier.send_trade(action="ENTRY", symbol=symbol, side=side_str,
                        amount=position.current_amount, price=position.entry_price,
                        reason=signal.reasoning)
    _notify_trade("ENTRY", symbol, side_str, position.current_amount,
                  position.entry_price, signal.reasoning, equity=equity)


# ── Exit execution ──────────────────────────────────────────────────

def _execute_exit(position, exit_signal, client, store, cb, notifier, cfg, mode,
                  jpy_free=0, base_free=0, equity=0):
    logger = logging.getLogger("tick.exit")
    close_amount = position.current_amount * exit_signal.close_ratio
    exit_side = "sell" if position.side == Side.BUY else "buy"
    if exit_side == "sell" and close_amount > base_free:
        close_amount = base_free * 0.99
    elif exit_side == "buy":
        cost = close_amount * exit_signal.exit_price
        if cost > jpy_free:
            close_amount = jpy_free / exit_signal.exit_price * 0.95
    market = client.get_market_info(position.symbol)
    prec = market.get("precision", {}).get("amount", 0.0001)
    dec = max(0, -int(math.floor(math.log10(prec)))) if prec > 0 else 4
    close_amount = round(close_amount, dec)
    mn = market.get("limits", {}).get("amount", {}).get("min", 0.0001)
    if close_amount < mn:
        logger.error("Exit amount too small for %s", position.symbol)
        return
    if mode == "dry-run":
        logger.info("DRY-RUN EXIT: %s %.8f %s", exit_side, close_amount, position.symbol)
        return
    actual_price = exit_signal.exit_price
    if mode == "paper":
        logger.info("PAPER EXIT: %s %.8f %s", exit_side, close_amount, position.symbol)
    else:
        op = _get_price(client, position.symbol, exit_side)
        result = _place_limit_order(client, position.symbol, exit_side, close_amount, op, logger)
        if not result:
            notifier.send_error(f"Exit {position.symbol} failed")
            return
        actual_price = result["price"]
        close_amount = result["filled"]

    if exit_signal.exit_type == "scaling":
        position.current_amount -= close_amount
        position.stop_price = position.entry_price
        transition(position, PositionState.SCALING_OUT, exit_signal.reasoning)
        transition(position, PositionState.TRAILING, "Trailing after scale-out")
    else:
        pnl = ((actual_price - position.entry_price) * close_amount
               if position.side == Side.BUY
               else (position.entry_price - actual_price) * close_amount)
        position.realized_pnl += pnl
        position.current_amount -= close_amount
        if position.current_amount < mn:
            position.current_amount = 0
            transition(position, PositionState.PENDING_EXIT, exit_signal.reasoning)
            transition(position, PositionState.CLOSED, f"PnL: {pnl:.2f}")
    store.save_position(position)
    if position.state == PositionState.CLOSED:
        cb.record_trade_result(position.realized_pnl)
    store.log_trade(TradeLog(timestamp=datetime.now(), action="exit",
                             symbol=position.symbol, side=exit_side,
                             amount=close_amount, price=actual_price,
                             reasoning=exit_signal.reasoning,
                             pnl=position.realized_pnl if position.state == PositionState.CLOSED else None))
    notifier.send_trade(action="EXIT", symbol=position.symbol, side=exit_side,
                        amount=close_amount, price=actual_price,
                        reason=exit_signal.reasoning,
                        pnl=position.realized_pnl if position.state == PositionState.CLOSED else None)
    _notify_trade("EXIT", position.symbol, exit_side, close_amount, actual_price,
                  exit_signal.reasoning,
                  pnl=position.realized_pnl if position.state == PositionState.CLOSED else None,
                  equity=equity)


if __name__ == "__main__":
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    mode = "dry-run" if a.dry_run else "paper" if a.paper else "live"
    run_tick(mode)
