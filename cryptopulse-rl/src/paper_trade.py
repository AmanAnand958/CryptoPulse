#!/usr/bin/env python3
"""
paper_trade.py
==============
Paper trading forward-test tracker for CryptoPulse RL.

⚠️  PAPER TRADING ONLY — NO REAL FUNDS ARE USED OR RISKED.

Simulates live trading using the latest cached LLM signal + LinUCB bandit
policy for each coin, tracks a virtual $10,000 portfolio per coin, and
logs cumulative returns + position history to:
  data/processed/paper_trade_log.json

Usage:
    python src/paper_trade.py                    # run today's paper trade step
    python src/paper_trade.py --show             # print portfolio summary
    python src/paper_trade.py --reset            # reset portfolio to $10,000
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

LOG_PATH = BASE_DIR / "data" / "processed" / "paper_trade_log.json"
SIGNAL_CACHE_PATH = BASE_DIR / "data" / "processed" / "signal_cache.json"
POLICY_PATH = BASE_DIR / "data" / "processed" / "policy_checkpoint.json"

INITIAL_CAPITAL = 10_000.0
COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano"]
TX_COST_RATE = 0.001   # 0.1% per trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return {coin: {"portfolio_value": INITIAL_CAPITAL, "position": 0.0, "trades": []} for coin in COINS}


def save_log(log: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def get_latest_signal(coin: str) -> dict:
    """Get the most recent cached signal for a coin."""
    if not SIGNAL_CACHE_PATH.exists():
        return {"direction": "hold", "confidence": 0.0, "rationale": "no_cache"}
    with open(SIGNAL_CACHE_PATH) as f:
        cache = json.load(f)
    coin_keys = {k: v for k, v in cache.items() if k.startswith(f"{coin}::")}
    if not coin_keys:
        return {"direction": "hold", "confidence": 0.0, "rationale": "no_signal"}
    latest_key = max(coin_keys.keys())
    return {**coin_keys[latest_key], "date": latest_key.split("::")[1]}


def get_latest_price(coin: str) -> float | None:
    """Get latest close price from prices.csv."""
    prices_path = BASE_DIR / "data" / "processed" / "prices.csv"
    if not prices_path.exists():
        return None
    df = pd.read_csv(prices_path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    coin_df = df[df["coin"] == coin].sort_values("date")
    if coin_df.empty:
        return None
    return float(coin_df.iloc[-1]["close"])


def run_step(log: dict, dry_run: bool = False) -> dict:
    """Execute one paper trading step for today."""
    today = str(date.today())

    from rl_policy import LinUCBBandit
    from features import build_state_vector

    try:
        bandit = LinUCBBandit.load(str(POLICY_PATH)) if POLICY_PATH.exists() else LinUCBBandit()
    except Exception:
        bandit = LinUCBBandit()

    print(f"\n{'='*60}")
    print(f"  CryptoPulse Paper Trade — {today}")
    print(f"  ⚠️  PAPER TRADING ONLY — NOT REAL FUNDS")
    print(f"{'='*60}")

    prices_path = BASE_DIR / "data" / "processed" / "prices.csv"
    prices_df = pd.read_csv(prices_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)

    for coin in COINS:
        signal = get_latest_signal(coin)
        direction = signal.get("direction", "hold")
        confidence = float(signal.get("confidence", 0.0))
        latest_price = get_latest_price(coin)

        coin_prices = prices_df[prices_df["coin"] == coin].sort_values("date")
        closes = coin_prices["close"].values
        if len(closes) < 8:
            logger.warning("Not enough price history for %s, skipping", coin)
            continue

        # Build state vector
        i = len(closes) - 1
        ret_1d = (closes[i] / closes[i-1] - 1) if closes[i-1] > 0 else 0.0
        ret_7d = (closes[i] / closes[i-7] - 1) if i >= 7 and closes[i-7] > 0 else 0.0
        rets_7d = [np.log(closes[j]/closes[j-1]) for j in range(max(1,i-6), i+1) if closes[j-1] > 0]
        vol_7d = float(np.std(rets_7d)) if len(rets_7d) > 1 else 0.01
        state = build_state_vector(
            direction=direction, confidence=confidence,
            ret_1d=ret_1d, ret_7d=ret_7d, vol_7d=vol_7d,
            past_directions=[], past_returns=[],
        )

        # Select arm (allocation fraction)
        arm_idx = bandit.select_arm(state, training=False)
        ARMS = [0.0, 0.25, 0.5, 1.0]
        allocation = ARMS[arm_idx]

        # Apply direction
        if direction == "hold":
            position = 0.0
        elif direction == "short":
            position = -allocation
        else:
            position = allocation

        # Track portfolio (paper)
        coin_log = log.setdefault(coin, {"portfolio_value": INITIAL_CAPITAL, "position": 0.0, "trades": []})
        prev_position = coin_log["position"]
        portfolio_value = coin_log["portfolio_value"]

        # Transaction cost for position change
        pos_change = abs(position - prev_position)
        tx_cost = pos_change * TX_COST_RATE * portfolio_value

        trade_record = {
            "date": today,
            "signal_date": signal.get("date", "unknown"),
            "direction": direction,
            "confidence": confidence,
            "arm": allocation,
            "position": round(position, 3),
            "price": latest_price,
            "portfolio_before": round(portfolio_value, 2),
            "tx_cost": round(tx_cost, 4),
            "rationale": signal.get("rationale", "")[:100],
        }

        coin_log["position"] = position
        coin_log["portfolio_value"] = round(portfolio_value - tx_cost, 4)
        coin_log["trades"].append(trade_record)

        ret_pct = ((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        arrow = "🟢" if position > 0 else ("🔴" if position < 0 else "⚪")
        print(f"\n  {arrow} {coin.upper():<12} | Signal: {direction.upper():<6} (conf={confidence:.2f})")
        print(f"     Bandit Alloc: {allocation:.0%}  Position: {position:+.2f}  Price: ${latest_price:,.2f}" if latest_price else
              f"     Bandit Alloc: {allocation:.0%}  Position: {position:+.2f}")
        print(f"     Portfolio: ${coin_log['portfolio_value']:,.2f}  (P&L: {ret_pct:+.2f}%)")
        print(f"     Rationale: {signal.get('rationale','')[:80]}")

    if not dry_run:
        save_log(log)
        bandit.save(str(POLICY_PATH))
        print(f"\n  ✅ Log saved → {LOG_PATH}")
        print(f"  ✅ Policy saved → {POLICY_PATH}")

    return log


def show_summary(log: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  CryptoPulse Paper Trade Portfolio Summary")
    print(f"  ⚠️  PAPER TRADING ONLY — NOT REAL FUNDS")
    print(f"{'='*60}")
    total_start = INITIAL_CAPITAL * len(COINS)
    total_now = 0.0
    for coin in COINS:
        coin_log = log.get(coin, {})
        val = coin_log.get("portfolio_value", INITIAL_CAPITAL)
        n_trades = len([t for t in coin_log.get("trades", []) if t.get("position", 0) != 0])
        ret = (val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"  {coin.upper():<14}: ${val:>10,.2f}  ({ret:+.2f}%)  |  {n_trades} active trades")
        total_now += val
    total_ret = (total_now - total_start) / total_start * 100
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<14}: ${total_now:>10,.2f}  ({total_ret:+.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="CryptoPulse paper trading tracker")
    parser.add_argument("--show", action="store_true", help="Show portfolio summary only")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio to $10,000")
    parser.add_argument("--dry-run", action="store_true", help="Show what would trade, don't save")
    args = parser.parse_args()

    if args.reset:
        log = {coin: {"portfolio_value": INITIAL_CAPITAL, "position": 0.0, "trades": []} for coin in COINS}
        save_log(log)
        print("Portfolio reset to $10,000 per coin.")
        return

    log = load_log()

    if args.show:
        show_summary(log)
        return

    log = run_step(log, dry_run=args.dry_run)
    show_summary(log)


if __name__ == "__main__":
    main()
