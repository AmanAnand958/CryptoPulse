"""
backtest.py
===========
Walk-forward backtest engine for CryptoPulse RL.

CRITICAL: This is the ONLY validation methodology used.
  - No single train/test split anywhere in this project.
  - Walk-forward: train on 60 days → evaluate on 14 days → roll forward.
  - Policy is reset at the start of each training window (from scratch)
    so each fold is independent — no data leakage across folds.

Tracked per step:
  - LLM signal (direction, confidence, rationale)
  - Bandit action (arm index + position multiplier)
  - Realized forward return
  - Portfolio value (bandit, buy-and-hold, llm-only)
  - Cumulative regret vs. oracle (perfect-foresight baseline)

Oracle policy: always picks the action that would have maximized the
  forward return — used only as a reference ceiling, never claimed as
  achievable in live trading.
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import timedelta

from features import build_state_vector, STATE_DIM
from rl_policy import LinUCBBandit, ACTIONS, N_ACTIONS, MAX_ALLOCATION, TX_COST_RATE, decide, compute_reward
from baselines import run_buy_and_hold, run_llm_only

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Walk-forward parameters
# ---------------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 60
EVAL_WINDOW_DAYS = 14
INITIAL_PORTFOLIO = 10_000.0

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "data" / "processed" / "backtest_results.json"


# ---------------------------------------------------------------------------
# Per-step bandit execution
# ---------------------------------------------------------------------------

def run_bandit_on_window(
    bandit: LinUCBBandit,
    coin: str,
    prices: pd.Series,       # indexed by date
    signals: pd.DataFrame,   # columns: date, direction, confidence, rationale
    past_directions: list,
    past_returns: list,
    start_portfolio: float,
    training: bool,
) -> tuple[pd.DataFrame, list, list]:
    """
    Run the bandit policy on one coin over one window (train or eval).

    Returns
    -------
    results_df : per-step records
    updated_past_directions : extended list for hit-rate computation
    updated_past_returns : extended list for hit-rate computation
    """
    sig_map = signals.set_index("date") if "date" in signals.columns else signals

    portfolio = start_portfolio
    prev_position = 0.0
    records = []

    dates = sorted(prices.index)
    for i, date in enumerate(dates):
        if i == 0:
            records.append({
                "date": date, "coin": coin, "training": training,
                "portfolio_value": portfolio, "daily_return": 0.0,
                "action_arm": 2, "position_multiplier": 0.0,
                "direction": "hold", "confidence": 0.0, "rationale": "start",
                "forward_return": 0.0,
            })
            continue

        # Get signal for this date
        if date in sig_map.index:
            sig_row = sig_map.loc[date]
            direction = sig_row.get("direction", "hold") if hasattr(sig_row, "get") else sig_row["direction"]
            confidence = float(sig_row.get("confidence", 0.0) if hasattr(sig_row, "get") else sig_row["confidence"])
            rationale = sig_row.get("rationale", "") if hasattr(sig_row, "get") else sig_row.get("rationale", "")
        else:
            direction, confidence, rationale = "hold", 0.0, "no_signal"

        # Compute feature returns
        prev_price = prices.iloc[i - 1]
        curr_price = prices.iloc[i]
        ret_1d = (curr_price / prev_price - 1) if prev_price > 0 else 0.0

        price_vals = prices.values
        ret_7d = (price_vals[i] / price_vals[max(0, i-7)] - 1) if i >= 7 else 0.0
        ret_30d = (price_vals[i] / price_vals[max(0, i-30)] - 1) if i >= 30 else 0.0

        rets_window = [
            np.log(price_vals[j] / price_vals[j-1])
            for j in range(max(1, i-6), i+1)
            if price_vals[j-1] > 0
        ]
        vol_7d = float(np.std(rets_window)) if len(rets_window) > 1 else 0.01

        unrealized_pnl = (curr_price / prev_price - 1) * prev_position

        state = build_state_vector(
            direction=direction,
            confidence=confidence,
            ret_1d=ret_1d,
            ret_7d=ret_7d,
            ret_30d=ret_30d,
            vol_7d=vol_7d,
            current_position=prev_position,
            unrealized_pnl=unrealized_pnl,
            past_directions=past_directions,
            past_returns=past_returns,
        )

        # Bandit decision
        arm, multiplier = decide(bandit, state, direction, training=training)

        # Position: multiplier × direction × max_alloc
        direction_sign = {"long": 1.0, "short": -1.0, "hold": 0.0}.get(direction, 0.0)
        position = multiplier * direction_sign
        alloc = position * MAX_ALLOCATION

        # P&L and transaction cost
        pnl = portfolio * alloc * ret_1d
        tx_cost = abs(alloc - prev_position * MAX_ALLOCATION) * TX_COST_RATE * portfolio
        portfolio = max(portfolio + pnl - tx_cost, 0.0)

        # Reward (for training update)
        # Forward return approximated as current ret_1d (we observe it just after decision)
        reward = compute_reward(multiplier, direction, ret_1d, vol_7d, prev_position)
        if training:
            bandit.update(arm, state, reward)

        # Track history for hit-rate
        past_directions.append(direction)
        past_returns.append(ret_1d)

        records.append({
            "date": date, "coin": coin, "training": training,
            "portfolio_value": portfolio,
            "daily_return": ret_1d,
            "action_arm": arm,
            "position_multiplier": multiplier,
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "forward_return": ret_1d,
            "reward": reward,
        })

        prev_position = position

    return pd.DataFrame(records), past_directions, past_returns


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    coin: str,
    train_days: int = TRAIN_WINDOW_DAYS,
    eval_days: int = EVAL_WINDOW_DAYS,
    start_portfolio: float = INITIAL_PORTFOLIO,
) -> dict:
    """
    Full walk-forward backtest for a single coin.

    Returns
    -------
    dict with keys:
      "bandit_results" : DataFrame of eval-window bandit steps
      "bah_results"    : DataFrame of eval-window buy-and-hold steps
      "llm_results"    : DataFrame of eval-window LLM-only steps
      "oracle_returns" : list of oracle (best-action) returns per eval step
      "windows"        : list of (train_start, train_end, eval_start, eval_end)
    """
    coin_prices = (
        prices_df[prices_df["coin"] == coin]
        .sort_values("date")
        .set_index("date")["close"]
    )
    coin_signals = signals_df[signals_df["coin"] == coin].copy()
    if "date" in coin_signals.columns:
        coin_signals["date"] = pd.to_datetime(coin_signals["date"])
        coin_signals = coin_signals.set_index("date")

    all_dates = coin_prices.index.sort_values()
    n = len(all_dates)

    bandit_eval_records = []
    bah_eval_records = []
    llm_eval_records = []
    oracle_returns = []
    windows = []

    fold = 0
    i = 0
    while i + train_days + eval_days <= n:
        train_start = all_dates[i]
        train_end = all_dates[i + train_days - 1]
        eval_start = all_dates[i + train_days]
        eval_end = all_dates[min(i + train_days + eval_days - 1, n - 1)]

        train_prices = coin_prices[train_start:train_end]
        eval_prices = coin_prices[eval_start:eval_end]

        train_signals = coin_signals[coin_signals.index.isin(train_prices.index)].reset_index()
        eval_signals = coin_signals[coin_signals.index.isin(eval_prices.index)].reset_index()

        logger.info(
            "Fold %d: train %s→%s (%d days), eval %s→%s (%d days)",
            fold, train_start.date(), train_end.date(), len(train_prices),
            eval_start.date(), eval_end.date(), len(eval_prices),
        )

        # Fresh bandit per fold (walk-forward: no leakage)
        bandit = LinUCBBandit()
        past_directions: list = []
        past_returns: list = []

        # --- Training phase ---
        _, past_directions, past_returns = run_bandit_on_window(
            bandit=bandit,
            coin=coin,
            prices=train_prices,
            signals=train_signals,
            past_directions=past_directions,
            past_returns=past_returns,
            start_portfolio=start_portfolio,
            training=True,
        )

        # --- Evaluation phase ---
        eval_df, past_directions, past_returns = run_bandit_on_window(
            bandit=bandit,
            coin=coin,
            prices=eval_prices,
            signals=eval_signals,
            past_directions=past_directions,
            past_returns=past_returns,
            start_portfolio=start_portfolio,
            training=False,
        )
        eval_df["fold"] = fold
        bandit_eval_records.append(eval_df)

        # --- Baselines on same eval period ---
        bah = run_buy_and_hold(eval_prices, start_portfolio)
        bah["fold"] = fold
        bah["coin"] = coin
        bah_eval_records.append(bah)

        llm = run_llm_only(
            eval_prices,
            eval_signals["direction"].set_axis(eval_prices.index) if len(eval_signals) == len(eval_prices) else
            eval_signals.set_index("date")["direction"].reindex(eval_prices.index).fillna("hold"),
            start_portfolio,
        )
        llm["fold"] = fold
        llm["coin"] = coin
        llm_eval_records.append(llm)

        # --- Oracle: perfect-foresight best action ---
        price_vals = eval_prices.values
        for j in range(1, len(price_vals)):
            fwd_ret = price_vals[j] / price_vals[j-1] - 1
            best_mult = max(ACTIONS, key=lambda m: m * fwd_ret)
            oracle_returns.append(best_mult * abs(fwd_ret) * MAX_ALLOCATION)

        windows.append((
            str(train_start.date()), str(train_end.date()),
            str(eval_start.date()), str(eval_end.date()),
        ))
        fold += 1
        i += eval_days   # roll forward by eval window

    return {
        "bandit_results": pd.concat(bandit_eval_records, ignore_index=True) if bandit_eval_records else pd.DataFrame(),
        "bah_results": pd.concat(bah_eval_records, ignore_index=True) if bah_eval_records else pd.DataFrame(),
        "llm_results": pd.concat(llm_eval_records, ignore_index=True) if llm_eval_records else pd.DataFrame(),
        "oracle_returns": oracle_returns,
        "windows": windows,
    }


def run_full_backtest(prices_df: pd.DataFrame, signals_df: pd.DataFrame) -> dict:
    """
    Run walk-forward backtest for all coins. Saves results to JSON.
    """
    from data_ingest import COINS, COIN_SYMBOLS
    all_results = {}

    for coin in COINS:
        if coin not in prices_df["coin"].unique():
            logger.warning("Coin %s not found in prices, skipping", coin)
            continue
        logger.info("=" * 60)
        logger.info("Running walk-forward for %s (%s)", coin, COIN_SYMBOLS[coin])
        result = run_walk_forward(prices_df, signals_df, coin)
        all_results[coin] = {
            k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v
            for k, v in result.items()
        }

    # Persist
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, default=str)
    logger.info("Backtest results saved → %s", RESULTS_PATH)
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_ingest import run_ingest

    prices, sentiment = run_ingest()
    # Load precomputed signals (must run llm_signal.py first)
    sig_path = BASE_DIR / "data" / "processed" / "signals.csv"
    if not sig_path.exists():
        logger.error("signals.csv not found. Run: python llm_signal.py first")
        sys.exit(1)
    signals = pd.read_csv(sig_path, parse_dates=["date"])
    results = run_full_backtest(prices, signals)
    logger.info("Walk-forward backtest complete.")
