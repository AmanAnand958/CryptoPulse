"""
baselines.py
============
Implements comparison baselines on the exact same walk-forward periods.

Baselines:
  1. Buy-and-hold  — buy at start of eval window, hold till end
  2. LLM-only (fixed-size) — always act on raw LLM signal at fixed 20%
     allocation, no bandit adjustment

The bandit vs. LLM-only comparison is the single most important result.
Both baselines are computed on identical dates/periods as the bandit.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MAX_ALLOCATION = 0.20   # 20% of portfolio per position
TX_COST_RATE = 0.001    # 0.1% per unit position change


def run_buy_and_hold(
    prices: pd.Series,
    start_portfolio: float = 10_000.0,
) -> pd.DataFrame:
    """
    Buy at first day's close, hold to end of window.

    Parameters
    ----------
    prices : pd.Series  — indexed by date, daily close prices
    start_portfolio : float — starting portfolio value

    Returns
    -------
    DataFrame with columns: [date, portfolio_value, daily_return]
    """
    prices = prices.sort_index()
    initial_price = prices.iloc[0]
    units = (start_portfolio * MAX_ALLOCATION) / initial_price

    records = []
    cash = start_portfolio * (1 - MAX_ALLOCATION)
    for date, price in prices.items():
        portfolio_value = cash + units * price
        records.append({"date": date, "portfolio_value": portfolio_value})

    df = pd.DataFrame(records)
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    return df


def run_llm_only(
    prices: pd.Series,
    signals: pd.Series,   # pd.Series indexed by date, values = "long"|"short"|"hold"
    start_portfolio: float = 10_000.0,
) -> pd.DataFrame:
    """
    Always act on LLM signal at fixed MAX_ALLOCATION position size.
    Position is rebalanced daily based on signal direction.

    Parameters
    ----------
    prices : pd.Series   — indexed by date, daily close prices
    signals : pd.Series  — indexed by date, direction strings
    start_portfolio : float

    Returns
    -------
    DataFrame with columns: [date, portfolio_value, daily_return, signal, position]
    """
    prices = prices.sort_index()
    signals = signals.reindex(prices.index).fillna("hold")

    portfolio = start_portfolio
    prev_position = 0.0
    records = []

    for i, (date, price) in enumerate(prices.items()):
        if i == 0:
            records.append({
                "date": date, "portfolio_value": portfolio,
                "daily_return": 0.0, "signal": "hold", "position": 0.0,
            })
            continue

        direction = signals.iloc[i]
        if direction == "long":
            position = MAX_ALLOCATION
        elif direction == "short":
            position = -MAX_ALLOCATION
        else:
            position = 0.0

        prev_price = prices.iloc[i - 1]
        price_return = (price / prev_price - 1) if prev_price > 0 else 0.0

        # Gross P&L from position
        pnl = portfolio * position * price_return
        # Transaction cost from position change
        tx_cost = abs(position - prev_position) * TX_COST_RATE * portfolio

        portfolio = portfolio + pnl - tx_cost
        portfolio = max(portfolio, 0.0)  # floor at 0

        records.append({
            "date": date,
            "portfolio_value": portfolio,
            "daily_return": (portfolio / records[-1]["portfolio_value"]) - 1,
            "signal": direction,
            "position": position,
        })
        prev_position = position

    return pd.DataFrame(records)


def run_baselines_on_window(
    prices_df: pd.DataFrame,    # full prices DF with date, coin, close
    signals_df: pd.DataFrame,   # full signals DF with date, coin, direction
    coin: str,
    window_dates: pd.DatetimeIndex,
    start_portfolio: float = 10_000.0,
) -> dict:
    """
    Run both baselines for a single coin over a specific eval window.

    Returns
    -------
    dict with keys "buy_and_hold" and "llm_only", each a DataFrame
    """
    coin_prices = (
        prices_df[prices_df["coin"] == coin]
        .set_index("date")["close"]
        .loc[window_dates]
        .dropna()
    )
    coin_signals = (
        signals_df[signals_df["coin"] == coin]
        .set_index("date")["direction"]
        .reindex(coin_prices.index)
        .fillna("hold")
    )

    return {
        "buy_and_hold": run_buy_and_hold(coin_prices, start_portfolio),
        "llm_only": run_llm_only(coin_prices, coin_signals, start_portfolio),
    }
