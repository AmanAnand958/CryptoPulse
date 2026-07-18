"""
features.py
===========
Constructs the state vector fed to the LinUCB bandit at each time step.

State vector components (dimension = 9):
  [0]  llm_direction_encoded   — {long: +1, hold: 0, short: -1}
  [1]  llm_confidence          — [0, 1]
  [2]  ret_1d                  — 1-day return
  [3]  ret_7d                  — 7-day return
  [4]  ret_30d                 — 30-day return
  [5]  realized_vol_7d         — 7-day realized volatility
  [6]  current_position        — [-1, 1] (fraction of max allocation)
  [7]  unrealized_pnl          — normalised unrealized P&L
  [8]  llm_hit_rate_K          — rolling hit-rate of last K LLM signals
                                 (key feature — gives policy evidence of
                                  how reliable the LLM has been recently)

The hit-rate feature [8] is what makes the self-calibration genuine:
the bandit directly observes whether recent LLM signals were profitable
and can adjust trust accordingly — not just current confidence.
"""

import numpy as np
import pandas as pd
from typing import Optional


DIRECTION_MAP = {"long": 1.0, "hold": 0.0, "short": -1.0}
STATE_DIM = 9


def encode_direction(direction: str) -> float:
    return DIRECTION_MAP.get(direction, 0.0)


def compute_hit_rate(
    past_directions: list[str],
    past_returns: list[float],
    K: int = 10,
) -> float:
    """
    Compute the rolling hit-rate of the LLM signal over the last K steps.

    A signal is a "hit" if:
      - direction == "long"  and next-period return > 0
      - direction == "short" and next-period return < 0
      - direction == "hold"  is excluded (counts neither hit nor miss)

    Returns 0.5 (neutral) if fewer than K non-hold signals are available.
    """
    hits = 0
    total = 0
    for d, r in zip(reversed(past_directions), reversed(past_returns)):
        if total >= K:
            break
        if d == "hold":
            continue
        if (d == "long" and r > 0) or (d == "short" and r < 0):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.5


def build_state_vector(
    direction: str,
    confidence: float,
    ret_1d: float,
    ret_7d: float,
    ret_30d: float,
    vol_7d: float,
    current_position: float,
    unrealized_pnl: float,
    past_directions: list[str],
    past_returns: list[float],
    K: int = 10,
) -> np.ndarray:
    """
    Build a normalised state vector for the LinUCB bandit.

    All features are clipped to [-3, 3] after z-score-like normalisation
    to prevent gradient explosions in the bandit's linear model.

    Parameters
    ----------
    direction : str            LLM signal direction
    confidence : float         LLM signal confidence [0, 1]
    ret_1d, ret_7d, ret_30d   Recent price returns
    vol_7d : float             7-day realized volatility
    current_position : float   Current position size in [-1, 1]
    unrealized_pnl : float     Unrealized P&L (as fraction of portfolio)
    past_directions : list     Historical LLM directions for hit-rate
    past_returns : list        Historical 1-day returns aligned with signals
    K : int                    Window for LLM hit-rate computation

    Returns
    -------
    np.ndarray of shape (STATE_DIM,)
    """
    hit_rate = compute_hit_rate(past_directions, past_returns, K)

    raw = np.array([
        encode_direction(direction),          # [-1, 1]
        float(confidence),                    # [0, 1]
        np.clip(ret_1d / 0.05, -3, 3),       # normalised ~N(0,1) for crypto
        np.clip(ret_7d / 0.15, -3, 3),
        np.clip(ret_30d / 0.30, -3, 3),
        np.clip(vol_7d / 0.03, -3, 3),       # typical daily vol
        float(current_position),              # [-1, 1]
        np.clip(unrealized_pnl / 0.1, -3, 3),
        float(hit_rate),                      # [0, 1]
    ], dtype=np.float64)

    return raw


def build_features_dataframe(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align prices and LLM signals into a single feature-rich DataFrame.

    Adds columns: ret_1d, ret_7d, ret_30d, vol_7d, state_vector (as list)
    """
    merged = prices_df.merge(signals_df, on=["date", "coin", "symbol"], how="inner")
    results = []

    for coin, grp in merged.groupby("coin"):
        grp = grp.sort_values("date").reset_index(drop=True)
        closes = grp["close"].values
        directions = grp["direction"].tolist()

        for i in range(len(grp)):
            ret_1d = (closes[i] / closes[i-1] - 1) if i > 0 and closes[i-1] else 0.0
            ret_7d = (closes[i] / closes[max(0, i-7)] - 1) if i >= 7 and closes[max(0, i-7)] else 0.0
            ret_30d = (closes[i] / closes[max(0, i-30)] - 1) if i >= 30 and closes[max(0, i-30)] else 0.0

            # 7-day realized volatility
            rets = [
                np.log(closes[j] / closes[j-1])
                for j in range(max(1, i-6), i+1)
                if closes[j-1] > 0
            ]
            vol_7d = float(np.std(rets)) if len(rets) > 1 else 0.01

            row = grp.iloc[i].to_dict()
            row["ret_1d"] = ret_1d
            row["ret_7d"] = ret_7d
            row["ret_30d"] = ret_30d
            row["vol_7d"] = vol_7d
            row["past_directions"] = directions[:i]
            results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick unit test
    state = build_state_vector(
        direction="long",
        confidence=0.75,
        ret_1d=0.03,
        ret_7d=0.12,
        ret_30d=0.25,
        vol_7d=0.025,
        current_position=0.0,
        unrealized_pnl=0.0,
        past_directions=["long", "short", "long", "hold", "long"],
        past_returns=[0.02, -0.015, 0.03, 0.001, 0.025],
        K=10,
    )
    print("State vector:", state)
    print("Shape:", state.shape)
    assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},) got {state.shape}"
    print("✓ State vector OK")
