"""
meta_label_baseline.py
======================
Meta-labeling classifier baseline for CryptoPulse RL.

WHAT IS META-LABELING? (López de Prado, "Advances in Financial ML", ch. 3)
  Rather than predicting direction directly, a meta-labeling classifier
  predicts whether to ACT on an existing signal (here: the LLM signal).
  The classifier takes the same features available to the bandit and outputs
  a binary decision: 1 = trade (follow the LLM signal), 0 = abstain.

WHY ADD THIS BASELINE?
  This baseline has NO live-exploration risk (it is purely supervised).
  Including it in the same backtest suite gives a clean answer to the
  question: "why use RL (the bandit) over a supervised classifier?"
  If the meta-labeling baseline beats the bandit, that is an important
  finding — it means exploration risk is not being compensated for.

ARCHITECTURE:
  - Primary model: LogisticRegression (interpretable, fast)
  - Also trains RandomForestClassifier for comparison
  - Features: same 9-dimensional state vector used by the bandit (features.py)
  - Label: 1 if |llm_direction return| > 0 AND direction was correct, else 0
    (i.e., the LLM signal was worth acting on)
  - Walk-forward: trained only on data before the eval window — same fold
    structure as the bandit for apples-to-apples comparison.

USAGE:
  Called from backtest.py — run_meta_label_baseline(train_df, eval_df)
  Returns a DataFrame of portfolio values for the eval window.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import build_state_vector

logger = logging.getLogger(__name__)

MAX_ALLOCATION = 0.20
TX_COST_RATE = 0.001


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _build_meta_label(direction: str, forward_return: float) -> int:
    """
    Meta-label: 1 if acting on the LLM signal would have been profitable.

    Rules:
      - direction="long"/"BUY" + forward_return > 0 → 1 (act)
      - direction="short"/"SELL" + forward_return < 0 → 1 (act)
      - direction="hold"/"HOLD" → 0 (abstain — HOLD never generates a trade)
      - anything else → 0 (don't act)
    """
    d = direction.lower()
    if d in ("long", "buy") and forward_return > 0:
        return 1
    if d in ("short", "sell") and forward_return < 0:
        return 1
    return 0


def build_meta_label_dataset(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    coin: str,
) -> pd.DataFrame:
    """
    Build a feature + label dataset for the meta-labeling classifier.

    Parameters
    ----------
    prices_df : DataFrame with [date, coin, close]
    signals_df : DataFrame with [date, coin, direction, confidence]
    coin : str

    Returns
    -------
    DataFrame with columns: [date, features (list), label, direction, forward_return]
    Point-in-time: forward_return is the NEXT day's return (what was realized
    after the signal was generated), not the current day's return.
    """
    coin_prices = (
        prices_df[prices_df["coin"] == coin]
        .sort_values("date")
        .reset_index(drop=True)
    )
    coin_prices["date"] = pd.to_datetime(coin_prices["date"])

    coin_signals = (
        signals_df[signals_df["coin"] == coin]
        .sort_values("date")
        .reset_index(drop=True)
    )
    coin_signals["date"] = pd.to_datetime(coin_signals["date"])

    closes = coin_prices["close"].tolist()
    price_dates = coin_prices["date"].tolist()

    sig_map = coin_signals.set_index("date")

    rows = []
    past_directions: list = []
    past_returns: list = []

    for i in range(1, len(coin_prices) - 1):  # -1 to have a forward return
        date = price_dates[i]

        # Forward return: return realized the day AFTER the signal
        forward_return = (closes[i + 1] / closes[i] - 1) if closes[i] > 0 else 0.0

        # Returns for feature construction
        ret_1d = (closes[i] / closes[i - 1] - 1) if closes[i - 1] > 0 else 0.0
        ret_7d = (closes[i] / closes[max(0, i - 7)] - 1) if i >= 7 and closes[max(0, i - 7)] > 0 else 0.0
        ret_30d = (closes[i] / closes[max(0, i - 30)] - 1) if i >= 30 and closes[max(0, i - 30)] > 0 else 0.0
        rets_7d = [
            np.log(closes[j] / closes[j - 1])
            for j in range(max(1, i - 6), i + 1)
            if closes[j - 1] > 0
        ]
        vol_7d = float(np.std(rets_7d)) if len(rets_7d) > 1 else 0.01

        # Signal for this date
        if date in sig_map.index:
            sig_row = sig_map.loc[date]
            if isinstance(sig_row, pd.DataFrame):
                sig_row = sig_row.iloc[0]
            direction = str(sig_row.get("direction", sig_row.get("action", "hold")))
            confidence = float(sig_row.get("confidence", 0.0))
        else:
            direction, confidence = "hold", 0.0

        # State vector (same as bandit)
        state = build_state_vector(
            direction=direction,
            confidence=confidence,
            ret_1d=ret_1d,
            ret_7d=ret_7d,
            ret_30d=ret_30d,
            vol_7d=vol_7d,
            current_position=0.0,
            unrealized_pnl=0.0,
            past_directions=past_directions,
            past_returns=past_returns,
        )

        # Meta-label: was acting on this signal profitable?
        label = _build_meta_label(direction, forward_return)

        rows.append({
            "date": date,
            "features": state.tolist(),
            "label": label,
            "direction": direction,
            "forward_return": forward_return,
            "confidence": confidence,
        })

        past_directions.append(direction)
        past_returns.append(ret_1d)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier training + inference
# ---------------------------------------------------------------------------

def train_meta_labeler(
    train_df: pd.DataFrame,
    model_type: str = "logistic",
) -> Pipeline:
    """
    Train a meta-labeling classifier on the training window.

    Parameters
    ----------
    train_df : DataFrame from build_meta_label_dataset()
    model_type : "logistic" or "random_forest"

    Returns
    -------
    Fitted sklearn Pipeline (StandardScaler + Classifier)
    """
    X = np.vstack(train_df["features"].values)
    y = train_df["label"].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info(
        "Meta-labeler training: %d samples, %d positive (act/profitable), %d negative (abstain)",
        len(y), n_pos, n_neg,
    )
    if n_pos < 5:
        logger.warning(
            "⚠ Meta-labeler fold has only %d positive training samples out of %d. "
            "Model is heavily sample-constrained and may underperform raw LLM signal due to extreme class imbalance.",
            n_pos, len(y)
        )

    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        )
    else:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    if len(np.unique(y)) < 2:
        logger.warning("Only one class in training data — meta-labeler will always predict %d", y[0])

    pipe.fit(X, y)
    train_acc = pipe.score(X, y)
    logger.info("Meta-labeler training accuracy: %.3f (in-sample, for sanity only)", train_acc)

    return pipe


def run_meta_label_baseline(
    train_prices: pd.Series,
    eval_prices: pd.Series,
    train_signals: pd.DataFrame,
    eval_signals: pd.DataFrame,
    coin: str,
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    start_portfolio: float = 10_000.0,
    model_type: str = "logistic",
) -> pd.DataFrame:
    """
    Run the meta-labeling baseline on one eval window.

    Train classifier on active LLM signals (long/short only) for dates <= train_end.
    Pools active signals across all coins up to train_end to achieve a viable sample size,
    strictly respecting point-in-time boundaries (no future data leakage).

    When classifier predicts 1 (act): trade at MAX_ALLOCATION.
    When classifier predicts 0 (abstain): HOLD.
    When primary signal is HOLD: position = 0 (meta-labeler is not invoked).

    Returns
    -------
    DataFrame with columns: [date, portfolio_value, daily_return, signal, meta_label, position]
    """
    train_end = train_prices.index.max()
    eval_dates = set(eval_prices.index)

    # Cross-coin pooling up to train_end (point-in-time: zero leakage)
    prices_past = prices_df[prices_df["date"] <= train_end].copy()
    signals_past = signals_df[signals_df["date"] <= train_end].copy()

    # Rebuild meta-label dataset for past data across all coins
    all_past_meta = []
    for c in prices_past["coin"].unique():
        c_meta = build_meta_label_dataset(prices_past, signals_past, c)
        all_past_meta.append(c_meta)
    
    train_meta_df = pd.concat(all_past_meta, ignore_index=True) if all_past_meta else pd.DataFrame()

    # Filter training data to ACTIVE signals ONLY (direction != hold)
    # Meta-labeling only applies to filtering active signals, not non-signal hold days!
    train_active_rows = train_meta_df[
        train_meta_df["direction"].str.lower().isin(["long", "buy", "short", "sell"])
    ]

    if len(train_active_rows) < 5:
        logger.warning(
            "Insufficient active LLM signals for meta-labeler (%d rows across all coins) — using raw LLM signal fallback",
            len(train_active_rows),
        )
        pipe = None
    else:
        pipe = train_meta_labeler(train_active_rows, model_type=model_type)

    # Build eval dataset for prediction (current coin only)
    eval_slice = prices_df[
        (prices_df["coin"] == coin) & (prices_df["date"].isin(eval_dates))
    ].copy()
    eval_sig_slice = signals_df[
        (signals_df["coin"] == coin) & (signals_df["date"].isin(eval_dates))
    ].copy()

    # Combined slice for eval rolling features
    combined_dates = set(train_prices.index) | eval_dates
    prices_comb = prices_df[
        (prices_df["coin"] == coin) & (prices_df["date"].isin(combined_dates))
    ].copy()
    signals_comb = signals_df[
        (signals_df["coin"] == coin) & (signals_df["date"].isin(combined_dates))
    ].copy()

    full_eval_meta = build_meta_label_dataset(prices_comb, signals_comb, coin)
    eval_rows = full_eval_meta[full_eval_meta["date"].isin(eval_dates)].set_index("date")

    # Simulate portfolio
    portfolio = start_portfolio
    prev_position = 0.0
    records = []

    for i, (date, price) in enumerate(eval_prices.items()):
        if i == 0:
            records.append({
                "date": date,
                "portfolio_value": portfolio,
                "daily_return": 0.0,
                "signal": "hold",
                "meta_label": 0,
                "position": 0.0,
            })
            continue

        prev_price = eval_prices.iloc[i - 1]
        price_return = (price / prev_price - 1) if prev_price > 0 else 0.0

        # Get signal for this date
        if date in eval_rows.index:
            row = eval_rows.loc[date]
            direction = row.get("direction", "hold") if hasattr(row, "get") else row["direction"]
            features = np.array(row["features"]).reshape(1, -1)
        else:
            direction, features = "hold", None

        # Meta-label prediction
        if pipe is not None and features is not None:
            try:
                meta_label = int(pipe.predict(features)[0])
            except Exception:
                meta_label = 0
        else:
            meta_label = 1 if direction.lower() not in ("hold", "HOLD") else 0

        # Position
        if meta_label == 1 and direction.lower() in ("long", "buy"):
            position = MAX_ALLOCATION
        elif meta_label == 1 and direction.lower() in ("short", "sell"):
            position = -MAX_ALLOCATION
        else:
            position = 0.0

        pnl = portfolio * position * price_return
        tx_cost = abs(position - prev_position) * TX_COST_RATE * portfolio
        portfolio = max(portfolio + pnl - tx_cost, 0.0)

        records.append({
            "date": date,
            "portfolio_value": portfolio,
            "daily_return": price_return,
            "signal": direction,
            "meta_label": meta_label,
            "position": position,
        })
        prev_position = position

    df = pd.DataFrame(records)
    meta_act_count = int((df["meta_label"] == 1).sum())
    llm_nonhold_count = int((df["signal"].str.lower().isin(["long", "buy", "short", "sell"])).sum())
    logger.info(
        "Meta-label baseline: %s eval=%d steps, trades=%d (meta=1), llm_nonhold=%d",
        coin, len(df), meta_act_count, llm_nonhold_count,
    )
    if meta_act_count == llm_nonhold_count and llm_nonhold_count > 0:
        logger.warning(
            "⚠ %s: meta_label=1 count (%d) equals llm_nonhold count (%d) — "
            "classifier may not be filtering signals. Check feature leakage.",
            coin, meta_act_count, llm_nonhold_count,
        )
    return df



if __name__ == "__main__":
    """Quick smoke test with synthetic data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Synthetic data
    n = 100
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = 40000.0 * np.cumprod(1 + np.random.normal(0, 0.02, n))
    directions = np.random.choice(["long", "short", "hold"], n)
    confidences = np.random.uniform(0.3, 0.9, n)

    prices_df = pd.DataFrame({"date": dates, "coin": "bitcoin", "close": closes})
    signals_df = pd.DataFrame({
        "date": dates,
        "coin": "bitcoin",
        "direction": directions,
        "confidence": confidences,
    })

    meta_df = build_meta_label_dataset(prices_df, signals_df, "bitcoin")
    print(f"Meta-label dataset: {len(meta_df)} rows")
    print(f"Label distribution: {meta_df['label'].value_counts().to_dict()}")

    pipe = train_meta_labeler(meta_df)
    print(f"Classifier trained: {type(pipe.named_steps['classifier']).__name__}")
    print("✓ Meta-labeling baseline OK")
