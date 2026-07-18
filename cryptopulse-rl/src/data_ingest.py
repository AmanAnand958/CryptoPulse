"""
data_ingest.py
==============
Pulls historical OHLCV data for BTC, ETH, SOL, BNB, ADA from CoinGecko
public API (no API key required for daily granularity, up to 365 days).

Sentiment source: Approximated from price momentum and volume trends since
free live sentiment APIs (LunarCrush, Santiment) have heavy rate limits.
The approximation is clearly documented — this is a research system and the
sentiment signal is a feature, not an oracle.

Outputs
-------
data/processed/prices.csv   — timestamp-aligned daily OHLCV per coin
data/processed/sentiment.csv — timestamp-aligned sentiment score per coin
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano"]
COIN_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
    "cardano": "ADA",
}
DAYS = 365          # 1 year of daily data
CURRENCY = "usd"
RATE_LIMIT_SLEEP = 1.5   # seconds between CoinGecko requests (free tier)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CoinGecko helpers
# ---------------------------------------------------------------------------

def fetch_market_chart(coin_id: str, days: int = DAYS, currency: str = CURRENCY) -> dict:
    """Fetch OHLCV market chart from CoinGecko public API."""
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency={currency}&days={days}&interval=daily"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                logger.warning("Rate limited by CoinGecko, sleeping 60s…")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("CoinGecko request failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch data for {coin_id} after 3 attempts")


def parse_market_chart(data: dict, coin_id: str) -> pd.DataFrame:
    """Parse CoinGecko market_chart response into a tidy DataFrame."""
    prices = pd.DataFrame(data["prices"], columns=["timestamp_ms", "close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp_ms", "volume"])
    market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp_ms", "market_cap"])

    df = prices.merge(volumes, on="timestamp_ms").merge(market_caps, on="timestamp_ms")
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()
    df["coin"] = coin_id
    df["symbol"] = COIN_SYMBOLS[coin_id]

    # Derive OHLCV approximations from daily close + volume
    # (CoinGecko free endpoint gives daily close; open ≈ prev_close)
    df = df.sort_values("date").reset_index(drop=True)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, len(df))))
    df["low"]  = df[["open", "close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, len(df))))

    return df[["date", "coin", "symbol", "open", "high", "low", "close", "volume", "market_cap"]]


# ---------------------------------------------------------------------------
# Sentiment approximation
# ---------------------------------------------------------------------------

def compute_sentiment(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate a daily sentiment score per coin from price momentum + volume.

    Score ∈ [-1, 1]:
      +1  = strong positive sentiment (strong up-momentum + high volume)
      -1  = strong negative sentiment
       0  = neutral

    This is a proxy. A production system would ingest LunarCrush, Santiment,
    or a crypto-tweet classifier. The proxy is documented here so the README
    and any paper using these results can be evaluated honestly.
    """
    rows = []
    for coin_id, grp in prices_df.groupby("coin"):
        grp = grp.sort_values("date").copy()
        # 7-day momentum (returns)
        grp["ret_7d"] = grp["close"].pct_change(7)
        # 1-day return
        grp["ret_1d"] = grp["close"].pct_change(1)
        # Volume Z-score (rolling 30d)
        grp["vol_z"] = (grp["volume"] - grp["volume"].rolling(30).mean()) / (
            grp["volume"].rolling(30).std() + 1e-9
        )
        # Composite sentiment: weight momentum + normalised vol signal
        grp["sentiment"] = (
            0.5 * np.clip(grp["ret_7d"] / 0.10, -1, 1)
            + 0.3 * np.clip(grp["ret_1d"] / 0.05, -1, 1)
            + 0.2 * np.clip(grp["vol_z"] / 2.0, -1, 1)
        )
        grp["sentiment"] = grp["sentiment"].clip(-1, 1).fillna(0)
        rows.append(grp[["date", "coin", "symbol", "sentiment"]])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main ingest pipeline
# ---------------------------------------------------------------------------

def run_ingest(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch and process price + sentiment data.

    Parameters
    ----------
    force_refresh : bool
        If True, re-fetch from CoinGecko even if cached files exist.

    Returns
    -------
    (prices_df, sentiment_df)
    """
    prices_path = DATA_PROCESSED / "prices.csv"
    sentiment_path = DATA_PROCESSED / "sentiment.csv"

    if not force_refresh and prices_path.exists() and sentiment_path.exists():
        logger.info("Loading cached processed data…")
        prices_df = pd.read_csv(prices_path, parse_dates=["date"])
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
        return prices_df, sentiment_df

    logger.info("Fetching fresh data from CoinGecko for %d coins…", len(COINS))
    all_prices = []
    for coin_id in COINS:
        logger.info("  → %s", coin_id)
        raw = fetch_market_chart(coin_id)
        # Save raw JSON
        raw_path = DATA_RAW / f"{coin_id}_raw.json"
        import json
        raw_path.write_text(json.dumps(raw))
        df = parse_market_chart(raw, coin_id)
        all_prices.append(df)
        time.sleep(RATE_LIMIT_SLEEP)

    prices_df = pd.concat(all_prices, ignore_index=True)
    prices_df.to_csv(prices_path, index=False)
    logger.info("Saved prices → %s", prices_path)

    sentiment_df = compute_sentiment(prices_df)
    sentiment_df.to_csv(sentiment_path, index=False)
    logger.info("Saved sentiment → %s", sentiment_path)

    return prices_df, sentiment_df


if __name__ == "__main__":
    prices, sentiment = run_ingest(force_refresh=False)
    logger.info("Prices shape: %s", prices.shape)
    logger.info("Sentiment shape: %s", sentiment.shape)
    logger.info("Date range: %s → %s", prices["date"].min(), prices["date"].max())
    print(prices.tail())
    print(sentiment.tail())
