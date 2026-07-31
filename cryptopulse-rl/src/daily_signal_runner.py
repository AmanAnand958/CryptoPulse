#!/usr/bin/env python3
"""
daily_signal_runner.py
======================
Production daily signal generation script.

Run this every day at market close (e.g. via cron or scheduler):
    python src/daily_signal_runner.py

What it does:
  1. Loads the latest prices and sentiment data (via data_ingest)
  2. Determines which (coin, date) pairs are NOT yet in signal_cache.json
  3. Generates LLM signals ONLY for missing dates (no re-calls for cached ones)
  4. Saves updated cache → data/processed/signal_cache.json
  5. Rebuilds signals.csv from the full cache
  6. Logs a summary of what was generated vs. what was already cached

Usage:
    python src/daily_signal_runner.py                     # generate for today only
    python src/daily_signal_runner.py --backfill           # fill ALL missing historical dates
    python src/daily_signal_runner.py --dry-run            # show what would be generated, don't call API
    python src/daily_signal_runner.py --date 2026-07-31   # force-generate for a specific date
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/daily_runner.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_PATH = BASE_DIR / "data" / "processed" / "signal_cache.json"
SIGNALS_CSV_PATH = BASE_DIR / "data" / "processed" / "signals.csv"
COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "cardano"]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info("Cache saved → %s (%d entries)", CACHE_PATH, len(cache))


def rebuild_signals_csv(cache: dict) -> pd.DataFrame:
    """Rebuild signals.csv from the full cache — only real LLM signals."""
    rows = []
    for key, signal in cache.items():
        if "::" not in key:
            continue
        coin, date_str = key.split("::", 1)
        rows.append({
            "date": date_str,
            "coin": coin,
            "direction": signal.get("direction", "hold"),
            "confidence": signal.get("confidence", 0.0),
            "rationale": signal.get("rationale", ""),
        })
    df = pd.DataFrame(rows, columns=["date", "coin", "direction", "confidence", "rationale"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin", "date"]).reset_index(drop=True)
    df.to_csv(SIGNALS_CSV_PATH, index=False)
    logger.info("signals.csv rebuilt → %s (%d rows)", SIGNALS_CSV_PATH, len(df))
    return df


def get_missing_dates(prices_df: pd.DataFrame, cache: dict) -> list[tuple[str, str]]:
    """Return (coin, date_str) pairs not yet in cache."""
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    missing = []
    for _, row in prices_df.iterrows():
        coin = str(row["coin"]).lower()
        date_str = str(row["date"].date())
        key = f"{coin}::{date_str}"
        if key not in cache:
            missing.append((coin, date_str))
    return missing


def main():
    parser = argparse.ArgumentParser(description="CryptoPulse daily signal generator")
    parser.add_argument("--backfill", action="store_true", help="Generate for ALL missing historical dates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without calling API")
    parser.add_argument("--date", type=str, default=None, help="Force-generate for a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CryptoPulse Daily Signal Runner — %s", date.today())
    logger.info("=" * 60)

    from data_ingest import run_ingest
    from llm_signal import generate_signals_for_dataframe

    prices_df, sentiment_df = run_ingest()
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.tz_localize(None)

    cache = load_cache()
    logger.info("Loaded cache: %d existing valid signals", len(cache))

    # Determine which dates to generate
    if args.date:
        target_date = args.date
        rows_to_generate = prices_df[
            prices_df["date"].dt.date == pd.Timestamp(target_date).date()
        ]
        logger.info("Force-generating for date: %s (%d rows)", target_date, len(rows_to_generate))
        # Remove from cache so it's regenerated
        for coin in COINS:
            key = f"{coin}::{target_date}"
            cache.pop(key, None)
    elif args.backfill:
        missing = get_missing_dates(prices_df, cache)
        logger.info("Backfill mode: %d missing (coin, date) pairs", len(missing))
        missing_set = {f"{c}::{d}" for c, d in missing}
        rows_to_generate = prices_df[
            prices_df.apply(
                lambda r: f"{r['coin']}::{str(r['date'].date())}" in missing_set, axis=1
            )
        ]
    else:
        # Default: generate for today only
        today_str = str(date.today())
        rows_to_generate = prices_df[
            prices_df["date"].dt.date == date.today()
        ]
        if rows_to_generate.empty:
            # Fall back to most recent date in dataset
            most_recent = prices_df["date"].max()
            rows_to_generate = prices_df[prices_df["date"] == most_recent]
            logger.info("Today not in dataset; using most recent date: %s", most_recent.date())

    if rows_to_generate.empty:
        logger.info("Nothing to generate — all dates already cached.")
        return

    logger.info("Generating signals for %d rows across %d coins",
                len(rows_to_generate), rows_to_generate["coin"].nunique())

    if args.dry_run:
        logger.info("[DRY RUN] Would generate:")
        for _, row in rows_to_generate.iterrows():
            logger.info("  %s @ %s", row["coin"], row["date"].date())
        logger.info("[DRY RUN] No API calls made.")
        return

    # Generate (use_cache=True so existing entries are not re-called)
    signals_df = generate_signals_for_dataframe(
        rows_to_generate, sentiment_df, use_cache=True
    )

    # Reload the updated cache (generate_signals_for_dataframe saves it)
    cache = load_cache()
    logger.info("Cache after generation: %d entries", len(cache))

    # Rebuild signals.csv
    df = rebuild_signals_csv(cache)
    direction_counts = df["direction"].value_counts().to_dict()
    logger.info("signals.csv direction breakdown: %s", direction_counts)

    logger.info("=" * 60)
    logger.info("Done. Run backtest + evaluate to update metrics:")
    logger.info("  python src/backtest.py && python src/evaluate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
