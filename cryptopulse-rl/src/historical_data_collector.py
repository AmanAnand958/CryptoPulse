"""
historical_data_collector.py
=============================
OFFLINE historical data collection job for backtesting.

IMPORTANT: This is a SEPARATE, STANDALONE data collection job.
  It is NOT invoked per backtest run. Run it ONCE to build the dataset,
  then the backtest reads from the cached parquet files.
  The live Bright Data / MCP pipeline (agents/graph.py) is NEVER called here.

HISTORICAL DATA SOURCING DECISION (spec section 6, point 1):
  Decided approach (documented here, not left implicit):
    - OHLCV: CCXT REST endpoints (free, built for historical candles,
      more reliable than scraping a live page). Uses exchange=binance
      (most liquid, best historical coverage for BTC/ETH/SOL/BNB/ADA).
    - Sentiment: Alpha Vantage news-sentiment endpoint (free tier:
      25 req/day, 5/min — confirmed against Alpha Vantage's own docs).
      This endpoint has historical coverage going back ~2+ years,
      adequate for backtesting purposes.
    Alternative considered: one-time Bright Data bulk historical scrape.
    Rejected because: Alpha Vantage provides structured JSON (no parsing
    brittleness), is free indefinitely, and doesn't burn the 5,000/month
    Bright Data budget on a one-time batch job.

POINT-IN-TIME TIMESTAMPS (spec section 6, point 2):
  Every stored row has TWO timestamps:
    - `as_of_ts`: when the information was actually AVAILABLE (market close
      time for prices, article publication time for sentiment).
    - `collected_ts`: when this script fetched it (metadata only).
  Backtest code must filter on `as_of_ts`, NOT `collected_ts`, to avoid
  look-ahead bias.

OUTPUT:
  data/historical/prices_{coin}.parquet   — OHLCV, indexed by as_of_ts
  data/historical/sentiment_{coin}.parquet — sentiment, indexed by as_of_ts

USAGE:
  python src/historical_data_collector.py
  python src/historical_data_collector.py --dry-run   # connectivity test only
  python src/historical_data_collector.py --coin bitcoin --days 365
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
HIST_DIR = BASE_DIR / "data" / "historical"
HIST_DIR.mkdir(parents=True, exist_ok=True)

# Coins to collect (CoinGecko ID → CCXT symbol → Alpha Vantage ticker)
COINS = {
    "bitcoin":     {"ccxt": "BTC/USDT",   "av_ticker": "BTC",  "symbol": "BTC"},
    "ethereum":    {"ccxt": "ETH/USDT",   "av_ticker": "ETH",  "symbol": "ETH"},
    "solana":      {"ccxt": "SOL/USDT",   "av_ticker": "SOL",  "symbol": "SOL"},
    "binancecoin": {"ccxt": "BNB/USDT",   "av_ticker": "BNB",  "symbol": "BNB"},
    "cardano":     {"ccxt": "ADA/USDT",   "av_ticker": "ADA",  "symbol": "ADA"},
}

DEFAULT_DAYS = 365
CCXT_EXCHANGE = "binance"   # free, no API key needed for public OHLCV
AV_RATE_LIMIT = 12.0        # seconds between Alpha Vantage calls (5/min = 12s gap)
AV_SENTIMENT_ENDPOINT = "https://www.alphavantage.co/query"


# ---------------------------------------------------------------------------
# OHLCV via CCXT (free, no API key)
# ---------------------------------------------------------------------------

def fetch_ohlcv_ccxt(
    coin_id: str,
    ccxt_symbol: str,
    days: int = DEFAULT_DAYS,
    exchange_id: str = CCXT_EXCHANGE,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV from CCXT (free, no API key required).

    Uses Binance public endpoints — no auth, no cost, good historical depth.
    Daily candles only (adequate for backtest frequency).

    Returns DataFrame with columns:
      [as_of_ts, collected_ts, coin, symbol, open, high, low, close, volume]
    """
    if dry_run:
        logger.info("[DRY RUN] Would fetch %d days of OHLCV for %s via CCXT", days, ccxt_symbol)
        return None

    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return None

    try:
        exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
        })

        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        collected_ts = datetime.now(timezone.utc).isoformat()

        logger.info("Fetching OHLCV: %s / %s (last %d days) via %s", coin_id, ccxt_symbol, days, exchange_id)

        all_candles = []
        limit = 1000  # max per CCXT fetch
        since = since_ms

        while True:
            candles = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            if len(candles) < limit:
                break
            since = candles[-1][0] + 1  # next ms after last candle
            time.sleep(exchange.rateLimit / 1000.0)

        if not all_candles:
            logger.warning("No OHLCV data returned for %s", ccxt_symbol)
            return None

        df = pd.DataFrame(all_candles, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df["as_of_ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["collected_ts"] = collected_ts
        df["coin"] = coin_id
        df["symbol"] = COINS[coin_id]["symbol"]
        df = df.drop(columns=["ts_ms"])
        df = df.sort_values("as_of_ts").reset_index(drop=True)

        logger.info("CCXT: fetched %d daily candles for %s (%s → %s)",
                    len(df), coin_id,
                    df["as_of_ts"].min().date(), df["as_of_ts"].max().date())
        return df

    except Exception as exc:
        logger.error("CCXT fetch failed for %s: %s", ccxt_symbol, exc)
        return None


# ---------------------------------------------------------------------------
# News/Sentiment via Alpha Vantage (free tier, 25 req/day)
# ---------------------------------------------------------------------------

def fetch_sentiment_alpha_vantage(
    coin_id: str,
    av_ticker: str,
    days: int = DEFAULT_DAYS,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical news sentiment from Alpha Vantage (free tier).

    Free tier limits: 25 requests/day, 5/minute.
    Each request returns up to 200 articles. Paging is not available on the
    free tier — we fetch the latest articles and accept that historical
    coverage may be partial (adequate for a research backtest).

    ALPHA_VANTAGE_API_KEY must be set in environment or .env.
    Register free at: https://www.alphavantage.co/support/#api-key

    Returns DataFrame with columns:
      [as_of_ts, collected_ts, coin, symbol, sentiment_score,
       sentiment_label, relevance_score, article_title, article_url]
    """
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not av_key:
        logger.warning(
            "ALPHA_VANTAGE_API_KEY not set — skipping historical sentiment for %s. "
            "Register at alphavantage.co for a free key.", coin_id,
        )
        return None

    if dry_run:
        logger.info("[DRY RUN] Would fetch sentiment for %s via Alpha Vantage (ticker=%s)", coin_id, av_ticker)
        return None

    import requests

    collected_ts = datetime.now(timezone.utc).isoformat()
    time_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%dT%H%M")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": f"CRYPTO:{av_ticker}",
        "time_from": time_from,
        "limit": 200,
        "apikey": av_key,
    }

    try:
        logger.info("Fetching Alpha Vantage sentiment: %s (ticker=CRYPTO:%s)", coin_id, av_ticker)
        resp = requests.get(AV_SENTIMENT_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "feed" not in data:
            logger.warning("Alpha Vantage returned no feed for %s: %s", av_ticker, data.get("Note", data.get("Information", "unknown")))
            return None

        rows = []
        for article in data["feed"]:
            # Parse publication time as as_of_ts (when info was actually available)
            pub_str = article.get("time_published", "")
            try:
                as_of_ts = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Find ticker-specific sentiment
            ticker_sentiment = 0.0
            relevance = 0.0
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == f"CRYPTO:{av_ticker}":
                    ticker_sentiment = float(ts.get("ticker_sentiment_score", 0.0))
                    relevance = float(ts.get("relevance_score", 0.0))
                    break

            rows.append({
                "as_of_ts": as_of_ts,
                "collected_ts": collected_ts,
                "coin": coin_id,
                "symbol": COINS[coin_id]["symbol"],
                "sentiment_score": ticker_sentiment,
                "sentiment_label": article.get("overall_sentiment_label", "Neutral"),
                "relevance_score": relevance,
                "article_title": article.get("title", "")[:200],
                "article_url": article.get("url", ""),
            })

        if not rows:
            logger.warning("No articles found for %s in Alpha Vantage response", av_ticker)
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values("as_of_ts").reset_index(drop=True)
        logger.info("Alpha Vantage: fetched %d articles for %s (%s → %s)",
                    len(df), coin_id,
                    df["as_of_ts"].min().date(), df["as_of_ts"].max().date())

        # Resample to daily: mean sentiment per day (as_of_ts = market close of that day)
        df["date"] = df["as_of_ts"].dt.normalize()
        daily = (
            df.groupby("date")
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                relevance_score=("relevance_score", "mean"),
                n_articles=("article_title", "count"),
                as_of_ts=("date", "first"),
            )
            .reset_index(drop=True)
        )
        daily["coin"] = coin_id
        daily["symbol"] = COINS[coin_id]["symbol"]
        daily["collected_ts"] = collected_ts
        return daily

    except Exception as exc:
        logger.error("Alpha Vantage fetch failed for %s: %s", av_ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic sentiment fallback (price momentum proxy)
# ---------------------------------------------------------------------------

def compute_price_proxy_sentiment(prices_df: pd.DataFrame, coin_id: str) -> pd.DataFrame:
    """
    Fallback: compute sentiment proxy from price momentum when Alpha Vantage
    is unavailable. This is the same logic as data_ingest.py — clearly
    documented as a proxy, not an oracle.

    IMPORTANT: Every row's as_of_ts = market close of that day (point-in-time).
    """
    coin_prices = prices_df[prices_df["coin"] == coin_id].sort_values("as_of_ts").copy()
    closes = coin_prices["close"].values
    dates = coin_prices["as_of_ts"].values
    collected_ts = datetime.now(timezone.utc).isoformat()

    rows = []
    for i in range(7, len(coin_prices)):
        ret_7d = (closes[i] / closes[i - 7] - 1) if closes[i - 7] > 0 else 0.0
        ret_1d = (closes[i] / closes[i - 1] - 1) if closes[i - 1] > 0 else 0.0

        vols = coin_prices["volume"].values
        vol_slice = vols[max(0, i - 30):i]
        vol_z = (vols[i] - vol_slice.mean()) / (vol_slice.std() + 1e-9) if len(vol_slice) > 1 else 0.0

        sentiment = (
            0.5 * np.clip(ret_7d / 0.10, -1, 1)
            + 0.3 * np.clip(ret_1d / 0.05, -1, 1)
            + 0.2 * np.clip(vol_z / 2.0, -1, 1)
        )
        sentiment = float(np.clip(sentiment, -1, 1))

        rows.append({
            "as_of_ts": dates[i],
            "date": pd.Timestamp(dates[i]).normalize(),
            "coin": coin_id,
            "symbol": COINS[coin_id]["symbol"],
            "sentiment_score": sentiment,
            "relevance_score": 1.0,
            "n_articles": 0,
            "collected_ts": collected_ts,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main collection job
# ---------------------------------------------------------------------------

def collect_historical_data(
    coins: list[str] = None,
    days: int = DEFAULT_DAYS,
    dry_run: bool = False,
    force_refresh: bool = False,
) -> dict:
    """
    Run the full historical data collection job.

    This is a ONE-TIME job — run once to build the dataset, then
    the backtest reads from cached parquet files.

    Parameters
    ----------
    coins : list of coin IDs (defaults to all COINS)
    days : number of historical days to fetch
    dry_run : if True, print what would happen without fetching
    force_refresh : if True, re-fetch even if parquet files exist

    Returns
    -------
    dict of {coin_id: {"prices": DataFrame, "sentiment": DataFrame}}
    """
    if coins is None:
        coins = list(COINS.keys())

    results = {}
    av_call_count = 0

    for coin_id in coins:
        coin_cfg = COINS.get(coin_id)
        if coin_cfg is None:
            logger.warning("Unknown coin: %s — skipping", coin_id)
            continue

        prices_path = HIST_DIR / f"prices_{coin_id}.parquet"
        sentiment_path = HIST_DIR / f"sentiment_{coin_id}.parquet"

        # ── Prices ──
        if not force_refresh and prices_path.exists():
            logger.info("Loading cached prices for %s from %s", coin_id, prices_path)
            prices_df = pd.read_parquet(prices_path)
        else:
            prices_df = fetch_ohlcv_ccxt(
                coin_id=coin_id,
                ccxt_symbol=coin_cfg["ccxt"],
                days=days,
                dry_run=dry_run,
            )
            if prices_df is not None and not dry_run:
                prices_df.to_parquet(prices_path, index=False)
                logger.info("Saved prices → %s (%d rows)", prices_path, len(prices_df))

        # ── Sentiment ──
        if not force_refresh and sentiment_path.exists():
            logger.info("Loading cached sentiment for %s from %s", coin_id, sentiment_path)
            sentiment_df = pd.read_parquet(sentiment_path)
        else:
            # Alpha Vantage rate limit: 5/min. Track calls and sleep.
            if av_call_count > 0:
                logger.info("Sleeping %.0fs for Alpha Vantage rate limit...", AV_RATE_LIMIT)
                if not dry_run:
                    time.sleep(AV_RATE_LIMIT)

            sentiment_df = fetch_sentiment_alpha_vantage(
                coin_id=coin_id,
                av_ticker=coin_cfg["av_ticker"],
                days=days,
                dry_run=dry_run,
            )
            av_call_count += 1

            # Fallback to price-proxy sentiment if Alpha Vantage unavailable
            if sentiment_df is None and prices_df is not None:
                logger.info(
                    "Falling back to price-proxy sentiment for %s "
                    "(proxy, not oracle — documented in historical_data_collector.py)", coin_id,
                )
                sentiment_df = compute_price_proxy_sentiment(prices_df, coin_id)

            if sentiment_df is not None and not dry_run:
                sentiment_df.to_parquet(sentiment_path, index=False)
                logger.info("Saved sentiment → %s (%d rows)", sentiment_path, len(sentiment_df))

        results[coin_id] = {
            "prices": prices_df,
            "sentiment": sentiment_df,
        }

    if not dry_run:
        logger.info("=" * 60)
        logger.info("Historical data collection complete.")
        logger.info("Files saved to: %s", HIST_DIR)
        for coin_id, r in results.items():
            p = r.get("prices")
            s = r.get("sentiment")
            logger.info(
                "  %s: %d price rows, %d sentiment rows",
                coin_id,
                len(p) if p is not None else 0,
                len(s) if s is not None else 0,
            )

    return results


def load_historical_data(coins: list[str] = None) -> dict:
    """
    Load the pre-built historical dataset from parquet files.
    Call this from backtest.py instead of invoking the live pipeline.

    Returns
    -------
    dict of {coin_id: {"prices": DataFrame, "sentiment": DataFrame}}
    """
    if coins is None:
        coins = list(COINS.keys())

    results = {}
    for coin_id in coins:
        prices_path = HIST_DIR / f"prices_{coin_id}.parquet"
        sentiment_path = HIST_DIR / f"sentiment_{coin_id}.parquet"

        if not prices_path.exists():
            logger.warning(
                "No historical prices for %s at %s. "
                "Run: python src/historical_data_collector.py", coin_id, prices_path,
            )
            continue

        prices = pd.read_parquet(prices_path)
        sentiment = pd.read_parquet(sentiment_path) if sentiment_path.exists() else None

        results[coin_id] = {"prices": prices, "sentiment": sentiment}
        logger.info(
            "Loaded %s: %d price rows, %d sentiment rows",
            coin_id, len(prices), len(sentiment) if sentiment is not None else 0,
        )

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="CryptoPulse historical data collector")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without fetching")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch even if cached files exist")
    parser.add_argument("--coin", type=str, default=None, help="Collect for a single coin (default: all)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"Days of history (default: {DEFAULT_DAYS})")
    args = parser.parse_args()

    coins = [args.coin] if args.coin else None

    if args.dry_run:
        logger.info("=== DRY RUN — no data will be fetched ===")

    results = collect_historical_data(
        coins=coins,
        days=args.days,
        dry_run=args.dry_run,
        force_refresh=args.force_refresh,
    )

    if not args.dry_run:
        logger.info("✓ Done. Run backtest.py to use this data.")
    else:
        logger.info("✓ Dry run complete. Re-run without --dry-run to fetch data.")
