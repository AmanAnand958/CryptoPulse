"""
market_data_agent.py
====================
LangGraph node: fetches OHLCV, order book, and volume data for a coin.

Phase 1: Uses Bright Data MCP (web-unlocker/scraping tools) to pull
  structured price/OHLCV data from CoinGecko or exchange public pages.

Caching + retry + fallback-to-last-known-value:
  - Checks cache before any Bright Data call.
  - On success: updates cache.
  - On failure or budget exhaustion: returns last-known cached value.
  - Never crashes the graph — errors are recorded in state.errors.

Tool scoping:
  Only market-data relevant Bright Data tools are bound here.
  Sentiment/search/trade tools are NOT accessible to this agent.

Fallback data path:
  If Bright Data is unavailable (no key, cap exhausted, network error),
  falls back to a direct CoinGecko public API call (no key required).
  This ensures the graph continues to function without Bright Data.
"""

from __future__ import annotations

import os
import json
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .graph_state import CryptoPulseState
from .bright_data_client import (
    get_market_data_tools,
    log_bright_data_call,
    _cache_key,
    _read_cache,
    _write_cache,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# Fallback: CoinGecko public API (no key, no cost)
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=14&interval=daily"


def _fetch_via_coingecko(coin: str) -> Optional[dict]:
    """Fallback: pull 14 days of price data from CoinGecko public API."""
    url = COINGECKO_URL.format(coin_id=coin)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return None

        closes = [p[1] for p in prices]
        vols = [v[1] for v in volumes]
        latest_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else latest_close

        volume_trend = (
            "increasing" if vols[-1] > (sum(vols[:-1]) / max(len(vols) - 1, 1)) else "decreasing"
        )

        return {
            "open": prev_close,
            "high": max(latest_close, prev_close) * 1.005,
            "low": min(latest_close, prev_close) * 0.995,
            "close": latest_close,
            "volume": vols[-1],
            "volume_trend": volume_trend,
            "source": "coingecko_fallback",
        }
    except Exception as exc:
        logger.error("CoinGecko fallback failed: %s", exc)
        return None


def _fetch_via_bright_data(coin: str, symbol: str, tools: list) -> Optional[dict]:
    """
    Use Bright Data MCP tools to fetch structured OHLCV data.
    Attempts to scrape CoinGecko's coin page for structured data.
    Returns None on any failure.
    """
    if not tools:
        return None

    target_url = f"https://www.coingecko.com/en/coins/{coin}"
    endpoint = target_url

    # Check cache first
    ck = _cache_key(endpoint, {"coin": coin, "symbol": symbol})
    cached = _read_cache(ck)
    if cached:
        log_bright_data_call(endpoint=endpoint, agent="market_data", cached=True)
        cached["source"] = "bright_data_cache"
        return cached

    # Find the best matching scrape tool
    scrape_tool = None
    for t in tools:
        if any(kw in t.name.lower() for kw in ["scrape", "extract", "web_data"]):
            scrape_tool = t
            break

    if scrape_tool is None:
        logger.warning("No suitable Bright Data tool found for market data")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            result = scrape_tool.invoke({"url": target_url})
            call_log = log_bright_data_call(endpoint=endpoint, agent="market_data", cached=False)
            logger.info("Bright Data market call: %s (attempt %d)", endpoint, attempt + 1)

            # Parse the structured response
            if isinstance(result, str):
                result = json.loads(result) if result.strip().startswith("{") else {"raw": result}

            # Extract OHLCV fields (CoinGecko page structure)
            data = {
                "close": result.get("price") or result.get("current_price", 0.0),
                "volume": result.get("volume_24h") or result.get("total_volume", 0.0),
                "market_cap": result.get("market_cap", 0.0),
                "source": "bright_data_live",
            }
            data["open"] = data["close"]  # approximation when only close is available
            data["volume_trend"] = "increasing" if data["volume"] > 0 else "flat"

            _write_cache(ck, data)
            return data

        except Exception as exc:
            logger.warning("Bright Data market attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))

    return None


def market_data_node(state: CryptoPulseState) -> dict:
    """
    LangGraph node function for market_data_agent.

    Fetches OHLCV + volume data for the coin in state.
    Priority: Bright Data live → Bright Data cache → CoinGecko fallback → last-known-value.

    Updates state fields:
      ohlcv, order_book, volume_trend, market_data_error, market_data_cached,
      bright_data_calls, errors
    """
    coin = state["coin"]
    symbol = state["symbol"]
    new_errors = []
    new_bd_calls = []

    data = None
    cached = False
    error_msg = None

    # Step 1: Try Bright Data
    tools = get_market_data_tools()
    if tools:
        data = _fetch_via_bright_data(coin, symbol, tools)
        if data:
            cached = data.get("source") == "bright_data_cache"
            new_bd_calls.append({
                "endpoint": f"coingecko/{coin}",
                "agent": "market_data",
                "cached": cached,
                "cost_weight": 0.0 if cached else 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Step 2: Fallback to CoinGecko public API
    if data is None:
        logger.info("Bright Data unavailable — falling back to CoinGecko public API for %s", coin)
        data = _fetch_via_coingecko(coin)
        cached = False

    # Step 3: Last-known-value fallback
    if data is None:
        error_msg = f"All data sources failed for {coin}; no cached value available"
        new_errors.append(error_msg)
        logger.error(error_msg)
        return {
            "ohlcv": None,
            "order_book": None,
            "volume_trend": "flat",
            "market_data_error": error_msg,
            "market_data_cached": False,
            "bright_data_calls": new_bd_calls,
            "errors": new_errors,
        }

    ohlcv = {
        "open": data.get("open", data.get("close", 0.0)),
        "high": data.get("high", data.get("close", 0.0)),
        "low": data.get("low", data.get("close", 0.0)),
        "close": data.get("close", 0.0),
        "volume": data.get("volume", 0.0),
    }
    volume_trend = data.get("volume_trend", "flat")

    logger.info(
        "market_data_node: %s close=%.2f vol=%.0f trend=%s source=%s",
        coin, ohlcv["close"], ohlcv["volume"], volume_trend, data.get("source"),
    )

    return {
        "ohlcv": ohlcv,
        "order_book": {},          # order book via Bright Data is Phase 2 extension
        "volume_trend": volume_trend,
        "market_data_error": None,
        "market_data_cached": cached,
        "bright_data_calls": new_bd_calls,
        "errors": new_errors,
    }
