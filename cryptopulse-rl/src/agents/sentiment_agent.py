"""
sentiment_agent.py
==================
LangGraph node: fetches news/social sentiment for a coin.

Phase 1: Uses Bright Data MCP search + scrape tools to pull news and
  social content directly (not a dedicated sentiment API).

Caching + retry + fallback:
  - Checks cache before any Bright Data call.
  - On success: updates cache, computes sentiment score from headlines.
  - On failure: returns last-known cached sentiment score (0.0 = neutral).
  - Never crashes the graph — errors are recorded in state.errors.

Tool scoping:
  Only search/news/social Bright Data tools are bound here.
  Market-price, trade, or on-chain tools are NOT accessible to this agent.

Fallback path:
  If Bright Data is unavailable, falls back to a price-momentum proxy
  sentiment score (same logic as data_ingest.py). Clearly documented
  as a proxy — not an oracle.
"""

from __future__ import annotations

import os
import json
import time
import logging
import re
import requests
from datetime import datetime, timezone
from typing import Optional

from .graph_state import CryptoPulseState
from .bright_data_client import (
    get_sentiment_tools,
    log_bright_data_call,
    _cache_key,
    _read_cache,
    _write_cache,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# Simple lexicon for fallback headline scoring
BULLISH_WORDS = {
    "surge", "rally", "bullish", "breakout", "record", "high", "gains",
    "adoption", "institutional", "etf", "approval", "moon", "soaring",
}
BEARISH_WORDS = {
    "crash", "plunge", "bearish", "sell-off", "selloff", "ban", "hack",
    "fraud", "scam", "collapse", "regulatory", "sec", "lawsuit", "dump",
}


def _score_headlines(headlines: list[str]) -> float:
    """
    Simple lexicon-based sentiment scoring of headlines.
    Returns a score in [-1, 1].
    """
    if not headlines:
        return 0.0

    total_score = 0.0
    for h in headlines:
        words = set(re.sub(r"[^a-z\s]", "", h.lower()).split())
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total_score += (bull - bear) / max(1, bull + bear + 1)

    return max(-1.0, min(1.0, total_score / len(headlines)))


def _fetch_sentiment_via_bright_data(coin: str, symbol: str, tools: list) -> Optional[dict]:
    """
    Use Bright Data search/scrape tools to pull news/social content.
    Returns {sentiment_score, news_headlines, source} or None on failure.
    """
    if not tools:
        return None

    # Search query for this coin
    query = f"{coin} {symbol} crypto news sentiment"
    endpoint = f"bright_data_search:{query[:50]}"

    ck = _cache_key(endpoint, {"coin": coin, "symbol": symbol})
    cached = _read_cache(ck)
    if cached:
        log_bright_data_call(endpoint=endpoint, agent="sentiment", cached=True)
        cached["source"] = "bright_data_cache"
        return cached

    # Find search tool
    search_tool = None
    for t in tools:
        if any(kw in t.name.lower() for kw in ["search", "news", "scrape", "web_data"]):
            search_tool = t
            break

    if search_tool is None:
        logger.warning("No suitable Bright Data tool found for sentiment")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            result = search_tool.invoke({"query": query})
            log_bright_data_call(endpoint=endpoint, agent="sentiment", cached=False)
            logger.info("Bright Data sentiment call: %s (attempt %d)", endpoint, attempt + 1)

            if isinstance(result, str):
                result = json.loads(result) if result.strip().startswith("{") else {"results": result}

            # Extract headlines from result
            headlines = []
            if isinstance(result, dict):
                for key in ["results", "items", "articles", "headlines"]:
                    if key in result and isinstance(result[key], list):
                        for item in result[key][:10]:
                            if isinstance(item, str):
                                headlines.append(item)
                            elif isinstance(item, dict):
                                headlines.append(item.get("title", item.get("text", "")))
                        break

            score = _score_headlines(headlines)
            data = {
                "sentiment_score": score,
                "news_headlines": headlines[:10],
                "source": "bright_data_live",
            }
            _write_cache(ck, data)
            return data

        except Exception as exc:
            logger.warning("Bright Data sentiment attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))

    return None


def _fallback_sentiment_from_price(ohlcv: Optional[dict]) -> float:
    """
    Proxy sentiment score from price data when all news sources fail.
    Positive close vs open → mildly bullish, negative → mildly bearish.
    Clearly documented as a proxy (same logic as data_ingest.py).
    """
    if not ohlcv:
        return 0.0
    open_p = ohlcv.get("open", 0)
    close_p = ohlcv.get("close", 0)
    if open_p <= 0:
        return 0.0
    ret = (close_p - open_p) / open_p
    return float(max(-1.0, min(1.0, ret / 0.05)))


def sentiment_node(state: CryptoPulseState) -> dict:
    """
    LangGraph node function for sentiment_agent.

    Fetches news/social sentiment for the coin in state.
    Priority: Bright Data live → Bright Data cache → price proxy fallback.

    Updates state fields:
      sentiment_score, news_headlines, sentiment_error, sentiment_cached,
      bright_data_calls, errors
    """
    coin = state["coin"]
    symbol = state["symbol"]
    ohlcv = state.get("ohlcv")
    new_errors = []
    new_bd_calls = []

    data = None
    cached = False
    error_msg = None

    # Step 1: Try Bright Data
    tools = get_sentiment_tools()
    if tools:
        data = _fetch_sentiment_via_bright_data(coin, symbol, tools)
        if data:
            cached = data.get("source") == "bright_data_cache"
            new_bd_calls.append({
                "endpoint": f"bright_data_search:{coin}_news",
                "agent": "sentiment",
                "cached": cached,
                "cost_weight": 0.0 if cached else 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Step 2: Fallback to price-proxy sentiment
    if data is None:
        logger.info(
            "Bright Data unavailable — falling back to price-proxy sentiment for %s "
            "(proxy, not oracle — see data_ingest.py for documentation)", coin,
        )
        score = _fallback_sentiment_from_price(ohlcv)
        data = {
            "sentiment_score": score,
            "news_headlines": [],
            "source": "price_proxy_fallback",
        }

    logger.info(
        "sentiment_node: %s score=%.3f headlines=%d source=%s",
        coin,
        data.get("sentiment_score", 0.0),
        len(data.get("news_headlines", [])),
        data.get("source"),
    )

    return {
        "sentiment_score": data.get("sentiment_score", 0.0),
        "news_headlines": data.get("news_headlines", []),
        "sentiment_error": error_msg,
        "sentiment_cached": cached,
        "bright_data_calls": new_bd_calls,
        "errors": new_errors,
    }
