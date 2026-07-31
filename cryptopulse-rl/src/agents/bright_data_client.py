"""
bright_data_client.py
=====================
MCP client factory for the Bright Data MCP server.

Responsibilities:
  1. Create an MCP client per invocation using langchain-mcp-adapters.
  2. Discover the server's tools and convert them to LangChain Tool objects.
  3. Return only the subset of tools relevant to the requesting agent
     (market-data tools vs. sentiment/search tools — never all tools to all agents).
  4. Log every live Bright Data call against the 5,000/month free-tier cap.

Phase 1 constraint:
  Bright Data's free MCP tier is capped at 5,000 requests/month.
  Cache aggressively. Never poll more often than data changes.
  Log every call so usage is visible, not discovered after the fact.

Phase 2 note:
  If Phase 1 gaps are identified (brittle scraping, approaching cap), add
  specialized free MCP servers (CCXT, CoinGecko community, DeFiLlama, FRED,
  Alpha Vantage). Those additions live alongside this client, not inside it.
"""

from __future__ import annotations

import os
import json
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
USAGE_LOG_PATH = BASE_DIR / "data" / "processed" / "bright_data_usage.jsonl"
CACHE_DIR = BASE_DIR / "data" / "processed" / "bright_data_cache"

USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Tool scoping per agent ────────────────────────────────────────────────
# Only bind relevant tools to each agent. This prevents, e.g., the sentiment
# agent from accessing trade-related or market-data-specific tools.

MARKET_DATA_TOOL_KEYWORDS = [
    "web_data", "scrape", "extract", "price", "ohlcv", "market",
    "coingecko", "exchange", "ticker",
]

SENTIMENT_TOOL_KEYWORDS = [
    "search", "news", "social", "reddit", "twitter", "headline",
    "scrape", "web_data",
]


def _tool_matches_keywords(tool_name: str, keywords: list[str]) -> bool:
    """Check if a tool name/description contains any of the given keywords."""
    name_lower = tool_name.lower()
    return any(kw in name_lower for kw in keywords)


def _cache_key(endpoint: str, params: dict) -> str:
    raw = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(cache_key: str) -> Optional[dict]:
    """Return cached response if it exists, else None."""
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _write_cache(cache_key: str, data: dict) -> None:
    path = CACHE_DIR / f"{cache_key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def log_bright_data_call(
    endpoint: str,
    agent: str,
    cached: bool,
    cost_weight: float = 1.0,
) -> dict:
    """
    Log a Bright Data call to the usage log file.

    Every call (cached or live) is logged so usage against the 5,000/month
    cap can be tracked. Cached calls have cost_weight=0.0.
    """
    record = {
        "endpoint": endpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "cost_weight": 0.0 if cached else cost_weight,
        "cached": cached,
    }
    with open(USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.debug(
        "Bright Data call logged: agent=%s endpoint=%s cached=%s weight=%.1f",
        agent, endpoint, cached, record["cost_weight"],
    )
    return record


def get_bright_data_usage_summary() -> dict:
    """Return total calls and estimated cost weight this month."""
    if not USAGE_LOG_PATH.exists():
        return {"total_calls": 0, "total_weight": 0.0, "live_calls": 0}

    total_weight = 0.0
    live_calls = 0
    total_calls = 0
    with open(USAGE_LOG_PATH) as f:
        for line in f:
            rec = json.loads(line.strip())
            total_calls += 1
            total_weight += rec.get("cost_weight", 0.0)
            if not rec.get("cached", False):
                live_calls += 1

    return {
        "total_calls": total_calls,
        "live_calls": live_calls,
        "total_weight": total_weight,
        "cap_remaining": max(0, 5000 - live_calls),
        "pct_used": round(live_calls / 5000 * 100, 1),
    }


# ─── MCP Client (langchain-mcp-adapters) ──────────────────────────────────

def get_market_data_tools() -> list:
    """
    Create an MCP client for Bright Data, discover tools, and return
    ONLY market-data relevant tools for market_data_agent.

    Returns a list of LangChain Tool objects.
    Returns [] if BRIGHT_DATA_API_KEY is not set (graceful degradation).
    """
    api_key = os.environ.get("BRIGHT_DATA_API_KEY", "")
    if not api_key:
        logger.warning(
            "BRIGHT_DATA_API_KEY not set — market_data_agent will use fallback data. "
            "Set this key in cryptopulse-rl/.env to enable live Bright Data fetching."
        )
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({
            "bright_data": {
                "command": "npx",
                "args": ["-y", "@brightdata/mcp"],
                "env": {"BRIGHT_DATA_API_KEY": api_key},
                "transport": "stdio",
            }
        })
        all_tools = client.get_tools()
        market_tools = [
            t for t in all_tools
            if _tool_matches_keywords(t.name, MARKET_DATA_TOOL_KEYWORDS)
        ]
        logger.info(
            "Bright Data: discovered %d tools, bound %d to market_data_agent",
            len(all_tools), len(market_tools),
        )
        return market_tools
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed. Run: pip install langchain-mcp-adapters")
        return []
    except Exception as exc:
        logger.error("Failed to connect to Bright Data MCP: %s", exc)
        return []


def get_sentiment_tools() -> list:
    """
    Create an MCP client for Bright Data and return ONLY sentiment/search
    tools for sentiment_agent. Never returns trade or market-price tools.

    Returns [] if BRIGHT_DATA_API_KEY is not set.
    """
    api_key = os.environ.get("BRIGHT_DATA_API_KEY", "")
    if not api_key:
        logger.warning(
            "BRIGHT_DATA_API_KEY not set — sentiment_agent will use fallback data."
        )
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({
            "bright_data": {
                "command": "npx",
                "args": ["-y", "@brightdata/mcp"],
                "env": {"BRIGHT_DATA_API_KEY": api_key},
                "transport": "stdio",
            }
        })
        all_tools = client.get_tools()
        sentiment_tools = [
            t for t in all_tools
            if _tool_matches_keywords(t.name, SENTIMENT_TOOL_KEYWORDS)
        ]
        logger.info(
            "Bright Data: discovered %d tools, bound %d to sentiment_agent",
            len(all_tools), len(sentiment_tools),
        )
        return sentiment_tools
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed.")
        return []
    except Exception as exc:
        logger.error("Failed to connect to Bright Data MCP for sentiment: %s", exc)
        return []
