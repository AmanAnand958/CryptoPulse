"""
graph_state.py
==============
Typed state schema for the CryptoPulse LangGraph multi-agent graph.

Uses TypedDict so LangGraph can validate, merge, and route on structured
fields rather than free text — enabling the supervisor to resolve conflicts
on specific fields (e.g., comparing technical RSI signal vs. sentiment signal).

Point-in-time note:
  All data fetched at each graph invocation is tagged with `as_of_timestamp`
  (the moment the information was actually available), NOT the collection time.
  This ensures backtest replay can filter correctly without look-ahead bias.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Annotated, List
from enum import Enum
import operator


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalOutput(TypedDict):
    """Structured output schema enforced on signal_generator_agent via with_structured_output."""
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # 0.0 – 1.0
    rationale: str       # single-sentence explanation (<100 words)
    horizon: str         # e.g. "1-3 days"


class BrightDataCall(TypedDict):
    """Logged for every Bright Data request to track usage against 5,000/month cap."""
    endpoint: str        # page URL or API endpoint
    timestamp: str       # ISO-8601 UTC
    agent: str           # "market_data" | "sentiment"
    cost_weight: float   # relative weight (1.0 = 1 standard request)
    cached: bool         # True if served from cache, not a live call


class CryptoPulseState(TypedDict):
    """
    Full typed state for the CryptoPulse LangGraph graph.

    Fields are populated progressively as nodes execute.
    Unset fields default to None — supervisor handles missing gracefully.
    """

    # --- Input ---
    coin: str                            # e.g. "bitcoin"
    symbol: str                          # e.g. "BTC"
    as_of_timestamp: str                 # ISO-8601 UTC point-in-time

    # --- market_data_agent output ---
    ohlcv: Optional[dict]                # {open, high, low, close, volume}
    order_book: Optional[dict]           # {bids, asks} snapshot
    volume_trend: Optional[str]          # "increasing" | "decreasing" | "flat"
    market_data_error: Optional[str]     # set if fetch failed/fell back to cache
    market_data_cached: bool             # True if served from last-known-value

    # --- sentiment_agent output ---
    sentiment_score: Optional[float]     # [-1, 1]
    news_headlines: Optional[list]       # list of str
    sentiment_error: Optional[str]
    sentiment_cached: bool

    # --- technical_analysis_node output (no LLM) ---
    rsi: Optional[float]                 # 0–100
    macd: Optional[float]               # MACD line value
    macd_signal: Optional[float]        # MACD signal line
    realized_vol: Optional[float]       # 7-day realized volatility

    # --- signal_generator_agent structured output ---
    signal: Optional[SignalOutput]       # structured trade signal

    # --- supervisor_agent output ---
    final_signal: Optional[SignalOutput] # resolved final signal

    # --- Bookkeeping (Annotated reducers for parallel fan-out updates) ---
    errors: Annotated[list, operator.add]             # accumulated error messages
    bright_data_calls: Annotated[list, operator.add]  # list of BrightDataCall dicts for usage logging
    messages: Annotated[list, operator.add]           # LangGraph message list (tool calls + responses)
