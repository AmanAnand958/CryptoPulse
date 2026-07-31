"""
graph.py
========
Assembles the CryptoPulse LangGraph multi-agent supervisor graph.

Graph topology (fan-out → serial → supervisor):

    START
      │
      ├──► market_data_node   ─┐   (parallel fan-out)
      │                         ▼
      └──► sentiment_node    ─► technical_analysis_node
                                 │
                                 ▼
                          signal_generator_node
                                 │
                                 ▼
                          supervisor_node
                                 │
                                END

Parallel execution:
  market_data_node and sentiment_node run in parallel using LangGraph's
  fan-out support (Send API). technical_analysis_node runs after market_data.
  signal_generator_node runs after both technical and sentiment complete.

Caching:
  Each agent node checks its own cache before any live call.
  The graph itself does not add an extra caching layer — caching lives
  inside each agent so the same underlying fetch isn't repeated per agent
  per cycle.

Usage:
    from agents.graph import build_graph, run_signal_graph

    result = run_signal_graph(coin="bitcoin", symbol="BTC")
    final_signal = result["final_signal"]
    # {action: "BUY", confidence: 0.72, rationale: "...", horizon: "1-3 days"}
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import StateGraph, END, START

from .graph_state import CryptoPulseState
from .market_data_agent import market_data_node
from .sentiment_agent import sentiment_node
from .technical_analysis_node import technical_analysis_node
from .signal_generator_agent import signal_generator_node
from .supervisor_agent import supervisor_node

logger = logging.getLogger(__name__)


# ─── Routing ───────────────────────────────────────────────────────────────

def after_parallel_data(state: CryptoPulseState) -> Literal["technical_analysis", "signal_generator"]:
    """
    After both market_data and sentiment complete:
    → run technical_analysis first (it depends on ohlcv from market_data)
    → then signal_generator (depends on both technical + sentiment)

    LangGraph handles join automatically when both fan-out branches complete.
    """
    return "technical_analysis"


# ─── Graph builder ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the CryptoPulse LangGraph supervisor graph.

    Returns a compiled graph ready for .invoke() calls.
    """
    builder = StateGraph(CryptoPulseState)

    # Register all nodes
    builder.add_node("market_data", market_data_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("technical_analysis", technical_analysis_node)
    builder.add_node("signal_generator", signal_generator_node)
    builder.add_node("supervisor", supervisor_node)

    # Fan-out: START → market_data AND sentiment in parallel
    builder.add_edge(START, "market_data")
    builder.add_edge(START, "sentiment")

    # technical_analysis runs after market_data completes
    builder.add_edge("market_data", "technical_analysis")

    # signal_generator runs after BOTH technical_analysis AND sentiment complete
    # LangGraph joins automatically when all predecessor nodes for a node finish
    builder.add_edge("technical_analysis", "signal_generator")
    builder.add_edge("sentiment", "signal_generator")

    # supervisor runs after signal_generator
    builder.add_edge("signal_generator", "supervisor")

    # END after supervisor
    builder.add_edge("supervisor", END)

    return builder.compile()


# Module-level compiled graph (lazy, compiled on first use)
_GRAPH = None


def _get_graph() -> StateGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ─── Public interface ──────────────────────────────────────────────────────

def run_signal_graph(
    coin: str,
    symbol: str,
    as_of_timestamp: str = None,
) -> CryptoPulseState:
    """
    Run the full multi-agent signal graph for a single (coin, timestamp).

    Parameters
    ----------
    coin : str
        CoinGecko coin ID, e.g. "bitcoin"
    symbol : str
        Ticker symbol, e.g. "BTC"
    as_of_timestamp : str, optional
        ISO-8601 UTC point-in-time. Defaults to now.

    Returns
    -------
    CryptoPulseState dict with all populated fields.
    Access final_signal for the execution layer.
    """
    if as_of_timestamp is None:
        as_of_timestamp = datetime.now(timezone.utc).isoformat()

    initial_state: CryptoPulseState = {
        "coin": coin,
        "symbol": symbol,
        "as_of_timestamp": as_of_timestamp,
        # Data agent fields — will be populated by nodes
        "ohlcv": None,
        "order_book": None,
        "volume_trend": None,
        "market_data_error": None,
        "market_data_cached": False,
        "sentiment_score": None,
        "news_headlines": None,
        "sentiment_error": None,
        "sentiment_cached": False,
        # Technical fields — populated by technical_analysis_node
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "realized_vol": None,
        # Signal fields — populated by signal_generator and supervisor
        "signal": None,
        "final_signal": None,
        # Bookkeeping
        "errors": [],
        "bright_data_calls": [],
        "messages": [],
    }

    graph = _get_graph()
    logger.info("Running signal graph for %s (%s) @ %s", coin, symbol, as_of_timestamp)

    result = graph.invoke(initial_state)

    final = result.get("final_signal") or {
        "action": "HOLD",
        "confidence": 0.0,
        "rationale": "Graph produced no final signal.",
        "horizon": "unknown",
    }

    logger.info(
        "Graph complete: %s → %s (conf=%.2f) | BD calls=%d | errors=%d",
        coin,
        final.get("action"),
        final.get("confidence", 0.0),
        len(result.get("bright_data_calls", [])),
        len(result.get("errors", [])),
    )

    return result


# ─── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load env
    from dotenv import load_dotenv
    import sys
    sys.path.insert(0, str(__file__).replace("/agents/graph.py", ""))
    load_dotenv()

    print("Building CryptoPulse LangGraph...")
    graph = build_graph()
    print("Graph nodes:", list(graph.nodes))
    print()

    print("Running signal graph for bitcoin...")
    result = run_signal_graph("bitcoin", "BTC")

    print("\n=== RESULT ===")
    print(f"Final Signal: {json.dumps(result.get('final_signal'), indent=2)}")
    print(f"RSI: {result.get('rsi')}")
    print(f"MACD: {result.get('macd')}")
    print(f"Sentiment: {result.get('sentiment_score')}")
    print(f"Errors: {result.get('errors')}")
    print(f"Bright Data calls: {len(result.get('bright_data_calls', []))}")
    print()
    print("✓ LangGraph smoke test complete")
