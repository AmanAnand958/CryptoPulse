"""
agents/__init__.py
==================
LangGraph multi-agent pipeline for CryptoPulse RL.

Architecture (supervisor pattern):
    ┌─────────────────────────────────────────┐
    │              LangGraph Graph             │
    │                                          │
    │  market_data_agent ──┐                   │
    │  (Bright Data MCP)   │                   │
    │                       ▼                  │
    │  sentiment_agent ────► technical_node    │
    │  (Bright Data MCP)    ▼                  │
    │                  signal_generator        │
    │                       ▼                  │
    │                  supervisor_agent        │
    └─────────────────────────────────────────┘

market_data_agent and sentiment_agent run in PARALLEL (fan-out).
technical_analysis_node is a pure function (no LLM).
signal_generator_agent uses with_structured_output.
supervisor_agent resolves conflicts and emits final_signal.
"""

from .graph import build_graph, run_signal_graph

__all__ = ["build_graph", "run_signal_graph"]
