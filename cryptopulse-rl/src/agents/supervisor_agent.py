"""
supervisor_agent.py
===================
LangGraph node: aggregates outputs from technical and sentiment pipelines,
resolves disagreements between technical and sentiment signals, and
produces the final_signal passed to the execution layer.

Resolution logic (structured field-by-field, not free text):
  1. If technical RSI and sentiment agree → boost confidence.
  2. If they disagree → reduce confidence, keep signal if strong,
     else override to HOLD.
  3. Final action maps: BUY ↔ "long", SELL ↔ "short", HOLD ↔ "hold"
     for compatibility with the existing execution layer vocabulary.

The supervisor operates entirely on structured fields from the state
(not raw text) — this is why graph_state.py uses TypedDict.
"""

from __future__ import annotations

import logging
from .graph_state import CryptoPulseState, SignalOutput

logger = logging.getLogger(__name__)

HOLD_SIGNAL: SignalOutput = {
    "action": "HOLD",
    "confidence": 0.0,
    "rationale": "Supervisor defaulted to HOLD due to missing inputs.",
    "horizon": "unknown",
}


def _technical_direction(rsi: float, macd: float, macd_signal: float) -> str:
    """
    Derive a simple technical direction from RSI and MACD.
    Returns "BUY", "SELL", or "HOLD".
    """
    rsi_signal = (
        "BUY" if rsi < 35 else
        "SELL" if rsi > 65 else
        "HOLD"
    )
    macd_signal_dir = (
        "BUY" if macd > macd_signal else
        "SELL" if macd < macd_signal else
        "HOLD"
    )

    # Agreement = use that direction; disagreement = HOLD
    if rsi_signal == macd_signal_dir:
        return rsi_signal
    if rsi_signal == "HOLD":
        return macd_signal_dir
    if macd_signal_dir == "HOLD":
        return rsi_signal
    return "HOLD"  # conflict


def _sentiment_direction(sentiment_score: float) -> str:
    """Convert sentiment score to a directional label."""
    if sentiment_score > 0.2:
        return "BUY"
    elif sentiment_score < -0.2:
        return "SELL"
    return "HOLD"


def supervisor_node(state: CryptoPulseState) -> dict:
    """
    LangGraph node function for supervisor_agent.

    Resolves technical vs. sentiment signal conflict on structured fields
    and produces the final_signal for the execution layer.

    State reads:  signal, rsi, macd, macd_signal, sentiment_score,
                  market_data_error, sentiment_error
    State writes: final_signal
    """
    signal: SignalOutput = state.get("signal") or HOLD_SIGNAL.copy()

    rsi = state.get("rsi", 50.0) or 50.0
    macd = state.get("macd", 0.0) or 0.0
    macd_signal_val = state.get("macd_signal", 0.0) or 0.0
    sentiment_score = state.get("sentiment_score", 0.0) or 0.0

    llm_action = signal.get("action", "HOLD")
    llm_confidence = float(signal.get("confidence", 0.0))

    # Derive technical and sentiment directions from structured fields
    tech_dir = _technical_direction(rsi, macd, macd_signal_val)
    sent_dir = _sentiment_direction(sentiment_score)

    logger.info(
        "supervisor_node: LLM=%s(%.2f) technical=%s sentiment=%s",
        llm_action, llm_confidence, tech_dir, sent_dir,
    )

    # Resolution logic
    agreements = sum([
        1 for d in [tech_dir, sent_dir] if d == llm_action
    ])

    if agreements == 2:
        # All three agree → boost confidence slightly
        final_confidence = min(1.0, llm_confidence * 1.1)
        final_action = llm_action
        rationale_prefix = "Technical and sentiment confirm LLM signal. "
    elif agreements == 1:
        # One agrees, one disagrees → keep LLM signal but reduce confidence
        final_confidence = llm_confidence * 0.8
        final_action = llm_action
        rationale_prefix = "Mixed signals: one indicator confirms, one disagrees. "
    else:
        # Full disagreement → if LLM confidence is high, keep it; else HOLD
        if llm_confidence >= 0.7:
            final_confidence = llm_confidence * 0.6
            final_action = llm_action
            rationale_prefix = "Technical/sentiment disagree with LLM (high-confidence override). "
        else:
            final_confidence = 0.0
            final_action = "HOLD"
            rationale_prefix = "Technical and sentiment both disagree with LLM signal — defaulting to HOLD. "

    final_signal: SignalOutput = {
        "action": final_action,
        "confidence": round(final_confidence, 4),
        "rationale": rationale_prefix + signal.get("rationale", ""),
        "horizon": signal.get("horizon", "1-3 days"),
    }

    logger.info(
        "supervisor_node: final → action=%s confidence=%.2f",
        final_signal["action"], final_signal["confidence"],
    )

    return {"final_signal": final_signal}
