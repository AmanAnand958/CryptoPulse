"""
signal_generator_agent.py
=========================
LangGraph node: synthesizes technical + sentiment context into a structured
trade signal using with_structured_output.

Structured output schema (enforced via Pydantic + with_structured_output):
  {
    action:     "BUY" | "SELL" | "HOLD"
    confidence: float  (0.0 – 1.0)
    rationale:  str    (single sentence, <100 words)
    horizon:    str    (e.g. "1-3 days")
  }

LLM: Groq llama-3.3-70b-versatile (existing key)
Caching: (coin, date_str, context_hash) → signal to avoid redundant LLM calls
  during backtests and avoid re-calling for the same market state.

Fallback: if LLM fails or parse fails, returns HOLD with confidence=0.0
  — pipeline never crashes.
"""

from __future__ import annotations

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .graph_state import CryptoPulseState, SignalOutput

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIGNAL_CACHE_PATH = BASE_DIR / "data" / "processed" / "agent_signal_cache.json"
SIGNAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_SLEEP = 2.0
RATE_LIMIT_SLEEP = 1.0

GROQ_MODEL = "llama-3.3-70b-versatile"

FALLBACK_SIGNAL: SignalOutput = {
    "action": "HOLD",
    "confidence": 0.0,
    "rationale": "Signal generation failed — defaulting to HOLD.",
    "horizon": "unknown",
}


# ─── Pydantic schema for with_structured_output ───────────────────────────

class TradeSignalSchema(BaseModel):
    """Structured output schema enforced on the LLM by with_structured_output."""
    action: str = Field(
        description="Trade action: must be exactly 'BUY', 'SELL', or 'HOLD'",
        pattern="^(BUY|SELL|HOLD)$",
    )
    confidence: float = Field(
        description="Confidence in the signal from 0.0 (no confidence) to 1.0 (very confident)",
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        description="Single sentence explanation of the signal, under 100 words",
        max_length=600,
    )
    horizon: str = Field(
        description="Expected time horizon for the trade, e.g. '1-3 days' or '1 week'",
    )


# ─── Cache helpers ─────────────────────────────────────────────────────────

def _load_signal_cache() -> dict:
    if SIGNAL_CACHE_PATH.exists():
        with open(SIGNAL_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_signal_cache(cache: dict) -> None:
    with open(SIGNAL_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _context_hash(coin: str, rsi: float, macd: float, sentiment: float, vol: float) -> str:
    """Hash key so identical market states reuse cached signals."""
    raw = f"{coin}:{rsi:.1f}:{macd:.4f}:{sentiment:.2f}:{vol:.4f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ─── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(state: CryptoPulseState) -> str:
    coin = state["coin"]
    symbol = state["symbol"]
    ohlcv = state.get("ohlcv") or {}
    rsi = state.get("rsi") if state.get("rsi") is not None else 50.0
    macd = state.get("macd") if state.get("macd") is not None else 0.0
    macd_signal = state.get("macd_signal") if state.get("macd_signal") is not None else 0.0
    realized_vol = state.get("realized_vol") if state.get("realized_vol") is not None else 0.01
    sentiment_score = state.get("sentiment_score") if state.get("sentiment_score") is not None else 0.0
    news_headlines = state.get("news_headlines", []) or []
    close = ohlcv.get("close", 0.0)
    volume = ohlcv.get("volume", 0.0)
    volume_trend = state.get("volume_trend", "flat") or "flat"

    sentiment_label = (
        "strongly bullish" if sentiment_score > 0.5 else
        "bullish" if sentiment_score > 0.15 else
        "neutral" if abs(sentiment_score) <= 0.15 else
        "bearish" if sentiment_score > -0.5 else
        "strongly bearish"
    )

    rsi_label = (
        "overbought" if rsi > 70 else
        "oversold" if rsi < 30 else
        "neutral"
    )

    macd_label = "bullish crossover" if macd > macd_signal else "bearish crossover" if macd < macd_signal else "flat"

    headlines_str = ""
    if news_headlines:
        headlines_str = "\nRecent news headlines:\n" + "\n".join(f"  - {h}" for h in news_headlines[:5])

    return f"""You are a quantitative crypto analyst. Analyze the following multi-source market data for {coin} ({symbol}) and output a structured trade signal.

TECHNICAL INDICATORS:
- Current price: ${close:,.2f}
- RSI (14): {rsi:.1f} ({rsi_label})
- MACD: {macd:.4f} vs Signal: {macd_signal:.4f} ({macd_label})
- 7-day realized volatility: {realized_vol * 100:.2f}%
- Volume: {volume:,.0f} (trend: {volume_trend})

SENTIMENT:
- Sentiment score: {sentiment_score:.3f} ({sentiment_label}){headlines_str}

TASK: Synthesize the technical and sentiment signals into a single trade recommendation for the next 1-3 days.

Rules:
- action must be exactly BUY, SELL, or HOLD
- confidence must be a float between 0.0 and 1.0
- rationale must be a single sentence under 100 words explaining the key driver
- horizon should be a short string like "1-3 days"
- If technical and sentiment conflict, weight technical signals more heavily and lower confidence
- If uncertain, use HOLD with low confidence"""


# ─── Node Function ─────────────────────────────────────────────────────────

def signal_generator_node(state: CryptoPulseState) -> dict:
    """
    LangGraph node function for signal_generator_agent.

    Uses with_structured_output to enforce the TradeSignalSchema.
    Caches results by (coin, context_hash) to avoid duplicate LLM calls.
    Falls back to HOLD on any failure.

    State reads:  coin, symbol, ohlcv, rsi, macd, macd_signal, realized_vol,
                  sentiment_score, news_headlines, volume_trend
    State writes: signal, errors
    """
    coin = state["coin"]
    errors = list(state.get("errors", []))

    rsi = state.get("rsi") if state.get("rsi") is not None else 50.0
    macd = state.get("macd") if state.get("macd") is not None else 0.0
    sentiment = state.get("sentiment_score") if state.get("sentiment_score") is not None else 0.0
    vol = state.get("realized_vol") if state.get("realized_vol") is not None else 0.01

    # Build context hash for caching
    ck = _context_hash(
        coin=coin,
        rsi=rsi,
        macd=macd,
        sentiment=sentiment,
        vol=vol,
    )
    cache = _load_signal_cache()

    # Cache hit
    if ck in cache:
        logger.info("signal_generator_node: cache hit for %s (hash=%s)", coin, ck)
        return {"signal": cache[ck], "errors": errors}

    # Build prompt
    prompt = _build_prompt(state)

    # LangChain with_structured_output
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("GROQ_API_KEY not set — using fallback HOLD signal")
        errors.append("GROQ_API_KEY missing")
        return {"signal": FALLBACK_SIGNAL, "errors": errors}

    signal: SignalOutput = FALLBACK_SIGNAL.copy()

    for attempt in range(MAX_RETRIES):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=groq_key,
                temperature=0.1,
                max_tokens=300,
            )
            structured_llm = llm.with_structured_output(TradeSignalSchema)
            result = structured_llm.invoke(prompt)

            signal = {
                "action": result.action,
                "confidence": float(result.confidence),
                "rationale": result.rationale[:600],
                "horizon": result.horizon,
            }
            logger.info(
                "signal_generator_node: %s → action=%s confidence=%.2f",
                coin, signal["action"], signal["confidence"],
            )
            break

        except Exception as exc:
            logger.warning("signal_generator attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                errors.append(f"signal_generator failed after {MAX_RETRIES} retries: {exc}")
                logger.error("All retries failed for %s — using HOLD fallback", coin)

    # Cache result
    cache[ck] = signal
    _save_signal_cache(cache)

    time.sleep(RATE_LIMIT_SLEEP)  # respect free-tier rate limit

    return {"signal": signal, "errors": errors}
