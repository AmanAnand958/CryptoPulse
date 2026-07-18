"""
llm_signal.py
=============
LLM-based trade signal generator using the Groq API (free tier).

Model: llama3-70b-8192 (Groq free tier)

For each (coin, timestamp), constructs a compact prompt from:
  - Recent price action (last N candle returns, volatility, volume trend)
  - Recent sentiment score

Returns a strictly-parsed JSON signal:
  {direction: "long"|"short"|"hold", confidence: 0.0-1.0, rationale: str}

Caching:
  Signals are cached to data/processed/signal_cache.json keyed by
  (coin, date_str) so repeated backtest runs do NOT re-call the API.
  This is critical for reproducibility and staying within free-tier limits.

Fallback:
  Any parse failure → {"direction": "hold", "confidence": 0.0, "rationale": "parse_failure"}
  Pipeline NEVER crashes on a bad LLM response.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # export GROQ_API_KEY="..." before running
GROQ_MODEL = "llama-3.3-70b-versatile"   # llama3-70b-8192 was decommissioned in 2025
MAX_RETRIES = 3
RETRY_SLEEP = 2.0        # seconds between retries
RATE_LIMIT_SLEEP = 1.0   # seconds between API calls (free tier: 30 req/min)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "data" / "processed" / "signal_cache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

FALLBACK_SIGNAL = {"direction": "hold", "confidence": 0.0, "rationale": "parse_failure"}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(coin: str, date_str: str) -> str:
    return f"{coin}::{date_str}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    coin: str,
    symbol: str,
    recent_closes: list[float],
    recent_volumes: list[float],
    volatility_7d: float,
    ret_1d: float,
    ret_7d: float,
    sentiment_score: float,
) -> str:
    """Build a compact, information-dense prompt for structured signal extraction."""
    recent_rets = [
        f"{((recent_closes[i] / recent_closes[i-1]) - 1) * 100:.2f}%"
        for i in range(1, min(8, len(recent_closes)))
    ]
    vol_trend = "increasing" if recent_volumes[-1] > np.mean(recent_volumes[:-1]) else "decreasing"
    sentiment_label = (
        "strongly bullish" if sentiment_score > 0.5 else
        "bullish" if sentiment_score > 0.15 else
        "neutral" if abs(sentiment_score) <= 0.15 else
        "bearish" if sentiment_score > -0.5 else
        "strongly bearish"
    )

    return f"""You are a quantitative crypto analyst. Analyze the following market data for {coin} ({symbol}) and output a structured trade signal.

MARKET DATA:
- Recent daily returns (last 7 days): {', '.join(recent_rets)}
- 1-day return: {ret_1d*100:.2f}%
- 7-day return: {ret_7d*100:.2f}%
- 7-day realized volatility: {volatility_7d*100:.2f}%
- Volume trend: {vol_trend}
- Sentiment score: {sentiment_score:.3f} ({sentiment_label})

Your task: determine the likely direction over the next 1-3 days.

Respond ONLY with a JSON object and absolutely nothing else — no markdown, no commentary, no code fences:
{{"direction": "long" or "short" or "hold", "confidence": <float 0.0 to 1.0>, "rationale": "<one sentence explanation>"}}

Rules:
- direction must be exactly one of: long, short, hold
- confidence must be a float between 0.0 and 1.0
- rationale must be a single sentence under 100 words
- If uncertain, use "hold" with low confidence
- Do NOT output anything other than the JSON object"""


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def _parse_signal(raw_text: str) -> dict:
    """
    Parse LLM response into signal dict.
    Tries multiple strategies before falling back.
    """
    raw_text = raw_text.strip()
    # Strategy 1: direct JSON parse
    try:
        obj = json.loads(raw_text)
        return _validate_signal(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find JSON substring
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        obj = json.loads(raw_text[start:end])
        return _validate_signal(obj)
    except (ValueError, json.JSONDecodeError):
        pass

    logger.warning("Failed to parse signal from: %s", raw_text[:200])
    return FALLBACK_SIGNAL.copy()


def _validate_signal(obj: dict) -> dict:
    """Validate and normalise a parsed signal dict."""
    direction = str(obj.get("direction", "hold")).lower().strip()
    if direction not in ("long", "short", "hold"):
        direction = "hold"
    confidence = float(obj.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(obj.get("rationale", "no rationale"))[:300]
    return {"direction": direction, "confidence": confidence, "rationale": rationale}


def generate_signal(
    coin: str,
    symbol: str,
    date_str: str,
    recent_closes: list[float],
    recent_volumes: list[float],
    volatility_7d: float,
    ret_1d: float,
    ret_7d: float,
    sentiment_score: float,
    use_cache: bool = True,
    client: Groq = None,
    cache: dict = None,
) -> dict:
    """
    Generate a trade signal for a single (coin, date) pair.

    Parameters are the market context at that time step.
    Returns dict with keys: direction, confidence, rationale.
    """
    cache_key = _cache_key(coin, date_str)

    # Cache hit
    if use_cache and cache is not None and cache_key in cache:
        return cache[cache_key]

    prompt = _build_prompt(
        coin, symbol, recent_closes, recent_volumes,
        volatility_7d, ret_1d, ret_7d, sentiment_score,
    )

    if client is None:
        client = Groq(api_key=GROQ_API_KEY)

    signal = FALLBACK_SIGNAL.copy()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   # low temp for consistent structured output
                max_tokens=200,
            )
            raw_text = response.choices[0].message.content
            signal = _parse_signal(raw_text)
            logger.debug("Signal for %s @ %s: %s", coin, date_str, signal)
            break
        except Exception as exc:
            logger.warning("Groq API attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                logger.error("All retries failed for %s @ %s, using fallback", coin, date_str)

    # Cache the result
    if cache is not None:
        cache[cache_key] = signal

    time.sleep(RATE_LIMIT_SLEEP)  # free-tier rate limit
    return signal


# ---------------------------------------------------------------------------
# Batch signal generation
# ---------------------------------------------------------------------------

def generate_signals_for_dataframe(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    lookback: int = 8,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Generate signals for all (coin, date) pairs in prices_df.

    Parameters
    ----------
    prices_df : DataFrame with columns [date, coin, symbol, close, volume]
    sentiment_df : DataFrame with columns [date, coin, sentiment]
    lookback : int  — number of past candles used for prompt context
    use_cache : bool — use/update local cache file

    Returns
    -------
    DataFrame with columns: [date, coin, symbol, direction, confidence, rationale]
    """
    cache = _load_cache() if use_cache else {}
    client = Groq(api_key=GROQ_API_KEY)

    sent_map = sentiment_df.set_index(["date", "coin"])["sentiment"].to_dict()
    results = []

    for coin, grp in prices_df.groupby("coin"):
        grp = grp.sort_values("date").reset_index(drop=True)
        symbol = grp["symbol"].iloc[0]
        closes = grp["close"].tolist()
        volumes = grp["volume"].tolist()
        dates = grp["date"].tolist()

        for i in range(lookback, len(grp)):
            date_str = str(dates[i].date() if hasattr(dates[i], "date") else dates[i])[:10]
            window_closes = closes[max(0, i - lookback): i + 1]
            window_vols = volumes[max(0, i - lookback): i + 1]

            # Returns
            ret_1d = (closes[i] / closes[i - 1] - 1) if closes[i - 1] else 0
            ret_7d = (closes[i] / closes[max(0, i - 7)] - 1) if closes[max(0, i - 7)] else 0

            # 7-day realized volatility (std of daily log returns)
            rets_7d = [
                np.log(closes[j] / closes[j - 1])
                for j in range(max(1, i - 6), i + 1)
                if closes[j - 1] > 0
            ]
            vol_7d = float(np.std(rets_7d)) if rets_7d else 0.01

            # Sentiment
            date_key = dates[i] if isinstance(dates[i], pd.Timestamp) else pd.Timestamp(dates[i])
            sentiment = sent_map.get((date_key, coin), 0.0)

            signal = generate_signal(
                coin=coin,
                symbol=symbol,
                date_str=date_str,
                recent_closes=window_closes,
                recent_volumes=window_vols,
                volatility_7d=vol_7d,
                ret_1d=ret_1d,
                ret_7d=ret_7d,
                sentiment_score=float(sentiment),
                use_cache=use_cache,
                client=client,
                cache=cache,
            )

            results.append({
                "date": dates[i],
                "coin": coin,
                "symbol": symbol,
                **signal,
            })

    # Persist cache
    if use_cache:
        _save_cache(cache)
        logger.info("Signal cache saved → %s (%d entries)", CACHE_PATH, len(cache))

    signals_df = pd.DataFrame(results)
    out_path = BASE_DIR / "data" / "processed" / "signals.csv"
    signals_df.to_csv(out_path, index=False)
    logger.info("Signals saved → %s", out_path)
    return signals_df


if __name__ == "__main__":
    from data_ingest import run_ingest
    prices, sentiment = run_ingest()
    # Quick test: generate signals for last 10 rows of BTC
    btc = prices[prices["coin"] == "bitcoin"].tail(12).copy()
    sent = sentiment[sentiment["coin"] == "bitcoin"].tail(12).copy()
    signals = generate_signals_for_dataframe(btc, sent, lookback=8)
    print(signals)
