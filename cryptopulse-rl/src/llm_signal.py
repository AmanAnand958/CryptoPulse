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
  "<coin>::<YYYY-MM-DD>" so repeated backtest runs do NOT re-call the API.
  Fallback (parse_failure / API error) signals are NEVER written to cache
  so failed dates are automatically retried on the next run.

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
from typing import Optional
from datetime import date
import pandas as pd
import numpy as np
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "data" / "processed" / "signal_cache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Primary & Secondary API Keys
GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Models
GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"   # Much higher rate limits on Groq free tier
GEMINI_MODEL = "gemini-2.5-flash"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"

MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 0.5    # Gap between API calls — reduces 429 pressure
FALLBACK_SIGNAL = {"direction": "hold", "confidence": 0.0, "rationale": "API fallback signal"}

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

    return f"""You are a neutral quantitative crypto analyst. Analyze the market data for {coin} ({symbol}) and output a calibrated trade signal.

MARKET DATA:
- Recent daily returns (last 7 days): {', '.join(recent_rets)}
- 1-day return: {ret_1d*100:.2f}%
- 7-day return: {ret_7d*100:.2f}%
- 7-day realized volatility: {volatility_7d*100:.2f}%
- Volume trend: {vol_trend}
- Sentiment score: {sentiment_score:.3f} ({sentiment_label})

Your task: determine the most probable direction over the next 1-3 days.

CALIBRATION REQUIREMENT — before deciding, explicitly consider:
1. What is the strongest evidence FOR a bullish (long) move?
2. What is the strongest evidence FOR a bearish (short) move?
3. Are the signals conflicting, weak, or noisy?
If the evidence for both directions is roughly balanced, or if volatility is very high, output "hold".
Only output "long" or "short" when you have clear, asymmetric evidence.
Aim for roughly equal long/short/hold frequency across many signals — do NOT default to short.

Respond ONLY with a JSON object and absolutely nothing else — no markdown, no commentary, no code fences:
{{"direction": "long" or "short" or "hold", "confidence": <float 0.0 to 1.0>, "rationale": "<one sentence explanation>"}}

Rules:
- direction must be exactly one of: long, short, hold
- confidence must be a float between 0.0 and 1.0
- rationale must be a single sentence under 100 words
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


def _call_groq(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Execute request against Groq API."""
    if not api_key:
        return None
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("Groq API (%s) call failed: %s", model, exc)
        return None


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    """Execute request against Gemini 2.5 Flash Lite via google.genai."""
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        logger.warning("Gemini API (%s) call failed: %s", GEMINI_MODEL, exc)
        return None


def _call_nvidia(prompt: str, api_key: str) -> Optional[str]:
    """Execute request against NVIDIA NIM API (high throughput / generous limits)."""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("NVIDIA API (%s) call failed: %s", NVIDIA_MODEL, exc)
        return None


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
    cache: dict = None,
) -> dict:
    """
    Generate a trade signal for a single (coin, date) pair with API key rotation:
    1. NVIDIA NIM API (meta/llama-3.3-70b-instruct - High Speed & High Limit)
    2. Groq Key 1 (llama-3.1-8b-instant)
    3. Groq Key 2 (llama-3.1-8b-instant)
    4. Gemini API Key (gemini-2.5-flash-lite)
    5. Fallback rule
    """
    cache_key = _cache_key(coin, date_str)

    # Cache hit
    if use_cache and cache is not None and cache_key in cache:
        return cache[cache_key]

    prompt = _build_prompt(
        coin, symbol, recent_closes, recent_volumes,
        volatility_7d, ret_1d, ret_7d, sentiment_score,
    )

    raw_text = None

    # Step 1: Try Gemini API (gemini-2.5-flash - Ultra High Throughput)
    if GEMINI_API_KEY:
        raw_text = _call_gemini(prompt, GEMINI_API_KEY)

    # Step 2: Try Groq Key 1
    if not raw_text and GROQ_API_KEY_1:
        logger.info("Rotating to Groq Key 1 (%s)...", GROQ_MODEL_FAST)
        raw_text = _call_groq(prompt, GROQ_API_KEY_1, GROQ_MODEL_FAST)

    # Step 3: Try Groq Key 2
    if not raw_text and GROQ_API_KEY_2:
        logger.info("Rotating to Groq Key 2 (%s)...", GROQ_MODEL_FAST)
        raw_text = _call_groq(prompt, GROQ_API_KEY_2, GROQ_MODEL_FAST)

    # Step 4: Try NVIDIA NIM API
    if not raw_text and NVIDIA_API_KEY:
        logger.info("Rotating to NVIDIA API (%s)...", NVIDIA_MODEL)
        raw_text = _call_nvidia(prompt, NVIDIA_API_KEY)

    # Parse or Fallback
    if raw_text:
        signal = _parse_signal(raw_text)
    else:
        logger.warning("All LLM API rotators failed for %s @ %s, using fallback", coin, date_str)
        signal = FALLBACK_SIGNAL.copy()

    # Cache the result ONLY if it is a real LLM output, NOT an API fallback / parse failure
    is_fallback = signal.get("rationale") in ("parse_failure", "API fallback signal")
    if cache is not None and not is_fallback:
        cache[cache_key] = signal

    time.sleep(RATE_LIMIT_SLEEP)
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
