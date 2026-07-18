"""
main.py — FastAPI server for CryptoPulse RL dashboard
=======================================================
Serves precomputed backtest results and live signal cache
to the React dashboard panels.

Endpoints:
  GET /api/signals         — latest LLM signals per coin
  GET /api/portfolio       — simulated paper portfolio history (bandit)
  GET /api/backtest        — walk-forward backtest results summary
  GET /api/metrics         — Sharpe / return / drawdown table
  GET /health              — service health check
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # cryptopulse-rl/
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

SIGNAL_CACHE_PATH = DATA_DIR / "signal_cache.json"
SIGNALS_CSV_PATH = DATA_DIR / "signals.csv"
BACKTEST_PATH = DATA_DIR / "backtest_results.json"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CryptoPulse RL API",
    description="LLM Signal + LinUCB Bandit Execution Layer — Research Backtesting System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict to dashboard origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    coin: str
    symbol: str
    date: str
    direction: str
    confidence: float
    rationale: str


class HealthResponse(BaseModel):
    status: str
    signals_available: bool
    backtest_available: bool
    metrics_available: bool


# ---------------------------------------------------------------------------
# Data loaders (cached in memory)
# ---------------------------------------------------------------------------
_signal_cache: dict[str, Any] | None = None
_backtest_cache: dict[str, Any] | None = None
_metrics_cache: dict[str, Any] | None = None


def _load_signals() -> dict:
    global _signal_cache
    if _signal_cache is None:
        if SIGNAL_CACHE_PATH.exists():
            with open(SIGNAL_CACHE_PATH) as f:
                _signal_cache = json.load(f)
        else:
            _signal_cache = {}
    return _signal_cache


def _load_backtest() -> dict:
    global _backtest_cache
    if _backtest_cache is None:
        if BACKTEST_PATH.exists():
            with open(BACKTEST_PATH) as f:
                _backtest_cache = json.load(f)
        else:
            _backtest_cache = {}
    return _backtest_cache


def _load_metrics() -> dict:
    global _metrics_cache
    if _metrics_cache is None:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                _metrics_cache = json.load(f)
        else:
            _metrics_cache = {}
    return _metrics_cache


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        signals_available=SIGNAL_CACHE_PATH.exists(),
        backtest_available=BACKTEST_PATH.exists(),
        metrics_available=METRICS_PATH.exists(),
    )


@app.get("/api/signals", response_model=list[Signal])
def get_signals():
    """
    Return the most recent LLM signal for each tracked coin.
    Signals are read from the cache file produced by llm_signal.py.
    """
    cache = _load_signals()
    if not cache:
        # Return mock signals so the dashboard works before a full pipeline run
        return _mock_signals()

    # cache keys are "coin::date_str"
    # group by coin, pick most recent
    by_coin: dict[str, dict] = {}
    for key, signal in cache.items():
        parts = key.split("::")
        if len(parts) != 2:
            continue
        coin, date_str = parts
        if coin not in by_coin or date_str > by_coin[coin]["date"]:
            by_coin[coin] = {
                "coin": coin,
                "symbol": coin[:3].upper(),
                "date": date_str,
                **signal,
            }

    return [Signal(**v) for v in by_coin.values()]


@app.get("/api/portfolio")
def get_portfolio(coin: str = "bitcoin"):
    """
    Return simulated paper portfolio history for the given coin.
    Sourced from bandit_results in the backtest output.
    ⚠️ Paper Trading — Not Real Funds
    """
    bt = _load_backtest()
    if not bt or coin not in bt:
        return {"warning": "Paper Trading — Not Real Funds", "data": _mock_portfolio()}

    bandit_records = bt[coin].get("bandit_results", [])
    if not bandit_records:
        return {"warning": "Paper Trading — Not Real Funds", "data": _mock_portfolio()}

    return {
        "warning": "Paper Trading — Not Real Funds",
        "coin": coin,
        "data": [
            {
                "date": r["date"],
                "portfolio_value": r.get("portfolio_value", 10000),
                "direction": r.get("direction", "hold"),
                "position_multiplier": r.get("position_multiplier", 0),
            }
            for r in bandit_records
        ],
    }


@app.get("/api/backtest")
def get_backtest_results():
    """
    Return walk-forward backtest summary (equity curves + windows).
    """
    bt = _load_backtest()
    if not bt:
        return {"note": "No backtest results yet. Run: python src/backtest.py", "data": {}}

    # Return lightweight version (just portfolio values, not full records)
    summary = {}
    for coin, results in bt.items():
        bandit = [{"date": r["date"], "value": r.get("portfolio_value", 10000)}
                  for r in results.get("bandit_results", [])]
        llm = [{"date": r["date"], "value": r.get("portfolio_value", 10000)}
               for r in results.get("llm_results", [])]
        bah = [{"date": r["date"], "value": r.get("portfolio_value", 10000)}
               for r in results.get("bah_results", [])]
        summary[coin] = {
            "bandit": bandit,
            "llm_only": llm,
            "buy_and_hold": bah,
            "windows": results.get("windows", []),
        }
    return summary


@app.get("/api/metrics")
def get_metrics():
    """Return policy comparison table (Sharpe, return, drawdown)."""
    metrics = _load_metrics()
    if not metrics:
        return {"note": "No metrics yet. Run: python src/evaluate.py"}
    return metrics


# ---------------------------------------------------------------------------
# Mock data for dashboard preview (before pipeline runs)
# ---------------------------------------------------------------------------

def _mock_signals() -> list[Signal]:
    import random
    coins = [
        ("bitcoin", "BTC"), ("ethereum", "ETH"), ("solana", "SOL"),
        ("binancecoin", "BNB"), ("cardano", "ADA"),
    ]
    directions = ["long", "short", "hold"]
    rationales = [
        "Strong upward momentum with increasing volume and positive sentiment.",
        "Price breaking above key resistance with bullish sentiment crossover.",
        "Mixed signals — high volatility with unclear directional bias.",
        "Bearish divergence on 7-day momentum; sentiment turning negative.",
        "Consolidation phase, LLM confidence too low to take a position.",
    ]
    result = []
    for coin, symbol in coins:
        d = random.choice(directions)
        result.append(Signal(
            coin=coin, symbol=symbol,
            date="2025-01-15",
            direction=d,
            confidence=round(random.uniform(0.4, 0.9), 2),
            rationale=random.choice(rationales),
        ))
    return result


def _mock_portfolio() -> list[dict]:
    """Generate a synthetic portfolio curve for demo."""
    import random
    portfolio = 10000.0
    data = []
    for i in range(90):
        daily_ret = random.gauss(0.001, 0.025)
        portfolio *= (1 + daily_ret)
        data.append({
            "date": f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            "portfolio_value": round(portfolio, 2),
            "direction": random.choice(["long", "short", "hold"]),
            "position_multiplier": random.choice([-1, -0.5, 0, 0.5, 1]),
        })
    return data


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
