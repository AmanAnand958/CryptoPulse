"""
technical_analysis_node.py
===========================
LangGraph node: computes technical indicators from market_data_agent output.

This is a PURE FUNCTION node — no LLM call, no external API.
It is fast, deterministic, and cannot fail due to rate limits or API issues.

Indicators computed:
  - RSI (14-period, computed from OHLCV close prices)
  - MACD (12/26/9 EMA, standard parameters)
  - Realized volatility (7-day, log returns)

Inputs come from state.ohlcv (set by market_data_agent).
If ohlcv is missing (market_data_agent failed), returns safe defaults.

Note: RSI and MACD require a window of historical closes. The node
computes approximate values when only the current snapshot is available,
and fuller values when historical data (passed in extended state) is present.
For live inference, this is a single-period approximation. For backtesting,
the backtest.py harness provides the full price series.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from .graph_state import CryptoPulseState

logger = logging.getLogger(__name__)


# ─── Technical Indicator Helpers ──────────────────────────────────────────

def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """
    Compute RSI from a list of close prices.
    Returns 50.0 (neutral) if not enough data.
    """
    if len(closes) < period + 1:
        return 50.0

    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    # Use simple moving average for initial RS (Wilder uses EMA; SA for simplicity)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(values: list[float], period: int) -> list[float]:
    """Compute EMA of a list of values."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    ema_values = [values[0]]
    for v in values[1:]:
        ema_values.append(v * k + ema_values[-1] * (1 - k))
    return ema_values


def _compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float]:
    """
    Compute MACD line and signal line.
    Returns (0.0, 0.0) if not enough data.
    """
    if len(closes) < slow:
        return 0.0, 0.0

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

    if len(macd_line) < signal:
        return macd_line[-1], 0.0

    signal_line = _ema(macd_line, signal)
    return float(macd_line[-1]), float(signal_line[-1])


def _compute_realized_vol(closes: list[float], window: int = 7) -> float:
    """
    Compute realized volatility (std of daily log returns) over window.
    Returns 0.01 (1%) as default if not enough data.
    """
    if len(closes) < 2:
        return 0.01
    window_closes = closes[-window - 1:]
    log_rets = [
        np.log(window_closes[i] / window_closes[i - 1])
        for i in range(1, len(window_closes))
        if window_closes[i - 1] > 0
    ]
    return float(np.std(log_rets)) if len(log_rets) > 1 else 0.01


# ─── Node Function ─────────────────────────────────────────────────────────

def technical_analysis_node(state: CryptoPulseState) -> dict:
    """
    LangGraph node function for technical_analysis_node.

    No LLM call. Computes RSI, MACD, and realized volatility from ohlcv.
    If ohlcv is None (market_data_agent failed), returns safe neutral defaults.

    State reads:  ohlcv
    State writes: rsi, macd, macd_signal, realized_vol
    """
    ohlcv = state.get("ohlcv")

    if ohlcv is None:
        logger.warning(
            "technical_analysis_node: ohlcv is None (market_data_agent failed). "
            "Using neutral defaults."
        )
        return {
            "rsi": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "realized_vol": 0.01,
        }

    # For a single-period live snapshot we can only compute approximate indicators.
    # The close price from ohlcv is used as a single data point.
    # For a full time series (backtest), the caller should pass extended price history.
    close = ohlcv.get("close", 0.0)
    historical_closes = ohlcv.get("historical_closes", [close])

    if not isinstance(historical_closes, list) or len(historical_closes) == 0:
        historical_closes = [close]

    rsi = _compute_rsi(historical_closes)
    macd_val, macd_sig = _compute_macd(historical_closes)
    realized_vol = _compute_realized_vol(historical_closes)

    logger.info(
        "technical_analysis_node: RSI=%.1f MACD=%.4f signal=%.4f vol=%.4f",
        rsi, macd_val, macd_sig, realized_vol,
    )

    return {
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "realized_vol": realized_vol,
    }
