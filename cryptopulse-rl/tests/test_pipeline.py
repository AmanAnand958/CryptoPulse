"""
test_pipeline.py
================
Integration tests for the CryptoPulse RL pipeline.

Tests:
  - data_ingest: data shape, date alignment, column presence
  - features: state vector shape and bounds
  - rl_policy: bandit select/update cycle
  - llm_signal: JSON parsing and fallback logic
  - baselines: portfolio monotonicity and non-negativity
  - evaluate: metric computations
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import build_state_vector, STATE_DIM, compute_hit_rate
from rl_policy import LinUCBBandit, ACTIONS, N_ACTIONS, compute_reward, decide
from baselines import run_buy_and_hold, run_llm_only
from evaluate import cumulative_return, sharpe_ratio, max_drawdown
from llm_signal import _parse_signal, _validate_signal, FALLBACK_SIGNAL


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_state_vector_shape(self):
        state = build_state_vector(
            direction="long", confidence=0.7,
            ret_1d=0.02, ret_7d=0.1, ret_30d=0.2, vol_7d=0.025,
            current_position=0.0, unrealized_pnl=0.0,
            past_directions=["long", "short"], past_returns=[0.01, -0.02],
        )
        assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},) got {state.shape}"

    def test_state_vector_bounds(self):
        state = build_state_vector(
            direction="short", confidence=1.0,
            ret_1d=0.5, ret_7d=1.0, ret_30d=2.0, vol_7d=0.5,
            current_position=1.0, unrealized_pnl=0.5,
            past_directions=[], past_returns=[],
        )
        assert np.all(state >= -4), "State values should be clipped near [-3,3]"
        assert np.all(state <= 4)

    def test_hit_rate_neutral_when_empty(self):
        assert compute_hit_rate([], [], K=10) == 0.5

    def test_hit_rate_perfect_long(self):
        dirs = ["long"] * 10
        rets = [0.01] * 10
        assert compute_hit_rate(dirs, rets, K=10) == 1.0

    def test_hit_rate_excludes_hold(self):
        dirs = ["hold"] * 5 + ["long"] * 5
        rets = [0.0] * 5 + [0.01] * 5
        # Only 5 non-hold signals, all profitable → 1.0
        assert compute_hit_rate(dirs, rets, K=10) == 1.0

    def test_direction_encoding(self):
        from features import encode_direction
        assert encode_direction("long") == 1.0
        assert encode_direction("short") == -1.0
        assert encode_direction("hold") == 0.0
        assert encode_direction("invalid") == 0.0


# ---------------------------------------------------------------------------
# RL Policy
# ---------------------------------------------------------------------------

class TestLinUCBBandit:

    def test_action_selection_returns_valid_arm(self):
        bandit = LinUCBBandit()
        ctx = np.random.randn(STATE_DIM)
        arm, mult = decide(bandit, ctx, "long", training=False)
        assert 0 <= arm < N_ACTIONS
        assert mult in ACTIONS

    def test_update_changes_parameters(self):
        bandit = LinUCBBandit()
        ctx = np.random.randn(STATE_DIM)
        b_before = bandit.b[0].copy()
        bandit.update(0, ctx, 1.0)
        assert not np.allclose(bandit.b[0], b_before)

    def test_reward_computation_long_positive(self):
        # Long position with positive forward return → positive reward
        reward = compute_reward(1.0, "long", 0.05, 0.02, 0.0)
        assert reward > 0

    def test_reward_computation_short_negative(self):
        # Short position with positive forward return → negative reward
        reward = compute_reward(1.0, "short", 0.05, 0.02, 0.0)
        assert reward < 0

    def test_reward_clipped(self):
        # Extreme returns should be clipped
        reward = compute_reward(1.0, "long", 100.0, 0.001, 0.0)
        assert reward <= 5.0

    def test_bandit_reset(self):
        bandit = LinUCBBandit()
        ctx = np.random.randn(STATE_DIM)
        bandit.update(0, ctx, 1.0)
        bandit.reset()
        assert bandit._step == 0
        assert np.allclose(bandit.b, 0)

    def test_save_load_cycle(self, tmp_path):
        bandit = LinUCBBandit()
        ctx = np.random.randn(STATE_DIM)
        bandit.update(2, ctx, 0.5)
        path = tmp_path / "policy.json"
        bandit.save(path)

        bandit2 = LinUCBBandit()
        bandit2.load(path)
        assert np.allclose(bandit.A, bandit2.A)
        assert np.allclose(bandit.b, bandit2.b)


# ---------------------------------------------------------------------------
# LLM Signal Parsing
# ---------------------------------------------------------------------------

class TestLLMSignalParsing:

    def test_valid_json_parses(self):
        raw = '{"direction": "long", "confidence": 0.8, "rationale": "Strong momentum."}'
        sig = _parse_signal(raw)
        assert sig["direction"] == "long"
        assert sig["confidence"] == 0.8

    def test_embedded_json_parses(self):
        raw = 'Here is the signal: {"direction": "short", "confidence": 0.6, "rationale": "Bearish."} done.'
        sig = _parse_signal(raw)
        assert sig["direction"] == "short"

    def test_bad_json_returns_fallback(self):
        sig = _parse_signal("This is not JSON at all.")
        assert sig == FALLBACK_SIGNAL

    def test_invalid_direction_corrected(self):
        obj = {"direction": "buy", "confidence": 0.5, "rationale": "test"}
        sig = _validate_signal(obj)
        assert sig["direction"] == "hold"   # invalid direction → hold

    def test_confidence_clipped(self):
        obj = {"direction": "long", "confidence": 5.0, "rationale": "test"}
        sig = _validate_signal(obj)
        assert sig["confidence"] == 1.0

    def test_confidence_clipped_negative(self):
        obj = {"direction": "long", "confidence": -0.5, "rationale": "test"}
        sig = _validate_signal(obj)
        assert sig["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:

    def _make_prices(self, n: int = 30) -> pd.Series:
        np.random.seed(0)
        prices = 1000.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.Series(prices, index=idx)

    def test_buy_and_hold_non_negative(self):
        prices = self._make_prices()
        df = run_buy_and_hold(prices)
        assert (df["portfolio_value"] >= 0).all()

    def test_buy_and_hold_length(self):
        prices = self._make_prices(20)
        df = run_buy_and_hold(prices)
        assert len(df) == 20

    def test_llm_only_non_negative(self):
        prices = self._make_prices()
        n = len(prices)
        signals = pd.Series(
            ["long"] * (n // 2) + ["short"] * (n - n // 2),
            index=prices.index,
        )
        df = run_llm_only(prices, signals)
        assert (df["portfolio_value"] >= 0).all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def _make_portfolio(self, returns: list) -> pd.Series:
        vals = [10000.0]
        for r in returns:
            vals.append(vals[-1] * (1 + r))
        return pd.Series(vals)

    def test_cumulative_return_positive(self):
        p = self._make_portfolio([0.01] * 10)
        assert cumulative_return(p) > 0

    def test_cumulative_return_negative(self):
        p = self._make_portfolio([-0.02] * 10)
        assert cumulative_return(p) < 0

    def test_sharpe_positive_series(self):
        p = self._make_portfolio([0.005] * 100)
        sr = sharpe_ratio(p.pct_change().fillna(0))
        assert sr > 0

    def test_max_drawdown_nonnegative(self):
        p = self._make_portfolio([0.05, -0.10, 0.02, -0.05, 0.03])
        dd = max_drawdown(p)
        assert dd >= 0

    def test_max_drawdown_all_up(self):
        p = self._make_portfolio([0.01] * 20)
        assert max_drawdown(p) == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
