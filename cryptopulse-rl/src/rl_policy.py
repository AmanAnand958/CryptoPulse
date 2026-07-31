"""
rl_policy.py
============
Contextual Bandit execution policy for CryptoPulse RL.

Primary implementation: LinUCB (Linear Upper Confidence Bound)
  — implemented from scratch using numpy (no library black-box)

WHY A BANDIT, NOT FULL MDP/PPO:
  1. The decision being made is: "how much of my max allocation should I
     deploy on this LLM signal right now?" — this is stateless per signal,
     not a sequential MDP. A bandit models exactly this: pick an action
     (allocation level) conditional on a context (market state), observe a reward.
  2. Crypto price series only weakly satisfy the Markov and stationarity
     assumptions that underpin full RL (value function decomposition,
     Bellman equations). A bandit makes no such assumptions.
  3. The bandit is interpretable: the weight vector θ_a tells you which
     context features most predicted reward for each action — a human can
     audit this. A neural PPO policy cannot be audited this way.
  4. With limited data (< 1 year of daily signals), a bandit's simpler
     hypothesis class is less prone to overfitting than a neural policy.
  PPO is included as a STRETCH EXTENSION clearly marked [secondary].

Action space — MEMORYLESS discrete allocation levels (section 4 of spec):
  ACTIONS = [0.0, 0.25, 0.50, 1.0]
  Each is a fraction of MAX_ALLOCATION (e.g. 0.25 → 25% of 20% = 5% of portfolio).
  The bandit chooses fresh each period — holding duration is NOT encoded here.
  Direction of the trade (long/short) is determined by the LLM signal, not the bandit.

  RATIONALE FOR CHANGE (from old ±multiplier scheme):
    The old ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0] implicitly encoded
    whether to follow or fade the signal AND how large a position to take.
    Folding both choices into one action introduced state dependence
    (what position am I currently in?) that the bandit assumption doesn't cover.
    The new scheme separates concerns:
      - LLM/supervisor decides DIRECTION (BUY/SELL/HOLD)
      - Bandit decides ALLOCATION SIZE (0%, 25%, 50%, 100% of max)

Reward — RISK-ADJUSTED, cost-aware (section 4 of spec):
  r = Sharpe-like term - tx_cost_penalty
    = (position_pnl / max(realized_vol, 0.001)) - |Δalloc| * TX_COST_RATE
  Clipped to [-5, 5] to prevent reward outliers from dominating.
  NOT raw PnL — the old reward was already risk-adjusted but now documented
  clearly as a Sharpe numerator term.

LinUCB Update Rule (documented):
  For each action arm a, maintain:
    A_a  ∈ R^{d×d}  — regularised feature covariance (init = I * alpha)
    b_a  ∈ R^d      — reward-weighted feature sum (init = 0)
  At decision time:
    θ_a = A_a^{-1} b_a
    UCB_a = θ_a^T x + alpha * sqrt(x^T A_a^{-1} x)
  Select action: argmax_a UCB_a
  After observing reward r:
    A_a ← A_a + x x^T
    b_a ← b_a + r * x
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Memoryless discrete allocation levels (fraction of MAX_ALLOCATION).
# 0.0 = sit out, 0.25 = quarter position, 0.50 = half position, 1.0 = full.
# Direction (long/short) is determined by the LLM/supervisor signal, NOT here.
ACTIONS = [0.0, 0.25, 0.50, 1.0]
N_ACTIONS = len(ACTIONS)

STATE_DIM = 9                              # must match features.py
ALPHA = 1.0                               # LinUCB exploration parameter
TX_COST_RATE = 0.001                      # 0.1% per unit position change
MAX_ALLOCATION = 0.20                     # max 20% of portfolio per trade
FORWARD_WINDOW = 14                       # days to evaluate reward
ROLLING_VOL_WINDOW = 7                    # window for rolling vol in reward

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_CHECKPOINT = BASE_DIR / "data" / "processed" / "policy_checkpoint.json"


# ---------------------------------------------------------------------------
# LinUCB Bandit (from scratch)
# ---------------------------------------------------------------------------

class LinUCBBandit:
    """
    Contextual LinUCB bandit with disjoint linear models per arm.

    Each arm maintains its own A matrix and b vector.
    The exploration-exploitation trade-off is controlled by alpha:
      - High alpha → more exploration (wider confidence bounds)
      - Low alpha  → more exploitation (trust current estimates)

    This is a disjoint model (not hybrid) — appropriate here because
    the four allocation levels are qualitatively different from each
    other (sit out vs. full position), so sharing parameters would
    misrepresent the problem structure.

    Arms are MEMORYLESS: each period's decision is independent of the
    previous period's action. The bandit does NOT track holding duration.
    """

    def __init__(self, n_arms: int = N_ACTIONS, dim: int = STATE_DIM, alpha: float = ALPHA):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        # Per-arm parameters — shape (n_arms, dim, dim) and (n_arms, dim)
        self.A = np.array([np.eye(dim) for _ in range(n_arms)])   # regularised covariance
        self.b = np.zeros((n_arms, dim))                            # reward-weighted features
        self._step = 0

    def select_action(self, context: np.ndarray) -> int:
        """
        Select the arm with the highest UCB score.

        UCB_a = θ_a^T x + alpha * sqrt(x^T A_a^{-1} x)

        Returns arm index (not the ACTIONS value — use ACTIONS[arm_idx]).
        """
        x = context.astype(np.float64)
        ucb_scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            exploration = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[a] = theta @ x + exploration
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update the selected arm's parameters with the observed reward.

        Update rule:
          A_a ← A_a + x x^T   (add outer product of context)
          b_a ← b_a + r * x   (add reward-weighted context)
        """
        x = context.astype(np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self._step += 1

    def get_theta(self, arm: int) -> np.ndarray:
        """Return the current weight vector for an arm (for interpretability)."""
        A_inv = np.linalg.inv(self.A[arm])
        return A_inv @ self.b[arm]

    def save(self, path: Path = POLICY_CHECKPOINT) -> None:
        """Persist policy weights to JSON (numpy arrays → lists)."""
        state = {
            "n_arms": self.n_arms,
            "dim": self.dim,
            "alpha": self.alpha,
            "step": self._step,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }
        with open(path, "w") as f:
            json.dump(state, f)
        logger.debug("Policy saved → %s", path)

    def load(self, path: Path = POLICY_CHECKPOINT) -> None:
        """Load policy weights from JSON."""
        with open(path) as f:
            state = json.load(f)
        self.n_arms = state["n_arms"]
        self.dim = state["dim"]
        self.alpha = state["alpha"]
        self._step = state["step"]
        self.A = np.array(state["A"])
        self.b = np.array(state["b"])
        logger.info("Policy loaded from %s (step=%d)", path, self._step)

    def reset(self) -> None:
        """Reset to prior (used at start of each walk-forward training window)."""
        self.A = np.array([np.eye(self.dim) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.dim))
        self._step = 0


# ---------------------------------------------------------------------------
# Reward computation — RISK-ADJUSTED, cost-aware (section 4 of spec)
# ---------------------------------------------------------------------------

def compute_reward(
    allocation_fraction: float,
    llm_direction: str,
    forward_return: float,
    forward_vol: float,
    prev_allocation: float,
) -> float:
    """
    Compute the risk-adjusted reward signal for a bandit update.

    The bandit arm selects an ALLOCATION FRACTION (0.0, 0.25, 0.50, 1.0).
    Direction (long/short) comes from the LLM signal.

    Actual position = allocation_fraction × direction_sign × MAX_ALLOCATION

    Reward = Sharpe-like term - transaction cost:
      r = (position × forward_return / max(realized_vol, 0.001)) - tx_cost

    This is a rolling Sharpe numerator (not annualised) — adequate for
    the bandit's per-period update. Long-run Sharpe is computed in evaluate.py.

    Clipped to [-5, 5] to prevent outliers from dominating the A/b updates.

    Parameters
    ----------
    allocation_fraction : float
        Bandit arm value from ACTIONS, e.g. 0.25 = 25% of MAX_ALLOCATION.
    llm_direction : str
        "BUY"/"long", "SELL"/"short", or "HOLD"/"hold" from supervisor.
    forward_return : float
        Observed 1-day (or forward window) return.
    forward_vol : float
        Realized volatility over the same forward window.
    prev_allocation : float
        Previous period's allocation fraction (for tx cost calculation).
    """
    # Normalise direction vocabulary (LLM uses BUY/SELL/HOLD, old code uses long/short/hold)
    direction_map = {
        "BUY": 1.0, "long": 1.0,
        "SELL": -1.0, "short": -1.0,
        "HOLD": 0.0, "hold": 0.0,
    }
    direction_sign = direction_map.get(llm_direction, 0.0)

    # Actual position as fraction of portfolio
    position = allocation_fraction * direction_sign * MAX_ALLOCATION

    # Risk-adjusted PnL: Sharpe numerator per step
    pnl = position * forward_return
    risk_adj_pnl = pnl / max(forward_vol, 0.001)

    # Transaction cost from changing position size
    prev_position = prev_allocation * direction_sign * MAX_ALLOCATION
    tx_cost = abs(position - prev_position) * TX_COST_RATE

    reward = float(np.clip(risk_adj_pnl - tx_cost, -5, 5))
    return reward


# ---------------------------------------------------------------------------
# Decision interface (used by backtest.py)
# ---------------------------------------------------------------------------

def decide(
    bandit: LinUCBBandit,
    context: np.ndarray,
    llm_direction: str,
    training: bool = False,
) -> tuple[int, float]:
    """
    Select an allocation level using the bandit policy.

    Returns
    -------
    (arm_index, allocation_fraction)
      arm_index          — index into ACTIONS list
      allocation_fraction — the fraction value (0.0, 0.25, 0.50, or 1.0)

    Note: if llm_direction is HOLD, allocation is forced to 0.0 regardless
    of the bandit's preference — the bandit sizes positions, not directions.
    """
    # Force zero allocation when LLM says hold
    direction_map = {"HOLD": True, "hold": True}
    if direction_map.get(llm_direction, False):
        return 0, 0.0  # arm 0 = 0% allocation

    arm = bandit.select_action(context)
    allocation = ACTIONS[arm]

    # During training, occasionally force-explore by random action
    # (epsilon = 0.05 ensures the policy sees all arms during training)
    if training and np.random.random() < 0.05:
        arm = np.random.randint(0, N_ACTIONS)
        allocation = ACTIONS[arm]

    return arm, allocation


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    bandit = LinUCBBandit()
    ctx = np.random.randn(STATE_DIM)

    arm, alloc = decide(bandit, ctx, "BUY", training=True)
    print(f"Selected arm={arm}, allocation_fraction={alloc} ({alloc*100:.0f}% of MAX_ALLOC)")

    reward = compute_reward(alloc, "BUY", 0.03, 0.025, 0.0)
    print(f"Reward: {reward:.4f}")

    bandit.update(arm, ctx, reward)
    print(f"Theta (arm {arm}): {bandit.get_theta(arm)}")
    bandit.save()

    print("\nActions (discrete allocation levels):", ACTIONS)
    print("✓ LinUCB bandit OK — memoryless arms, risk-adjusted reward")
