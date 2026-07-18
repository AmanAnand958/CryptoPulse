"""
rl_policy.py
============
Contextual Bandit execution policy for CryptoPulse RL.

Primary implementation: LinUCB (Linear Upper Confidence Bound)
  — implemented from scratch using numpy (no library black-box)

WHY A BANDIT, NOT FULL MDP/PPO:
  1. The decision being made is: "how much should I trust this LLM signal
     right now?" — this is stateless per signal, not a sequential MDP.
     A bandit models exactly this: pick an action (position multiplier)
     conditional on a context (market state), observe a reward.
  2. Crypto price series only weakly satisfy the Markov and stationarity
     assumptions that underpin full RL (value function decomposition,
     Bellman equations). A bandit makes no such assumptions.
  3. The bandit is interpretable: the weight vector θ_a tells you which
     context features most predicted reward for each action — a human can
     audit this. A neural PPO policy cannot be audited this way.
  4. With limited data (< 1 year of daily signals), a bandit's simpler
     hypothesis class is less prone to overfitting than a neural policy.
  PPO is included as a STRETCH EXTENSION clearly marked [secondary].

Action space (5 discrete actions):
  ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
  Each is a multiplier on the fixed max allocation (e.g., 20% of portfolio).
  - Negative: go against the LLM signal (short)
  - 0: ignore the signal entirely
  - Positive: follow the LLM signal at partial or full allocation

Reward:
  r = (forward_return / realized_vol_forward) - tx_cost_penalty
  where tx_cost_penalty = |Δposition| * TX_COST_RATE

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
ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # position multipliers
N_ACTIONS = len(ACTIONS)
STATE_DIM = 9                              # must match features.py
ALPHA = 1.0                               # LinUCB exploration parameter
TX_COST_RATE = 0.001                      # 0.1% per unit position change
MAX_ALLOCATION = 0.20                     # max 20% of portfolio per trade
FORWARD_WINDOW = 14                       # days to evaluate reward

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
    the five position-size actions are qualitatively different from each
    other (ignoring signal vs. following it vs. fading it), so sharing
    parameters would misrepresent the problem structure.
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
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    action_multiplier: float,
    llm_direction: str,
    forward_return: float,
    forward_vol: float,
    prev_position: float,
) -> float:
    """
    Compute the reward signal for a bandit update.

    The actual position taken = action_multiplier × sign(llm_direction) × MAX_ALLOCATION
    (action_multiplier can be negative = fade the signal)

    Reward = Sharpe-like ratio minus transaction cost:
      r = (position × forward_return / max(forward_vol, 0.001)) - tx_cost

    Clipped to [-5, 5] to prevent reward outliers from dominating.
    """
    direction_sign = {"long": 1.0, "short": -1.0, "hold": 0.0}.get(llm_direction, 0.0)
    position = action_multiplier * direction_sign * MAX_ALLOCATION
    pnl = position * forward_return
    risk_adj_pnl = pnl / max(forward_vol, 0.001)
    tx_cost = abs(position - prev_position * MAX_ALLOCATION) * TX_COST_RATE
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
    Select an action using the bandit policy.

    Returns
    -------
    (arm_index, position_multiplier)
      arm_index         — index into ACTIONS list
      position_multiplier — the actual multiplier value (e.g., 0.5)
    """
    arm = bandit.select_action(context)
    multiplier = ACTIONS[arm]

    # During training, occasionally force-explore by random action
    # (epsilon = 0.05 ensures the policy sees all arms during training)
    if training and np.random.random() < 0.05:
        arm = np.random.randint(0, N_ACTIONS)
        multiplier = ACTIONS[arm]

    return arm, multiplier


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    bandit = LinUCBBandit()
    ctx = np.random.randn(STATE_DIM)
    arm, mult = decide(bandit, ctx, "long", training=True)
    print(f"Selected arm={arm}, multiplier={mult}")
    reward = compute_reward(mult, "long", 0.03, 0.025, 0.0)
    print(f"Reward: {reward:.4f}")
    bandit.update(arm, ctx, reward)
    print(f"Theta (arm {arm}): {bandit.get_theta(arm)}")
    bandit.save()
    print("✓ LinUCB bandit OK")
