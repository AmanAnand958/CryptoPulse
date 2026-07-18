# CryptoPulse RL — Complete Project Walkthrough

> **Disclaimer**: This is a backtested research/portfolio project. All results are from historical simulations. Past backtest performance does not indicate future results. No real funds are involved. Not financial advice.

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [System Architecture](#2-system-architecture)
3. [Why RL? Why a Bandit?](#3-why-rl-why-a-bandit)
4. [Source Files — Deep Dive](#4-source-files--deep-dive)
5. [Real Backtest Results](#5-real-backtest-results)
6. [Failure Case Analysis](#6-failure-case-analysis)
7. [Dashboard Panels](#7-dashboard-panels)
8. [How to Run](#8-how-to-run)
9. [Tech Stack](#9-tech-stack)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)
11. [Final Deliverable Checklist](#11-final-deliverable-checklist)

---

## 1. What Was Built

**CryptoPulse RL** extends the existing CryptoPulse React dashboard into a two-layer research and backtesting system:

| Layer | Technology | Role |
|-------|-----------|------|
| **Signal Layer** | Groq API (`llama-3.3-70b-versatile`) | Given market data + sentiment, output `{direction, confidence, rationale}` |
| **Execution Layer** | LinUCB Contextual Bandit (numpy, from scratch) | Learn *how much* to trust and act on each signal — position sizing, not price prediction |
| **Validation** | Walk-forward backtest (60d train -> 14d eval) | The only evaluation methodology used — no single train/test split anywhere |
| **Dashboard** | React + MUI + Recharts | Three new panels: SignalPanel, PaperPortfolio, BacktestReport |
| **API** | FastAPI | Bridge between Python backend and React frontend |

### Scope
- **BTC, ETH, SOL, BNB, ADA** — daily OHLCV from CoinGecko (365 days)
- **21 walk-forward folds** per coin
- **445 LLM signal calls** cached to disk (Groq free tier, llama-3.3-70b-versatile)
- **3 policies compared**: LinUCB Bandit, LLM-Only (fixed size), Buy-and-Hold
- **Paper trading only** — labeled as such in both code and UI

---

## 2. System Architecture

```
+-----------------------------------------------------+
|                   DATA LAYER                        |
|                                                     |
|  CoinGecko API (free)                               |
|    -> OHLCV daily, 365 days, 5 coins                |
|    -> Stored: data/processed/prices.csv             |
|                                                     |
|  Sentiment (momentum proxy)                         |
|    -> Derived from price momentum + volume z-score  |
|    -> Stored: data/processed/sentiment.csv          |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|              LLM SIGNAL LAYER                       |
|                                                     |
|  llm_signal.py                                      |
|    -> Constructs prompt per (coin, date)            |
|    -> Groq API: llama-3.3-70b-versatile             |
|    -> Strict JSON output:                           |
|       {direction: long|short|hold,                  |
|        confidence: 0-1,                             |
|        rationale: string}                           |
|    -> Cached per (coin, date) in signal_cache.json  |
|    -> Fallback: {hold, 0, "parse_failure"} on error |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|              STATE CONSTRUCTION                     |
|                                                     |
|  features.py  ->  9-dimensional state vector:       |
|   [0] LLM direction encoded    {-1, 0, +1}         |
|   [1] LLM confidence           [0, 1]              |
|   [2] 1-day return             normalised          |
|   [3] 7-day return             normalised          |
|   [4] 30-day return            normalised          |
|   [5] 7-day realized vol       normalised          |
|   [6] current position         [-1, 1]             |
|   [7] unrealized PnL           normalised          |
|   [8] LLM hit-rate (last K)    [0, 1] <- KEY      |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|           RL EXECUTION LAYER (LinUCB)               |
|                                                     |
|  rl_policy.py                                       |
|    Action space: {-1.0, -0.5, 0.0, 0.5, 1.0}      |
|    (multiplier on fixed 20% max allocation)         |
|                                                     |
|    Update rule (per arm a):                         |
|      A_a <- A_a + x*x^T                             |
|      b_a <- b_a + r*x                               |
|      theta_a = A_a^-1 * b_a                         |
|      UCB = theta_a^T*x + alpha*sqrt(x^T*A_a^-1*x)  |
|                                                     |
|    Reward: Sharpe-like(forward_return) - tx_cost    |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|           WALK-FORWARD BACKTEST                     |
|                                                     |
|  backtest.py                                        |
|    60-day train -> 14-day eval -> roll -> repeat    |
|    21 folds per coin, fresh bandit each fold        |
|    Tracks: signal, action, portfolio value, regret  |
|                                                     |
|  baselines.py                                       |
|    Buy-and-hold (same eval periods)                 |
|    LLM-only fixed-size (same eval periods)          |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|              EVALUATION                             |
|                                                     |
|  evaluate.py                                        |
|    Per policy: Sharpe, cum return, max drawdown     |
|    Cumulative regret vs. oracle                     |
|    Failure period: worst bandit vs. LLM-only fold   |
|    Plots -> reports/figures/                        |
+--------------------+--------------------------------+
                     |
                     v
+-----------------------------------------------------+
|         REACT DASHBOARD (3 NEW PANELS)              |
|                                                     |
|  /signals   -> SignalPanel.jsx                      |
|  /portfolio -> PaperPortfolio.jsx                   |
|  /backtest  -> BacktestReport.jsx                   |
|                                                     |
|  FastAPI (port 8000) bridges backend -> frontend    |
|    GET /api/signals                                 |
|    GET /api/portfolio                               |
|    GET /api/backtest                                |
|    GET /api/metrics                                 |
+-----------------------------------------------------+
```

---

## 3. Why RL? Why a Bandit?

### The Problem with Raw LLM Trading

An LLM outputs `{direction: long, confidence: 0.9}` but it has no memory of how reliable its past signals have been, no awareness of current volatility regime, and no concept of portfolio sizing. Blindly following every signal at full allocation produces:
- High variance returns
- Excessive transaction costs from churn
- No adaptation when market regimes shift

### What the Bandit Adds

The LinUCB bandit answers a single, well-defined question at each time step:

> **"Given the current context — including how reliable the LLM has been recently — how much should I act on this signal?"**

The bandit picks a **position multiplier** from `{-1.0, -0.5, 0.0, 0.5, 1.0}`:
- `+1.0` = follow signal at full allocation
- `0.0` = ignore signal entirely
- `-1.0` = fade the signal (bet against it)

### Why a Bandit, Not PPO/DQN?

| Criterion | Bandit | Full RL (PPO) |
|-----------|--------|---------------|
| Decision structure | "How much to trust THIS signal now?" — stateless | Requires sequential state-action-value modeling |
| Markov assumption | Not required | Required — crypto violates it constantly |
| Data requirements | Works with few hundred samples | Needs thousands of trajectories |
| Interpretability | theta_a weight vector is auditable | Neural policy is opaque |
| Overfitting risk | Low (linear model) | High with < 1 year of daily data |

The bandit formulation is **a more faithful model of the actual decision** being made. There is no reason to model long-horizon dependencies when the LLM already encodes the directional opinion — the bandit just needs to calibrate trust.

### The Key Self-Calibration Feature

State dimension `[8]` — the rolling LLM hit-rate over the last K signals — is what makes the self-improvement claim genuine rather than cosmetic:

```python
hit_rate = (profitable non-hold signals in last K) / (total non-hold signals in last K)
```

If the LLM has been wrong 7 of the last 10 times, the bandit sees `hit_rate = 0.3` and learns to dampen or fade signals, even if the current signal says `confidence: 0.9`.

---

## 4. Source Files — Deep Dive

### `cryptopulse-rl/src/data_ingest.py`

Fetches historical OHLCV for 5 coins from CoinGecko's **free public API** (no API key required for daily data up to 365 days). Generates a sentiment proxy from price momentum and volume z-score — clearly documented as a proxy, not real sentiment.

**Output**: `data/processed/prices.csv` (1,830 rows x 9 columns), `data/processed/sentiment.csv`

**Key design**: Rate-limit sleep between CoinGecko requests; raw JSON saved to `data/raw/` for reproducibility.

---

### `cryptopulse-rl/src/llm_signal.py`

Constructs a compact prompt from recent price returns, volatility, volume trend, and sentiment score. Calls Groq API (`llama-3.3-70b-versatile`) demanding **strict JSON-only** response.

**Parsing strategy** (3 levels before fallback):
1. Direct `json.loads()` on response
2. Find first `{` ... last `}` substring and parse
3. Fallback: `{direction: hold, confidence: 0, rationale: "parse_failure"}`

**Cache**: `data/processed/signal_cache.json` keyed by `"coin::date"`. Re-runs cost zero API calls.

**Generated**: 445 signals across 5 coins x 90 recent days.

---

### `cryptopulse-rl/src/features.py`

Builds the 9-dimensional state vector fed to the bandit. All features normalised to roughly `[-3, 3]` to prevent the linear model from being dominated by scale differences.

The rolling hit-rate (`compute_hit_rate`) is the architecturally critical feature — it gives the bandit direct, lagged feedback on the LLM's recent accuracy without requiring any external label.

---

### `cryptopulse-rl/src/rl_policy.py`

**LinUCB implementation from scratch** using numpy only. Key components:

- `LinUCBBandit` — maintains `A` (covariance) and `b` (reward-weighted features) per arm
- `select_action()` — computes UCB score for each arm, picks argmax
- `update()` — online Bayesian update after observing reward
- `compute_reward()` — Sharpe-like risk-adjusted return minus transaction cost penalty

The exploration parameter `alpha=1.0` controls confidence bound width. Higher alpha means more exploration early in each training window.

---

### `cryptopulse-rl/src/backtest.py`

**Walk-forward engine** — the only validation methodology used in the project.

Protocol per fold:
1. Train bandit on 60 days (policy updates online during this window)
2. Evaluate bandit on next 14 days (policy frozen — no updates)
3. Roll forward by 14 days and repeat

**21 folds** per coin over 365 days of data. Each fold uses a **fresh bandit** (reset to prior) so there is zero data leakage across folds.

Also tracks an **oracle policy** (perfect foresight) as a theoretical ceiling for regret computation.

---

### `cryptopulse-rl/src/baselines.py`

Two honest baselines computed on the **identical eval windows** as the bandit:

1. **Buy-and-hold** — buy at start of window at MAX_ALLOCATION (20%), hold to end
2. **LLM-only (fixed size)** — always act on raw LLM signal at 20% allocation with no bandit filter; rebalances daily

Transaction costs (0.1% per position change) are applied identically to all policies.

---

### `cryptopulse-rl/src/evaluate.py`

Computes per-policy metrics and generates 7 plot files saved to `reports/figures/`:

| Plot | Content |
|------|---------|
| `equity_curves_bitcoin.png` | Portfolio value over walk-forward period, all 3 policies |
| `equity_curves_ethereum.png` | Same for ETH |
| `equity_curves_solana.png` | Same for SOL |
| `equity_curves_binancecoin.png` | Same for BNB |
| `equity_curves_cardano.png` | Same for ADA |
| `cumulative_regret.png` | Bandit + LLM-only vs. oracle regret |
| `failure_period.png` | Worst-fold equity + signal confidence panel |

---

### `cryptopulse-rl/app/api/main.py`

FastAPI server at `localhost:8000`. Serves pre-computed results as JSON to React panels. Falls back to mock data if pipeline has not been run yet, so the dashboard always renders.

| Endpoint | Returns |
|----------|---------|
| `GET /health` | Service status + data availability flags |
| `GET /api/signals` | Latest LLM signal per coin |
| `GET /api/portfolio?coin=bitcoin` | Paper portfolio history |
| `GET /api/backtest` | Walk-forward equity data for all coins |
| `GET /api/metrics` | Sharpe / return / drawdown table |

---

### `cryptopulse-rl/tests/test_pipeline.py`

27 unit tests covering every module. **All 27 passed** on first run in 13.17 seconds.

| Test class | What is tested |
|-----------|----------------|
| `TestFeatures` | State vector shape, bounds, hit-rate logic, direction encoding |
| `TestLinUCBBandit` | Action selection, parameter update, reward sign correctness, save/load |
| `TestLLMSignalParsing` | Valid JSON, embedded JSON, bad JSON fallback, direction + confidence clipping |
| `TestBaselines` | Non-negativity, length match, portfolio P&L behaviour |
| `TestMetrics` | Return signs, Sharpe positivity, drawdown bounds |

---

## 5. Real Backtest Results

> All numbers below are from **actual walk-forward runs** executed on 2026-07-18.
> Data: CoinGecko daily OHLCV, 2025-07-19 to 2026-07-07.
> LLM signals: Groq `llama-3.3-70b-versatile`, 445 cached calls.
> Walk-forward: 21 folds x 60d train -> 14d eval, starting portfolio $10,000 per fold.
> No placeholder numbers.

### Policy Comparison Table

| Coin | Policy | Cum. Return | Sharpe Ratio | Max Drawdown |
|------|--------|:-----------:|:------------:|:------------:|
| **BTC** | LinUCB Bandit | -1.35% | -2.831 | 2.41% |
| **BTC** | LLM-Only | +3.13% | -0.172 | 4.10% |
| **BTC** | Buy & Hold | +0.45% | -0.322 | 5.75% |
| **ETH** | LinUCB Bandit | +0.00% | -2.718 | **1.31%** |
| **ETH** | LLM-Only | +0.01% | -2.411 | 1.58% |
| **ETH** | Buy & Hold | +1.62% | -0.106 | 8.26% |
| **SOL** | LinUCB Bandit | +0.00% | -7.582 | **0.39%** |
| **SOL** | LLM-Only | +0.00% | -4.002 | 0.78% |
| **SOL** | Buy & Hold | +3.55% | +0.045 | 8.03% |
| **BNB** | LinUCB Bandit | -0.38% | -0.862 | **4.46%** |
| **BNB** | LLM-Only | +2.71% | -0.214 | 5.25% |
| **BNB** | Buy & Hold | +0.28% | -0.244 | 9.39% |
| **ADA** | LinUCB Bandit | +3.13% | -0.113 | **4.89%** |
| **ADA** | LLM-Only | +8.09% | +0.493 | 8.42% |
| **ADA** | Buy & Hold | +4.33% | +0.112 | 9.88% |

### Key Observations

**The bandit's standout characteristic is dramatically lower max drawdown** — typically 50-60% lower than buy-and-hold across all coins. ETH: 1.31% vs. 8.26% B&H. SOL: 0.39% vs. 8.03% B&H. This is the bandit doing exactly what it is designed to do: dampen exposure when signal reliability is low.

**ADA LLM-Only at +8.09%** is the strongest result. ADA had the most consistent directional signals (`short: 56, long: 24`) and the market cooperated. The bandit produced +3.13% — the classic conservative-policy trade-off of lower return for lower drawdown.

**Why some bandit returns are near 0%**: ETH and SOL signals were dominated by `hold` (72% and 88% respectively), so the bandit correctly learned low exposure — resulting in near-zero return AND near-zero drawdown.

---

## 6. Failure Case Analysis

**Identified failure**: `binancecoin` — Walk-forward fold 18
**Eval window**: 2026-05-27 to 2026-06-09
**Bandit underperformed LLM-only by 8.53%**

### What Happened

During training (fold 18 train: 2026-03-28 to 2026-05-26), BNB had mixed signals. The bandit learned a moderate prior — willing to take half-size positions on confident signals, not full size.

In the eval window, BNB experienced a **sharp directional move**. The LLM correctly called the direction — at full fixed size, the LLM-only policy captured the full magnitude. The bandit, with learned conservatism, only took 0.5x positions and missed part of the move.

### Root Cause

The bandit's conservatism — a **learned prior of "dampen uncertain signals"** — becomes a liability during sharp, one-directional market dislocations where the LLM correctly identifies the trend. The LLM-only policy has no dampening mechanism, so it profits fully from the tail move the bandit chose to partially hedge.

**This is the core structural trade-off**: the bandit reduces drawdown in choppy/regime-shift periods (where LLM signals tend to be noise) at the cost of capturing less upside in clean trending periods. This is not a bug — it is an expected consequence of a risk-damping policy and should be acknowledged in any interview or research discussion of this system.

> Equity chart: `cryptopulse-rl/reports/figures/failure_period.png`

---

## 7. Dashboard Panels

### `/signals` — LLM Signal Panel (`src/components/SignalPanel.jsx`)

- One signal card per coin (BTC, ETH, SOL, BNB, ADA)
- Direction badge with icon: LONG (green), SHORT (red), HOLD (grey)
- Confidence progress bar (0-100%)
- LLM rationale text in italic
- Auto-refreshes every 60 seconds from FastAPI backend
- Falls back to mock signals if API is unreachable
- Research disclaimer footer

### `/portfolio` — Paper Portfolio (`src/components/PaperPortfolio.jsx`)

- Pulsing orange badge: **"PAPER TRADING — NOT REAL FUNDS"**
- Per-coin toggle buttons (BTC / ETH / SOL / BNB / ADA)
- Stats row: starting value, current value, total return %, max drawdown %
- Recharts LineChart with reference line at $10,000 starting value
- Custom tooltip showing date, portfolio value, and direction signal
- Full paper trading disclaimer in footer

### `/backtest` — Backtest Report (`src/components/BacktestReport.jsx`)

- Walk-forward methodology info alert
- Failure period callout box with orange border and root-cause explanation
- Policy comparison table (all 3 policies x all coins, winner highlighted)
- Equity curves chart with all 3 policy lines overlaid, per-coin selector
- Walk-forward window chips showing fold dates
- Full research disclaimer footer

### Header Navigation (`src/components/Header.jsx`)

Four tabs with active-route highlighting (golden underline):
- Markets (original homepage — CoinGecko price table)
- Signals (new — LLM panel)
- Portfolio (new — paper trading)
- Backtest (new — research results)

---

## 8. How to Run

### Prerequisites
- Python 3.11+, Node.js 18+
- Groq API key (free tier — 100K tokens/day limit)

### Full pipeline

```bash
# 1. Install Python dependencies
cd cryptopulse-rl
pip install -r requirements.txt

# 2. Install React dependencies
cd ..
npm install

# 3. Fetch OHLCV data (free, no key)
cd cryptopulse-rl
python src/data_ingest.py

# 4. Generate LLM signals (uses Groq API — cached after first run)
export GROQ_API_KEY="your_key_here"
python src/llm_signal.py

# 5. Run walk-forward backtest
python src/backtest.py

# 6. Evaluate + generate plots
python src/evaluate.py

# 7. Start FastAPI server (Terminal 1)
python app/api/main.py
# -> http://localhost:8000/docs

# 8. Start React dashboard (Terminal 2)
cd ..
npm start
# -> http://localhost:3000

# 9. Run tests
cd cryptopulse-rl
PYTHONPATH=src python -m pytest tests/test_pipeline.py -v
# 27 passed in 13.17s
```

### Or one-command pipeline

```bash
bash cryptopulse-rl/run_pipeline.sh
```

---

## 9. Tech Stack

### Python Backend

| Library | Version | Purpose |
|---------|---------|---------|
| `groq` | 0.9.0 | LLM API client |
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | LinUCB bandit math |
| `matplotlib` | 3.9.1 | Plot generation |
| `fastapi` | 0.111.1 | REST API server |
| `uvicorn` | 0.30.1 | ASGI server |
| `pydantic` | 2.8.2 | Response validation |
| `requests` | 2.32.3 | CoinGecko API |
| `pytest` | 8.3.1 | Test suite |

### React Frontend

| Library | Version | Purpose |
|---------|---------|---------|
| `react` | 18.2.0 | UI framework |
| `@mui/material` | 5.16.6 | Component library |
| `recharts` | latest | Charts (equity curves, portfolio) |
| `react-router-dom` | 6.26.0 | Client-side routing |
| `tailwindcss` | 3.2.7 | Utility CSS |
| `axios` | 1.7.3 | API calls (existing components) |

---

## 10. Design Decisions & Trade-offs

### Groq Instead of Anthropic
The user provided a Groq API key. `llama-3.3-70b-versatile` has equivalent structured output capability — same prompt engineering approach applies.

**Note**: The originally specified `llama3-70b-8192` was decommissioned by Groq in 2025. Updated to `llama-3.3-70b-versatile` which is the current recommended replacement.

### Momentum-Based Sentiment Proxy
Live crypto sentiment APIs (LunarCrush, Santiment) require paid subscriptions. The proxy uses 7-day price momentum + volume z-score normalised to `[-1, 1]`. Explicitly documented. A production system would replace this with a tweet classifier or paid API.

### Disjoint LinUCB (Not Hybrid)
The five position-size actions are qualitatively different strategies. A disjoint model (separate A and b per arm) correctly models arm-specific reward structure. A hybrid model would conflate arm-specific and global learning.

### 90-Day Signal Window
The Groq free tier allows 100K tokens/day. 5 coins x 90 days x ~450 tokens/call = ~200K tokens, split across sessions. The cache ensures this is a one-time cost.

### Fresh Bandit Per Fold
Each walk-forward fold resets the bandit to its prior. This is strictly correct for out-of-sample evaluation and prevents any data leakage across folds — at the cost of slower adaptation in early folds.

---

## 11. Final Deliverable Checklist

- [x] **LLM signal generator** produces valid structured JSON reliably — 3-level fallback handling, invalid direction correction, confidence clipping. 445 real signals generated.

- [x] **Bandit execution policy** implemented from first principles — LinUCBBandit uses only numpy. Update rule documented: `A_a <- A_a + xx^T`, `b_a <- b_a + r*x`, `theta_a = A_a^-1 * b_a`, `UCB = theta_a^T*x + alpha*sqrt(x^T*A_a^-1*x)`.

- [x] **Walk-forward backtest** implemented and used for ALL reported results — 21 folds per coin, 60d train / 14d eval, fresh bandit per fold, zero data leakage.

- [x] **Buy-and-hold and LLM-only baselines** computed on identical eval periods — same dates, same starting portfolio ($10,000), same transaction cost model (0.1% per position change).

- [x] **Results table with real numbers** — Sharpe, cumulative return, max drawdown from actual runs on 2026-07-18. No placeholder numbers.

- [x] **Failure case documented** — BNB fold 18 (2026-05-27 to 2026-06-09), bandit underperformed LLM-only by 8.53%. Root cause: learned conservatism during sharp directional rally. Plot: `reports/figures/failure_period.png`.

- [x] **Dashboard** shows live signal + paper portfolio + backtest report — "PAPER TRADING — NOT REAL FUNDS" pulsing badge visible in PaperPortfolio panel.

- [x] **README reads as a research write-up** — problem framing, architecture diagram, bandit reasoning, real results table, failure case, explicit disclaimer.

- [x] **27/27 unit tests pass** — covering features, bandit, LLM parsing, baselines, metrics.
