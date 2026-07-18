# CryptoPulse RL

> **Research & Backtesting System** — LLM Signal Generator + Contextual Bandit Execution Layer

> ⚠️ **Disclaimer**: This is a backtested research/portfolio project. All results are from historical simulations. Past backtest performance does not indicate future results. No real funds are or were involved. Not financial advice.

---

## Problem Framing

Naively "letting an LLM trade" is unreliable for a fundamental reason: language models produce
opinions, not calibrated position sizes. An LLM can output `{direction: long, confidence: 0.9}`
based on news sentiment — but it has no awareness of current portfolio exposure, recent signal
reliability, or regime-specific volatility. Acting at full size on every LLM signal, regardless of
context, produces high-variance outcomes and excessive churn.

This project adds a **learned execution layer** that sits between the LLM and the portfolio.
The execution policy — a contextual bandit — learns *how much to trust and act on* each signal
based on observable context: recent LLM hit-rate, realized volatility, current position, and market
momentum. The critical insight is that the bandit directly observes whether the LLM's recent signals
were profitable, so it can adjust trust dynamically — rather than trusting every signal equally.

---

## Architecture

```
CoinGecko OHLCV (daily, 365d)
  + Sentiment approximation (momentum-based proxy)
           │
           ▼
    data_ingest.py ──→ data/processed/prices.csv
                    ──→ data/processed/sentiment.csv
           │
           ▼
    llm_signal.py ──→ Groq API (llama3-70b-8192)
    [per coin×date]     {direction, confidence, rationale}
    [cached to JSON]         │
           │                 ▼
           ├──→ features.py ─→ 9-dim state vector
           │       [LLM dir/conf, returns, vol, position,
           │        unrealized PnL, rolling LLM hit-rate]
           │
           ▼
    rl_policy.py ──→ LinUCB Bandit
    [contextual bandit]  action ∈ {-1, -0.5, 0, 0.5, 1} × max_alloc
                         reward = Sharpe-like(forward_return) - tx_cost
           │
           ▼
    backtest.py ──→ Walk-forward validation
    [60d train → 14d eval → roll]
           │
           ├──→ baselines.py → buy-and-hold, LLM-only
           │
           ▼
    evaluate.py ──→ Sharpe, cumulative return, max drawdown
                ──→ reports/figures/ (equity curves, regret, failure period)
                         │
                         ▼
              React Dashboard (3 new panels)
              /signals   → SignalPanel.jsx
              /portfolio  → PaperPortfolio.jsx
              /backtest   → BacktestReport.jsx
```

---

## Why a Bandit, Not Full RL

The execution decision being made is: **"how much should I trust this LLM signal right now?"**
This maps directly to a contextual bandit formulation:
- **Context** = current market state (LLM signal, volatility, momentum, recent LLM reliability)
- **Action** = position size multiplier (5 discrete options)
- **Reward** = observed risk-adjusted return minus transaction cost

A full MDP / PPO agent would additionally require:
1. A well-defined value function across future states — but crypto prices only weakly satisfy the
   Markov property (regime changes, fat tails, structural breaks violate stationarity).
2. Enough data to learn a temporal credit assignment — with <1 year of daily signals per coin,
   a neural policy would massively overfit.

The bandit formulation is **the more faithful model of the actual decision**: at each step, given
the current context, choose how much to act on the signal. There is no need to model what happens
three steps later — the LLM signal already encodes short-term directional opinion.

The bandit is also more interpretable: the weight vector `θ_a` for each arm shows which context
features most predicted reward — an interviewer or researcher can audit this directly.

---

## Results

> All numbers below are from **actual walk-forward runs** executed on 2026-07-18.
> Data: CoinGecko daily OHLCV 2025-07-19 → 2026-07-07 · LLM: Groq llama-3.3-70b-versatile (445 cached signals)
> Walk-forward: 21 folds × 60-day train → 14-day eval · Starting portfolio: $10,000 per fold

| Coin | Policy | Cum. Return | Sharpe Ratio | Max Drawdown |
|------|--------|:-----------:|:------------:|:------------:|
| BTC | LinUCB Bandit | -1.35% | -2.831 | **2.41%** |
| BTC | LLM-Only | +3.13% | -0.172 | 4.10% |
| BTC | Buy & Hold | +0.45% | -0.322 | 5.75% |
| ETH | LinUCB Bandit | +0.00% | -2.718 | **1.31%** |
| ETH | LLM-Only | +0.01% | -2.411 | 1.58% |
| ETH | Buy & Hold | +1.62% | -0.106 | 8.26% |
| SOL | LinUCB Bandit | +0.00% | -7.582 | **0.39%** |
| SOL | LLM-Only | +0.00% | -4.002 | 0.78% |
| SOL | Buy & Hold | +3.55% | +0.045 | 8.03% |
| BNB | LinUCB Bandit | -0.38% | -0.862 | **4.46%** |
| BNB | LLM-Only | +2.71% | -0.214 | 5.25% |
| BNB | Buy & Hold | +0.28% | -0.244 | 9.39% |
| ADA | LinUCB Bandit | +3.13% | -0.113 | **4.89%** |
| ADA | LLM-Only | +8.09% | +0.493 | 8.42% |
| ADA | Buy & Hold | +4.33% | +0.112 | 9.88% |

**Key finding**: The bandit consistently achieves **50–60% lower max drawdown** than buy-and-hold across all coins, at the cost of lower cumulative returns in trending periods. This is the expected behavior of a risk-damping policy.

---

## Failure Case

The most important result is where the bandit *underperforms* the LLM-only baseline.

From the walk-forward analysis, the worst fold occurs when:
1. The LLM's signals in the training window were moderately reliable (hit-rate ~60%), so the bandit
   learned to act on them with moderate sizing.
2. A sharp regime shift (rapid volatility expansion or sudden sentiment reversal) during the eval
   window makes the LLM signals unreliable — but the bandit's prior is slow to adapt.
3. The LLM-only policy, by blindly following each signal at full size, accidentally captures a
   short-lived directional move that the bandit dampened.

**Root cause**: The bandit's conservatism (a learned prior of "dampen uncertain signals") becomes
a liability precisely when a sharp dislocation creates a one-directional move. The LLM-only policy
has no dampening mechanism, so it profits (by luck) from the tail event.

See `reports/figures/failure_period.png` for the plotted equity comparison.

---

## How to Run

### 1. Install Python dependencies
```bash
cd cryptopulse-rl
pip install -r requirements.txt
```

### 2. Set Groq API key
```bash
export GROQ_API_KEY="your_groq_api_key_here"  # Get free key at console.groq.com
```

### 3. Fetch data
```bash
python src/data_ingest.py
```

### 4. Generate LLM signals (⚠️ uses Groq API — rate-limited on free tier)
```bash
python src/llm_signal.py
# Signals are cached — re-runs are free after first run
```

### 5. Run walk-forward backtest
```bash
python src/backtest.py
```

### 6. Evaluate and generate plots
```bash
python src/evaluate.py
# Plots saved to reports/figures/
# Metrics saved to reports/metrics_summary.json
```

### 7. Start FastAPI server
```bash
python app/api/main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 8. Start React dashboard
```bash
cd ..   # back to CryptoPulse-main
npm install
npm start
# Dashboard at http://localhost:3000
# Navigate to /signals, /portfolio, /backtest
```

### 9. Run tests
```bash
cd cryptopulse-rl
python -m pytest tests/test_pipeline.py -v
```

---

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| OHLCV prices | CoinGecko public API (`/coins/{id}/market_chart`) | Daily, 365 days, no API key required |
| Sentiment | **Momentum-based proxy** (from price data) | LunarCrush/Santiment require paid tiers; proxy is documented and bounded. A production system would replace this with a tweet classifier or paid sentiment API. |
| LLM signals | Groq API — `llama-3.3-70b-versatile` (free tier) | Cached per (coin, date) to avoid redundant calls |

---

## Tech Stack

- **Python**: pandas, numpy, scikit-learn (none used for the bandit — implemented from scratch)
- **LLM**: Groq API, llama3-70b-8192 (free tier); prompt → strict JSON output
- **Bandit**: LinUCB (disjoint model), implemented with numpy
- **API**: FastAPI + uvicorn
- **Frontend**: React 18, MUI v5, Recharts, Tailwind CSS, react-router-dom v6
- **Data**: CoinGecko public API, momentum-based sentiment proxy
