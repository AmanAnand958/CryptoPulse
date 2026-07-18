# Prompt: Build "CryptoPulse RL" — LLM Signal Generator + RL Execution Layer (Backtest System)

Copy everything below into your agent.

---

## Project Brief

Extend the existing **CryptoPulse** dashboard into a research/backtesting system called `cryptopulse-rl`. The system has two layers:

1. **Signal layer (LLM)**: given market data + news/sentiment, an LLM outputs a structured trade signal (direction + confidence + rationale).
2. **Execution layer (RL)**: a contextual bandit / lightweight RL policy decides *how much to trust and act on* that signal — position sizing, not price prediction — learning over time from realized risk-adjusted returns.

This is explicitly a **backtesting and research system, not a live trading bot**. Do not build or claim any live capital execution. Every result must be clearly labeled as backtested, with walk-forward validation, not a single train/test split.

## Repo Structure

```
cryptopulse-rl/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # price history, news/sentiment
│   └── processed/
├── src/
│   ├── data_ingest.py       # historical price + news/sentiment fetch
│   ├── llm_signal.py        # LLM-based signal generator
│   ├── features.py          # state features for RL layer
│   ├── rl_policy.py         # bandit/PPO execution policy
│   ├── backtest.py          # walk-forward backtest engine
│   ├── baselines.py         # buy-and-hold, fixed-size LLM-only baseline
│   └── evaluate.py          # metrics, plots, comparison tables
├── app/
│   └── dashboard/           # existing CryptoPulse React dashboard, extended with:
│       ├── SignalPanel.jsx      # live LLM signal + rationale
│       ├── PaperPortfolio.jsx   # simulated portfolio value over time
│       └── BacktestReport.jsx   # walk-forward results view
├── tests/
│   └── test_pipeline.py
└── reports/
    └── figures/
```

## Step-by-step build instructions

### 1. Data ingestion (`src/data_ingest.py`)
- Pull historical OHLCV data for 3-5 major coins (e.g., BTC, ETH, SOL) at a reasonable granularity (hourly or daily) from a free source (e.g., CoinGecko or Binance public API, whichever is reachable in this environment).
- Pull or approximate news/sentiment: either a free crypto news/sentiment API, or fall back to a public sentiment dataset (e.g., a Kaggle crypto-tweets sentiment dataset) if live sentiment isn't accessible. Clearly document which source was actually used in the README.
- Store cleaned, timestamp-aligned price + sentiment data in `data/processed/`.

### 2. LLM signal generator (`src/llm_signal.py`)
- For each time step, construct a compact prompt containing: recent price action summary (e.g., last N candles' returns, volatility, volume trend), and recent relevant news/sentiment snippets.
- Call an LLM (use the Anthropic API — Claude — via a simple wrapper) with a **strict structured output** requirement: respond only in JSON with fields `{direction: long|short|hold, confidence: 0-1, rationale: string}`. Enforce this with explicit prompt instructions and JSON parsing with retries/fallback (default to `hold, confidence 0` on parse failure — never let a malformed response silently break the pipeline).
- Cache LLM signals per (coin, timestamp) to avoid redundant calls during repeated backtest runs — this matters both for cost and for reproducibility.
- Log rationale text alongside every signal; this is used later for the dashboard's "why" panel and for qualitative failure analysis.

### 3. State features for RL layer (`src/features.py`)
Combine into a state vector per time step:
- LLM signal direction + confidence
- Recent realized volatility, momentum (e.g., 1h/24h/7d returns)
- Current position size and unrealized PnL
- A rolling measure of "LLM signal reliability" — e.g., a trailing hit-rate of the LLM's last K signals — so the policy has direct evidence of how trustworthy the signal has recently been. This is the key feature that makes the "self-improving" framing genuine rather than cosmetic.

### 4. Execution policy (`src/rl_policy.py`)
- Implement as a **contextual bandit** (LinUCB or Thompson Sampling) first — this is the primary, most defensible version. Action space: discrete position-size multipliers applied to the LLM's suggested direction, e.g., `{-1x, -0.5x, 0x (ignore signal), 0.5x, 1x}` of a fixed max allocation.
- Reward: risk-adjusted return over a fixed forward window (e.g., Sharpe-like ratio: forward return divided by realized volatility over that window), minus a transaction-cost penalty proportional to position change, so the policy is penalized for churning.
- Optionally, as a stretch extension clearly marked as secondary, implement a simple PPO agent on the same state/action/reward setup and compare against the bandit — but do not make this the primary deliverable; the bandit is the honest, defensible core.
- Explicitly document why a bandit is a more appropriate choice than full MDP-based RL here: crypto price series only weakly satisfy Markov/stationarity assumptions, and the bandit formulation ("how much should I trust this signal right now") is a more faithful model of the actual decision being made than a full sequential MDP would be. State this reasoning in the README — it is a real strength of the project, not a limitation to hide.

### 5. Walk-forward backtest engine (`src/backtest.py`)
- **Do not use a single train/test split.** Implement walk-forward validation: train/update the policy on a rolling window (e.g., 60 days), evaluate forward on the next window (e.g., 14 days), then roll forward and repeat across the full historical period. This must be the only validation methodology used or reported.
- Track, per step: LLM signal, chosen action, realized return, cumulative portfolio value, cumulative regret vs. an oracle policy (perfect foresight, for reference only — not as a claimed achievable target).

### 6. Baselines (`src/baselines.py`)
Implement and evaluate, on the exact same walk-forward periods:
- **Buy-and-hold** for each coin.
- **LLM-only, fixed-size**: always act on the LLM's raw signal at a fixed position size, no RL/bandit layer.
- The entire thesis of this project rests on showing the bandit layer improves on the LLM-only baseline — this comparison is the single most important result in the whole project and must be computed honestly from real backtest runs.

### 7. Evaluation (`src/evaluate.py`)
Report, per policy (buy-and-hold, LLM-only, bandit-execution, and PPO if built):
- Cumulative return, Sharpe ratio, max drawdown, over the full walk-forward period.
- Cumulative regret plot (bandit vs. LLM-only vs. oracle).
- **A clearly identified failure period**: find and show at least one walk-forward window where the RL-execution policy underperforms the LLM-only baseline, and give a real explanation (e.g., a sharp regime shift, a period where the LLM's sentiment signal was stale or contradicted by price action). This is required, not optional — a project with zero acknowledged failure modes is far less credible than one that shows exactly where and why it breaks.
- Save all plots to `reports/figures/`.

### 8. Dashboard extension (`app/dashboard/`)
Extend the existing CryptoPulse React/Tailwind dashboard with three additions:
- **SignalPanel**: shows the current LLM signal, confidence, and rationale text for each tracked coin, updating live.
- **PaperPortfolio**: a simulated (paper) portfolio value chart, driven by the bandit policy's decisions on live/recent data — clearly labeled "Paper Trading — Not Real Funds" in the UI itself, not just the README.
- **BacktestReport**: a view rendering the walk-forward backtest results (equity curves for all policies, the comparison table, and the identified failure-period callout).

### 9. README.md — write as a research write-up
Structure:
- **Problem framing**: one paragraph on why naively "let an LLM trade" is unreliable, and why a learned execution layer that calibrates trust in the signal is a more defensible design.
- **Architecture diagram** (ASCII or Mermaid is fine) showing LLM signal layer → state construction → bandit execution layer → backtest/paper portfolio.
- **Why a bandit, not full RL**: the reasoning from step 4, stated explicitly.
- **Results table**: buy-and-hold vs. LLM-only vs. bandit-execution (and PPO if built), with Sharpe, cumulative return, max drawdown, all computed from actual walk-forward runs — no placeholder numbers.
- **Failure case section**: the identified underperformance window, with a plotted example and explanation.
- **Explicit disclaimer**: this is a backtested research/portfolio project, not a live trading system; past backtest performance does not indicate future results; not financial advice.
- **How to run**: data ingestion, signal generation (note LLM API cost/rate-limit considerations), backtest, dashboard.
- **Tech stack**: Python, Anthropic API (Claude), scikit-learn/numpy for the bandit, pandas, React, Tailwind.

### 10. requirements.txt
Pin: `anthropic`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`, `pydantic`, `fastapi` (if you add a small API layer between backend and dashboard), `pytest`. For the dashboard, standard `react`, `tailwindcss`, plus a charting library (`recharts` or similar) already likely present from the original CryptoPulse build.

## Quality bar / constraints
- Every metric in the README must come from an actual executed backtest run — no fabricated or estimated numbers.
- The system must never claim, imply, or be capable of live trade execution with real funds. Paper trading only, labeled as such in both code and UI.
- Walk-forward validation is mandatory; a single train/test split anywhere in the evaluation is not acceptable for this project's claims to hold up.
- The failure-case section is mandatory, not optional polish — it is what makes the project credible under interview questioning.
- Keep the LLM signal generator's prompt and parsing logic clearly separated from the RL policy logic — an interviewer should be able to see exactly where "language model opinion" ends and "learned execution decision" begins.
- After building, run the full pipeline, confirm it executes end-to-end, and paste real resulting metrics and at least one real plot reference into the README.

## Final deliverable checklist
- [ ] LLM signal generator produces valid structured JSON reliably (with fallback handling)
- [ ] Bandit execution policy implemented from first principles (not just a library call), with documented update rule
- [ ] Walk-forward backtest implemented and used for all reported results
- [ ] Buy-and-hold and LLM-only baselines computed on identical periods
- [ ] Results table with real numbers (Sharpe, return, drawdown) for all policies
- [ ] At least one clearly documented failure/underperformance case
- [ ] Dashboard shows live signal + paper portfolio + backtest report, with "Paper Trading" clearly labeled
- [ ] README reads as a research write-up, including the explicit "why bandit, not full RL" reasoning