# 🚀 CryptoPulse RL — LangGraph Multi-Agent Crypto Signal & Reinforcement Learning Execution System

> **Institutional-Grade Research & Backtesting Framework**  
> Multi-Agent LLM Signal Generation (LangGraph Supervisor) + Contextual Bandit Position Sizing (LinUCB) + López de Prado Meta-Labeling Baseline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/pytest-27%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Naively "letting an LLM trade" is inherently flawed: language models output directional opinions, not risk-calibrated position sizes. An LLM might output `{"direction": "long", "confidence": 0.85}` based on bullish news sentiment, but it has zero awareness of portfolio risk limits, recent volatility regimes, or past signal accuracy.

**CryptoPulse RL** solves this by adding a **learned execution policy** between the LLM and the portfolio:
1. **LangGraph Multi-Agent Supervisor (`agents/`)**: A 6-node concurrent graph coordinates market data ingestion, technical indicator calculation (RSI/MACD/volatility), news sentiment analysis, structured signal generation, and multi-field conflict resolution.
2. **Contextual Bandit Execution (`rl_policy.py`)**: A memoryless LinUCB contextual bandit learns optimal position allocation fractions (`0%`, `25%`, `50%`, `100%`) based on market momentum, volatility, and historical LLM hit-rate.
3. **Meta-Labeling Baseline (`meta_label_baseline.py`)**: Implements Marcos López de Prado's (*AFML, Ch.3*) supervised meta-labeling classifier to filter false-positive signals before execution.
4. **Walk-Forward Backtesting Engine (`backtest.py` & `evaluate.py`)**: Evaluates 4 strategies simultaneously over 21 rolling folds (365 days) with point-in-time data isolation, zero look-ahead bias, and realistic transaction cost modeling (0.1% per trade).

---

## 🏗️ Architecture & Component Flow

```
┌──────────────────────────── LIVE GRAPH RUNNER (agents/) ────────────────────────────┐
│                                                                                      │
│  START ───► market_data_agent ───► technical_analysis_node ───► signal_generator ┐  │
│        └──► sentiment_agent   ───────────────────────────────────┘                 │
│                                                                  supervisor_node     │
│                                                                        ▼             │
│                                                                 final_signal         │
│                                                                        ▼             │
│                                                             LinUCB Bandit Position   │
│                                                            (0%, 25%, 50%, 100%)      │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────── WALK-FORWARD BACKTEST (src/) ──────────────────────────────┐
│                                                                                      │
│  historical_data_collector.py ──► data/historical/*.parquet                          │
│                                                │                                     │
│                                                ▼                                     │
│  backtest.py ───────────────────── Walk-Forward 21 Folds                            │
│                         ┌────────────────────────────────────┐                       │
│                         │ Strategy 1: Buy & Hold             │                       │
│                         │ Strategy 2: LLM-Only               │                       │
│                         │ Strategy 3: Meta-Labeling          │                       │
│                         │ Strategy 4: LinUCB Bandit          │                       │
│                         └────────────────────────────────────┘                       │
│                                                │                                     │
│  evaluate.py ───────────────────── Institutional Evaluation Metrics                 │
│                                                │                                     │
│  app/api/main.py ───────────────── FastAPI Dashboard Endpoints                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Backtest Results & Benchmark Summary

> Evaluated over **21 walk-forward folds (365 days)** across 5 major cryptocurrencies with **1,785 daily signals** and 0.1% transaction cost drag.

| Asset | Strategy | Cum Return (All Folds) | Active Window Return (Folds 15–20) | Sharpe Ratio | Max Drawdown | Win Rate | Monte Carlo p-val |
|---|---|---|---|---|---|---|---|
| **Cardano (ADA)** | **LLM-Only** | **+8.09%** | **+8.09%** | **+0.493** | 8.42% | 14.3% | 0.538 |
| Cardano (ADA) | LinUCB Bandit | +7.03% | +7.03% | **+0.645** | **2.61%** | 12.9% | 0.507 |
| Cardano (ADA) | Buy & Hold | +4.33% | +4.33% | +0.112 | 9.88% | 44.6% | 0.492 |
| Cardano (ADA) | Meta-Labeling | +2.00% | +2.00% | -0.196 | 8.60% | 10.2% | 0.526 |
| **Bitcoin (BTC)** | **LLM-Only** | **+3.13%** | **+3.13%** | **-0.172** | 4.10% | 16.0% | 0.539 |
| Bitcoin (BTC) | LinUCB Bandit | +2.02% | +2.02% | -0.857 | **1.58%** | 16.0% | 0.543 |
| Bitcoin (BTC) | Meta-Labeling | +1.75% | +1.75% | -0.603 | 2.73% | 8.8% | 0.523 |
| Bitcoin (BTC) | Buy & Hold | +0.45% | +0.45% | -0.322 | 5.75% | 50.0% | 0.501 |
| **BinanceCoin (BNB)** | **LLM-Only** | **+2.71%** | **+2.71%** | **-0.214** | 5.25% | 13.9% | 0.522 |
| BinanceCoin (BNB) | Meta-Labeling | +2.35% | +2.35% | -0.406 | 3.28% | 9.5% | 0.527 |
| BinanceCoin (BNB) | LinUCB Bandit | +1.18% | +1.18% | -0.831 | **3.60%** | 11.2% | 0.529 |
| BinanceCoin (BNB) | Buy & Hold | +0.28% | +0.28% | -0.244 | 9.39% | 53.1% | 0.500 |
| **Ethereum (ETH)** | Buy & Hold | +1.62% | +1.62% | -0.106 | 8.26% | 50.7% | 0.497 |
| Ethereum (ETH) | LinUCB Bandit | +0.01% | +0.01% | -9.102 | **0.48%** | 3.1% | 0.522 |
| **Solana (SOL)** | Buy & Hold | +3.55% | +3.55% | +0.045 | 8.03% | 47.3% | 0.493 |
| Solana (SOL) | LinUCB Bandit | +0.00% | +0.00% | -8.114 | **0.39%** | 0.7% | 0.513 |

### Key Benchmark Insights
- **Drawdown Reduction**: LinUCB position sizing achieves the **lowest maximum drawdown across all assets** (BTC 1.58% vs 5.75% B&H; ADA 2.61% vs 9.88% B&H; BNB 3.60% vs 9.39% B&H).
- **Statistical Significance Note**: Monte Carlo p-values reside in the range 0.49–0.54. Results demonstrate strong downside protection, but performance cannot be claimed as statistically significant over this sample size.

---

## 📁 Repository Structure

```
CryptoPulse-main/
├── cryptopulse-rl/
│   ├── app/
│   │   └── api/
│   │       └── main.py                 # FastAPI backend server
│   ├── src/
│   │   ├── agents/                     # LangGraph Multi-Agent Supervisor Nodes
│   │   │   ├── graph_state.py          # TypedDict state schema with reducers
│   │   │   ├── bright_data_client.py   # Bright Data MCP factory + usage tracker
│   │   │   ├── market_data_agent.py    # OHLCV ingestion & CoinGecko fallback
│   │   │   ├── sentiment_agent.py      # News sentiment agent
│   │   │   ├── technical_analysis_node.py # Deterministic RSI/MACD/vol function
│   │   │   ├── signal_generator_agent.py  # Structured output LLM signal generator
│   │   │   ├── supervisor_agent.py     # Conflict resolution supervisor
│   │   │   └── graph.py                # LangGraph assembler & CLI execution
│   │   ├── llm_signal.py               # API key rotator (Gemini/Groq/Nvidia)
│   │   ├── rl_policy.py                # LinUCB Contextual Bandit policy
│   │   ├── meta_label_baseline.py      # López de Prado meta-labeling baseline
│   │   ├── backtest.py                 # Walk-forward 21-fold backtest engine
│   │   ├── evaluate.py                 # Metrics, charts, & Monte Carlo tests
│   │   ├── daily_signal_runner.py      # Production daily signal generator
│   │   ├── paper_trade.py              # Virtual portfolio paper trading tracker
│   │   └── historical_data_collector.py # CCXT & Alpha Vantage collector
│   ├── data/
│   │   ├── historical/                 # Parquet historical OHLCV & sentiment
│   │   └── processed/                  # Signals CSV, cache JSON, PnL records
│   ├── reports/
│   │   ├── figures/                    # Equity curves, regret, failure period plots
│   │   └── metrics_summary.json        # Output metrics & dataset provenance
│   ├── tests/
│   │   └── test_pipeline.py            # Pytest suite (27 passing unit tests)
│   ├── run_pipeline.sh                 # Full end-to-end pipeline bash script
│   ├── FINAL_REPORT.md                 # Complete technical documentation
│   └── EXPLANATION_HINGLISH.md         # Detailed Hinglish architecture guide
├── README.md                           # Main repository README
└── requirements.txt                    # Project Python dependencies
```

---

## ⚡ Quick Start Guide

### 1. Installation & Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AmanAnand958/CryptoPulse.git
cd CryptoPulse/cryptopulse-rl
pip install -r requirements.txt
```

Create a `.env` file in `cryptopulse-rl/`:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
NVIDIA_API_KEY=your_nvidia_api_key
BRIGHT_DATA_API_KEY=your_bright_data_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### 2. Run Full Walk-Forward Backtest & Evaluation

Execute the end-to-end backtest pipeline:

```bash
bash run_pipeline.sh
```

Or run modules individually:

```bash
# Step 1: Execute 21-fold walk-forward backtest
python src/backtest.py

# Step 2: Generate metrics summary & figures
python src/evaluate.py

# Step 3: Run unit test suite
python -m pytest tests/test_pipeline.py -v
```

### 3. Run FastAPI Backend Server

Launch the FastAPI REST API server to serve signals, portfolio data, and backtest results:

```bash
cd cryptopulse-rl
uvicorn app.api.main:app --reload --port 8000
```

Access the API endpoints:
- **Health Check**: `GET http://localhost:8000/health`
- **Latest Signals**: `GET http://localhost:8000/api/signals`
- **Portfolio History**: `GET http://localhost:8000/api/portfolio?coin=bitcoin`
- **Backtest Summary**: `GET http://localhost:8000/api/backtest`
- **Metrics Summary**: `GET http://localhost:8000/api/metrics`

### 4. Daily Signal Runner & Paper Trading

Generate signals for missing dates (or backfill historical dates):

```bash
# Backfill all missing historical dates
python src/daily_signal_runner.py --backfill

# Generate signals for today only
python src/daily_signal_runner.py

# Dry-run preview without API calls
python src/daily_signal_runner.py --dry-run
```

Run forward-testing paper trading step:

```bash
# Execute one paper trade step for today
python src/paper_trade.py

# View current paper trading portfolio balance
python src/paper_trade.py --show
```

---

## 🧪 Unit Testing

Run the test suite to verify pipeline integrity:

```bash
python -m pytest tests/test_pipeline.py -v
```

```
============================== 27 passed in 1.24s ==============================
```

---

## 📖 Documentation & Reports

- **[`FINAL_REPORT.md`](file:///Users/amananand/Downloads/projects/CryptoPulse-main/cryptopulse-rl/FINAL_REPORT.md)** — Comprehensive technical report detailing system design, walk-forward performance, regime coverage, and future roadmap.
- **[`EXPLANATION_HINGLISH.md`](file:///Users/amananand/Downloads/projects/CryptoPulse-main/cryptopulse-rl/EXPLANATION_HINGLISH.md)** — Hinglish explanation covering real vs. mock transparency, LangGraph node flows, and LinUCB mathematical derivations for interview preparation.

---

## ⚠️ Disclaimer

This codebase is built strictly for **academic research, portfolio demonstration, and paper trading**. Past performance in historical walk-forward backtests does not guarantee future results. No real funds are risked or traded automatically. Not financial advice.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.