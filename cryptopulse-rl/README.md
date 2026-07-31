# 🚀 CryptoPulse RL — LangGraph Multi-Agent Crypto Signal & Reinforcement Learning Execution System

> **Institutional-Grade Research & Backtesting Framework**  
> Multi-Agent LLM Signal Generation (LangGraph Supervisor) + Contextual Bandit Position Sizing (LinUCB) + López de Prado Meta-Labeling Baseline

---

## 📌 Executive Summary

Naively "letting an LLM trade" is inherently flawed: language models output directional opinions, not risk-calibrated position sizes. An LLM might output `{"direction": "long", "confidence": 0.85}` based on bullish news sentiment, but it has zero awareness of portfolio risk limits, recent volatility regimes, or past signal accuracy.

**CryptoPulse RL** solves this by adding a **learned execution policy** between the LLM and the portfolio:
1. **LangGraph Multi-Agent Supervisor (`agents/`)**: A 6-node concurrent graph coordinates market data ingestion, technical indicator calculation (RSI/MACD/volatility), news sentiment analysis, structured signal generation, and multi-field conflict resolution.
2. **Contextual Bandit Execution (`rl_policy.py`)**: A memoryless LinUCB contextual bandit learns optimal position allocation fractions (`0%`, `25%`, `50%`, `100%`) based on market momentum, volatility, and historical LLM hit-rate.
3. **Meta-Labeling Baseline (`meta_label_baseline.py`)**: Implements Marcos López de Prado's (*AFML, Ch.3*) supervised meta-labeling classifier to filter false-positive signals before execution.
4. **Walk-Forward Backtesting Engine (`backtest.py` & `evaluate.py`)**: Evaluates 4 strategies simultaneously over 21 rolling folds (365 days) with point-in-time data isolation, zero look-ahead bias, and realistic transaction cost modeling (0.1% per trade).

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

---

## ⚡ Quick Commands

```bash
# 1. Run full walk-forward backtest
python src/backtest.py

# 2. Compute evaluation metrics & charts
python src/evaluate.py

# 3. Start FastAPI server
uvicorn app.api.main:app --reload

# 4. Daily signal runner (backfill all missing dates)
python src/daily_signal_runner.py --backfill

# 5. Paper trading portfolio tracker
python src/paper_trade.py --show

# 6. Run unit tests
python -m pytest tests/test_pipeline.py -v
```

For full details, see [`FINAL_REPORT.md`](FINAL_REPORT.md) and [`EXPLANATION_HINGLISH.md`](EXPLANATION_HINGLISH.md).
