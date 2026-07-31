#!/usr/bin/env bash
# run_pipeline.sh — Run the full CryptoPulse RL pipeline end-to-end
# Usage: bash run_pipeline.sh
set -e

cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "=============================================="
echo "  CryptoPulse RL — Full Pipeline Runner"
echo "=============================================="

export GROQ_API_KEY="${GROQ_API_KEY:?'Set GROQ_API_KEY env var before running'}"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo ""
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/5] Fetching OHLCV data from CoinGecko..."
python src/data_ingest.py

echo ""
echo "[3/5] Generating LLM signals via Groq API (uses cache if available)..."
python -c "
import sys; sys.path.insert(0, 'src')
from data_ingest import run_ingest
from llm_signal import generate_signals_for_dataframe
prices, sentiment = run_ingest()
signals = generate_signals_for_dataframe(prices, sentiment, lookback=8, use_cache=True)
print(f'Generated {len(signals)} signals for {signals[\"coin\"].nunique()} coins')
print(signals.groupby(\"coin\")[\"direction\"].value_counts())
"

echo ""
echo "[4/5] Running walk-forward backtest..."
python -c "
import sys, pandas as pd; sys.path.insert(0, 'src')
from pathlib import Path
from data_ingest import run_ingest
from backtest import run_full_backtest
prices, _ = run_ingest()
sig_path = Path('data/processed/signals.csv')
if not sig_path.exists():
    print('ERROR: signals.csv not found. Run step 3 first.'); sys.exit(1)
signals = pd.read_csv(sig_path, parse_dates=['date'])
results = run_full_backtest(prices, signals)
print('Backtest complete.')
"

echo ""
echo "[5/5] Computing metrics and generating plots..."
python -c "
import sys, json; sys.path.insert(0, 'src')
from pathlib import Path
from evaluate import run_evaluation
results_path = Path('data/processed/backtest_results.json')
if not results_path.exists():
    print('ERROR: backtest_results.json not found. Run step 4 first.'); sys.exit(1)
with open(results_path) as f:
    all_results = json.load(f)
summary = run_evaluation(all_results)
print('Evaluation complete.')
"

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Figures: reports/figures/"
echo "  Metrics: reports/metrics_summary.json"
echo ""
echo "  To start the API server:"
echo "    cd cryptopulse-rl && uvicorn app.api.main:app --reload"
echo ""
echo "  To run daily signal generation (backfill ALL missing dates):"
echo "    python src/daily_signal_runner.py --backfill"
echo ""
echo "  To run daily signal generation (today only):"
echo "    python src/daily_signal_runner.py"
echo ""
echo "  To view paper trading portfolio:"
echo "    python src/paper_trade.py --show"
echo ""
echo "  To run one paper trading step:"
echo "    python src/paper_trade.py"
echo ""
echo "  To run tests:"
echo "    python -m pytest tests/test_pipeline.py -v"
echo "=============================================="
