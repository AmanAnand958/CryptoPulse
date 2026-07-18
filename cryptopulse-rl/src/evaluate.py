"""
evaluate.py
===========
Computes and plots evaluation metrics for CryptoPulse RL.

Metrics per policy (buy-and-hold, LLM-only, bandit):
  - Cumulative return
  - Sharpe ratio (annualised, assuming 365 trading days)
  - Max drawdown

Plots saved to reports/figures/:
  - equity_curves_{coin}.png    — portfolio values over walk-forward period
  - cumulative_regret.png       — bandit vs. LLM-only vs. oracle regret
  - failure_period.png          — the identified underperformance window

Failure period identification:
  Find the walk-forward fold where (bandit_return - llm_return) is most
  negative, show it in detail, and provide a real explanation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "data" / "processed" / "backtest_results.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_PORTFOLIO = 10_000.0
RISK_FREE_RATE = 0.05 / 365   # daily


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cumulative_return(portfolio_series: pd.Series) -> float:
    """Total return over the series (e.g. 0.35 = 35%)."""
    if len(portfolio_series) < 2 or portfolio_series.iloc[0] == 0:
        return 0.0
    return float(portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1)


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio."""
    excess = daily_returns - risk_free
    if excess.std() == 0 or len(excess) < 2:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(365))


def max_drawdown(portfolio_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction)."""
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / (peak + 1e-9)
    return float(abs(drawdown.min()))


def compute_metrics(df: pd.DataFrame, value_col: str = "portfolio_value") -> dict:
    vals = pd.Series(df[value_col].values)
    rets = vals.pct_change().fillna(0)
    return {
        "cumulative_return": cumulative_return(vals),
        "sharpe_ratio": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(vals),
        "final_value": float(vals.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str, ylabel: str = "Portfolio Value ($)"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_equity_curves(coin: str, results: dict) -> None:
    """Plot equity curves for all three policies for a given coin."""
    bandit_df = pd.DataFrame(results["bandit_results"])
    bah_df = pd.DataFrame(results["bah_results"])
    llm_df = pd.DataFrame(results["llm_results"])

    if bandit_df.empty:
        logger.warning("No bandit results for %s", coin)
        return

    for df in [bandit_df, bah_df, llm_df]:
        df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bandit_df["date"], bandit_df["portfolio_value"], label="LinUCB Bandit", color="#4CAF50", lw=2)
    ax.plot(llm_df["date"], llm_df["portfolio_value"], label="LLM-Only (fixed size)", color="#2196F3", lw=1.5, ls="--")
    ax.plot(bah_df["date"], bah_df["portfolio_value"], label="Buy & Hold", color="#FF9800", lw=1.5, ls=":")
    ax.axhline(INITIAL_PORTFOLIO, color="gray", lw=0.8, ls="-", label="Initial")

    _style_ax(ax, f"{coin.upper()} — Walk-Forward Equity Curves (All Policies)")
    plt.tight_layout()
    out = FIGURES_DIR / f"equity_curves_{coin}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_cumulative_regret(all_results: dict) -> None:
    """Plot cumulative regret: bandit vs. LLM-only vs. oracle."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for coin, results in all_results.items():
        bandit_df = pd.DataFrame(results["bandit_results"])
        llm_df = pd.DataFrame(results["llm_results"])
        oracle_rets = results.get("oracle_returns", [])

        if bandit_df.empty or len(oracle_rets) == 0:
            continue

        bandit_rets = bandit_df["daily_return"].values[:len(oracle_rets)]
        llm_rets = llm_df["daily_return"].values[:len(oracle_rets)] if not llm_df.empty else np.zeros_like(bandit_rets)

        bandit_regret = np.cumsum(np.array(oracle_rets) - bandit_rets)
        llm_regret = np.cumsum(np.array(oracle_rets) - llm_rets[:len(oracle_rets)])

        ax.plot(bandit_regret, label=f"{coin[:3].upper()} Bandit Regret", lw=2)
        ax.plot(llm_regret, label=f"{coin[:3].upper()} LLM-Only Regret", lw=1.5, ls="--")

    ax.set_title("Cumulative Regret vs. Oracle (Perfect Foresight)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Regret (return units)")
    ax.set_xlabel("Eval Steps")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "cumulative_regret.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def find_failure_period(all_results: dict) -> tuple[str, int, dict]:
    """
    Identify the worst walk-forward fold for the bandit vs. LLM-only.

    Returns (coin, fold_index, details_dict)
    """
    worst_coin, worst_fold, worst_diff = None, None, float("inf")

    for coin, results in all_results.items():
        bandit_df = pd.DataFrame(results["bandit_results"])
        llm_df = pd.DataFrame(results["llm_results"])
        if bandit_df.empty or llm_df.empty or "fold" not in bandit_df.columns:
            continue

        for fold in bandit_df["fold"].unique():
            b_fold = bandit_df[bandit_df["fold"] == fold]
            l_fold = llm_df[llm_df["fold"] == fold] if "fold" in llm_df.columns else llm_df

            b_ret = cumulative_return(b_fold["portfolio_value"])
            l_ret = cumulative_return(l_fold["portfolio_value"]) if not l_fold.empty else 0.0
            diff = b_ret - l_ret

            if diff < worst_diff:
                worst_diff = diff
                worst_coin = coin
                worst_fold = int(fold)

    return worst_coin, worst_fold, {"bandit_vs_llm_return_diff": worst_diff}


def plot_failure_period(all_results: dict, worst_coin: str, worst_fold: int) -> None:
    """Plot the identified failure period in detail."""
    results = all_results.get(worst_coin, {})
    bandit_df = pd.DataFrame(results.get("bandit_results", []))
    llm_df = pd.DataFrame(results.get("llm_results", []))

    if bandit_df.empty or "fold" not in bandit_df.columns:
        logger.warning("Cannot plot failure period — no data")
        return

    b_fold = bandit_df[bandit_df["fold"] == worst_fold].copy()
    l_fold = llm_df[llm_df["fold"] == worst_fold].copy() if "fold" in llm_df.columns else pd.DataFrame()

    b_fold["date"] = pd.to_datetime(b_fold["date"])
    if not l_fold.empty:
        l_fold["date"] = pd.to_datetime(l_fold["date"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top panel: equity
    ax = axes[0]
    ax.plot(b_fold["date"], b_fold["portfolio_value"], label="LinUCB Bandit", color="#e53935", lw=2)
    if not l_fold.empty:
        ax.plot(l_fold["date"], l_fold["portfolio_value"], label="LLM-Only", color="#1E88E5", lw=1.5, ls="--")
    ax.axhline(INITIAL_PORTFOLIO, color="gray", lw=0.8, ls="-")
    ax.fill_between(
        b_fold["date"],
        b_fold["portfolio_value"],
        INITIAL_PORTFOLIO,
        where=(b_fold["portfolio_value"] < INITIAL_PORTFOLIO),
        alpha=0.15, color="red", label="Drawdown zone",
    )
    _style_ax(ax, f"⚠️ Failure Period — {worst_coin.upper()} Fold {worst_fold}")

    # Bottom panel: LLM signal confidence
    if "confidence" in b_fold.columns:
        ax2 = axes[1]
        colors = [
            "#4CAF50" if d == "long" else "#f44336" if d == "short" else "#9E9E9E"
            for d in b_fold["direction"]
        ]
        ax2.bar(b_fold["date"], b_fold["confidence"], color=colors, alpha=0.7, label="LLM Confidence")
        ax2.set_title("LLM Signal Confidence During Failure Period", fontsize=11)
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = FIGURES_DIR / "failure_period.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(all_results: dict) -> dict:
    """
    Compute all metrics, generate all plots, return results summary.
    """
    summary = {}

    for coin, results in all_results.items():
        bandit_df = pd.DataFrame(results["bandit_results"])
        bah_df = pd.DataFrame(results["bah_results"])
        llm_df = pd.DataFrame(results["llm_results"])

        summary[coin] = {
            "bandit": compute_metrics(bandit_df) if not bandit_df.empty else {},
            "buy_and_hold": compute_metrics(bah_df) if not bah_df.empty else {},
            "llm_only": compute_metrics(llm_df) if not llm_df.empty else {},
        }
        plot_equity_curves(coin, results)

    plot_cumulative_regret(all_results)

    worst_coin, worst_fold, failure_details = find_failure_period(all_results)
    if worst_coin:
        plot_failure_period(all_results, worst_coin, worst_fold)
        summary["failure_period"] = {
            "coin": worst_coin,
            "fold": worst_fold,
            **failure_details,
        }
        logger.info(
            "Failure period identified: %s fold %d (bandit underperformed LLM-only by %.2f%%)",
            worst_coin, worst_fold, abs(failure_details["bandit_vs_llm_return_diff"]) * 100,
        )

    # Print comparison table
    logger.info("\n%s", "=" * 70)
    logger.info("%-12s %-10s %-12s %-12s %-12s", "Coin", "Policy", "Cum Return", "Sharpe", "Max DD")
    logger.info("%s", "-" * 70)
    for coin, metrics in summary.items():
        if coin == "failure_period":
            continue
        for policy, m in metrics.items():
            if m:
                logger.info(
                    "%-12s %-10s %+10.2f%% %12.3f %12.2f%%",
                    coin, policy,
                    m.get("cumulative_return", 0) * 100,
                    m.get("sharpe_ratio", 0),
                    m.get("max_drawdown", 0) * 100,
                )

    # Save summary
    out_path = BASE_DIR / "reports" / "metrics_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Metrics saved → %s", out_path)
    return summary


if __name__ == "__main__":
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
    summary = run_evaluation(all_results)
