"""Dedicated XLE experiment for comparing covariance construction methods.

This intentionally separates covariance-estimator research from the main
commodity-feature experiment. Every optimized strategy uses the same full-window
historical mean returns, constraints, rebalance schedule, and asset universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from port_opt.strategy import (
    EWMA_Portfolio,
    EWMAPCA_Historical_Mean_Strategy,
    Ledoit_Wolf_Portfolio,
    Markowitz_Portfolio,
    PCA_Historical_Mean_Strategy,
)

from .backtest import BacktestResult, run_walk_forward_backtest
from .metrics import summarize_implementation, summarize_performance
from .statistics import run_pre_specified_return_comparisons
from .visualizations import save_implementation_comparison, save_risk_return_comparison
from .xle_experiment import (
    BASELINE_TICKER,
    DEFAULT_LOOKBACK_PERIODS,
    DEFAULT_REBALANCE_FREQUENCY,
    XLE_TICKERS,
    load_xle_returns,
    select_backtest_panel,
)


@dataclass(frozen=True)
class XLECovarianceExperimentResult:
    """Return series, comparable metrics, and per-strategy audit trails."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtests: dict[str, BacktestResult]
    statistical_tests: pd.DataFrame


def run_xle_covariance_experiment(
    *,
    training_start: str = "2021-01-01",
    backtest_start: str = "2024-01-01",
    end_date: str = "2026-08-01",
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04 / 252,
    num_principal_components: int | None = None,
    ewma_half_life: int = 63,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    hac_lag: int = 20,
) -> XLECovarianceExperimentResult:
    """Run a common-protocol XLE covariance-construction comparison.

    The optimized strategies differ only in covariance construction: sample
    covariance, ordinary PCA, Ledoit--Wolf, EWMA, and EWMA-PCA.
    ``ewma_half_life`` is measured in available trading observations. The two
    non-optimized comparison baselines are equal-weight constituents and XLE.
    """
    asset_returns, baseline_returns = load_xle_returns(
        training_start,
        end_date,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    backtest_panel, effective_lookback = select_backtest_panel(
        asset_returns, backtest_start, lookback_periods
    )
    strategies = {
        "Markowitz sample covariance / Historical Means Sharpe": Markowitz_Portfolio(
            risk_free_rate
        ),
        "PCA covariance / Historical Means Sharpe": PCA_Historical_Mean_Strategy(
            risk_free_rate, num_principal_components
        ),
        "Ledoit-Wolf covariance / Historical Means Sharpe": Ledoit_Wolf_Portfolio(
            risk_free_rate
        ),
        f"EWMA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe": EWMA_Portfolio(
            risk_free_rate, ewma_half_life
        ),
        f"EWMA-PCA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe": EWMAPCA_Historical_Mean_Strategy(
            risk_free_rate, ewma_half_life, num_principal_components
        ),
    }
    strategy_backtests = {
        label: run_walk_forward_backtest(
            backtest_panel,
            strategy.weights_from_returns,
            lookback_periods=effective_lookback,
            rebalance_frequency=rebalance_frequency,
        )
        for label, strategy in strategies.items()
    }
    evaluation_index = next(iter(strategy_backtests.values())).portfolio_returns.index
    if not all(
        evaluation_index.equals(backtest.portfolio_returns.index)
        for backtest in strategy_backtests.values()
    ):
        raise RuntimeError("covariance strategies produced different backtest dates")

    daily_returns = pd.DataFrame(
        {
            **{
                label: backtest.portfolio_returns
                for label, backtest in strategy_backtests.items()
            },
            "Equal-weight XLE constituents": asset_returns.loc[evaluation_index].mean(
                axis=1
            ),
            "XLE baseline": baseline_returns.loc[evaluation_index],
        }
    )
    statistical_tests = run_pre_specified_return_comparisons(
        daily_returns,
        {
            "EWMA-PCA covariance versus PCA covariance": (
                f"EWMA-PCA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe",
                "PCA covariance / Historical Means Sharpe",
            )
        },
        hac_lag=hac_lag,
    )
    return XLECovarianceExperimentResult(
        daily_returns=daily_returns,
        cumulative_returns=(1.0 + daily_returns).cumprod(),
        performance_metrics=summarize_performance(
            daily_returns, risk_free_rate=risk_free_rate
        ),
        implementation_metrics=summarize_implementation(strategy_backtests),
        strategy_backtests=strategy_backtests,
        statistical_tests=statistical_tests,
    )


def save_xle_covariance_growth_chart(
    result: XLECovarianceExperimentResult, output_path: str | Path
) -> None:
    """Write the sole standard visual product for this experiment."""
    ax = result.cumulative_returns.plot(
        figsize=(14, 7), title="XLE covariance-construction comparison"
    )
    ax.set_ylabel("Growth of $1 (gross)")
    ax.set_xlabel("Date")
    ax.figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def save_xle_covariance_experiment_visuals(
    result: XLECovarianceExperimentResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write growth, risk/return, and implementation comparisons plus tables."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "growth_comparison": output_directory / "growth-comparison.png",
        "performance_metrics": output_directory / "performance-metrics.csv",
        "implementation_metrics": output_directory / "implementation-metrics.csv",
        "risk_return_comparison": output_directory / "risk-return-comparison.png",
        "implementation_comparison": output_directory / "implementation-comparison.png",
        "statistical_tests": output_directory / "statistical-tests.csv",
    }
    save_xle_covariance_growth_chart(result, paths["growth_comparison"])
    save_risk_return_comparison(
        result.performance_metrics, paths["risk_return_comparison"]
    )
    save_implementation_comparison(
        result.implementation_metrics, paths["implementation_comparison"]
    )
    result.performance_metrics.to_csv(paths["performance_metrics"])
    result.implementation_metrics.to_csv(paths["implementation_metrics"])
    result.statistical_tests.to_csv(paths["statistical_tests"])
    return paths


if __name__ == "__main__":
    experiment = run_xle_covariance_experiment(num_principal_components=1)
    visual_paths = save_xle_covariance_experiment_visuals(
        experiment, "covariance_research_outputs"
    )
    print(experiment.performance_metrics)
    print(experiment.implementation_metrics)
    print(*visual_paths.values(), sep="\n")
