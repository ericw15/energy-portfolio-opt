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
    Markowitz_Portfolio,
    PCA_Historical_Mean_Strategy,
)

from .backtest import BacktestResult
from .experiment_core import (
    ExperimentOutputs,
    assemble_experiment_outputs,
    run_labelled_strategies,
    save_standard_summary_products,
)
from .statistics import run_pre_specified_return_comparisons
from .xle_data import (
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
    covariance, ordinary PCA, EWMA, and EWMA-PCA.
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
        f"EWMA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe": EWMA_Portfolio(
            risk_free_rate, ewma_half_life
        ),
        f"EWMA-PCA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe": EWMAPCA_Historical_Mean_Strategy(
            risk_free_rate, ewma_half_life, num_principal_components
        ),
    }
    strategy_backtests = run_labelled_strategies(
        backtest_panel,
        {
            label: strategy.weights_from_returns
            for label, strategy in strategies.items()
        },
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
    )
    outputs = assemble_experiment_outputs(
        strategy_backtests,
        asset_returns,
        baseline_returns,
        risk_free_rate=risk_free_rate,
    )
    statistical_tests = run_pre_specified_return_comparisons(
        outputs.daily_returns,
        {
            "EWMA-PCA covariance versus PCA covariance": (
                f"EWMA-PCA covariance ({ewma_half_life}-day half-life) / Historical Means Sharpe",
                "PCA covariance / Historical Means Sharpe",
            )
        },
        hac_lag=hac_lag,
    )
    return XLECovarianceExperimentResult(
        daily_returns=outputs.daily_returns,
        cumulative_returns=outputs.cumulative_returns,
        performance_metrics=outputs.performance_metrics,
        implementation_metrics=outputs.implementation_metrics,
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
        "statistical_tests": output_directory / "statistical-tests.csv",
    }
    save_xle_covariance_growth_chart(result, paths["growth_comparison"])
    paths.update(
        save_standard_summary_products(
            ExperimentOutputs(
                daily_returns=result.daily_returns,
                cumulative_returns=result.cumulative_returns,
                performance_metrics=result.performance_metrics,
                implementation_metrics=result.implementation_metrics,
                strategy_backtests=result.strategy_backtests,
            ),
            output_directory,
        )
    )
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
