"""Development experiment for selecting the PCA covariance dimension.

This module varies only the number of retained PCA components. It is intended
for an earlier development period: select and lock a component count here before
using it in a separate final evaluation experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from port_opt.strategy import PCA_Historical_Mean_Strategy

from .backtest import BacktestResult
from .experiment_core import (
    ExperimentOutputs,
    assemble_experiment_outputs,
    run_labelled_strategies,
    save_standard_summary_products,
)
from .xle_data import (
    BASELINE_TICKER,
    DEFAULT_LOOKBACK_PERIODS,
    DEFAULT_REBALANCE_FREQUENCY,
    XLE_TICKERS,
    load_xle_returns,
    select_backtest_panel,
)

DEFAULT_NUM_PRINCIPAL_COMPONENTS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class XLEPCADimensionExperimentResult:
    """Comparable results and audit trails across a PCA component-count grid."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtests: dict[str, BacktestResult]


def _validate_component_grid(
    num_principal_components: Sequence[int], maximum_components: int
) -> tuple[int, ...]:
    component_grid = tuple(num_principal_components)
    if not component_grid:
        raise ValueError("num_principal_components must not be empty")
    if len(set(component_grid)) != len(component_grid):
        raise ValueError("num_principal_components must not contain duplicates")
    if any(
        not isinstance(component_count, int)
        or isinstance(component_count, bool)
        or not 1 <= component_count <= maximum_components
        for component_count in component_grid
    ):
        raise ValueError(
            "each component count must be an integer between 1 and "
            f"{maximum_components}"
        )
    return component_grid


def run_xle_pca_dimension_experiment(
    *,
    training_start: str = "2018-01-01",
    backtest_start: str = "2021-01-01",
    end_date: str = "2023-12-31",
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04 / 252,
    num_principal_components: Sequence[int] = DEFAULT_NUM_PRINCIPAL_COMPONENTS,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> XLEPCADimensionExperimentResult:
    """Compare fixed PCA component counts on a development-only period.

    All optimized strategies use the same PCA covariance construction,
    historical mean expected returns, optimizer, universe, and rolling schedule.
    Only the number of retained principal components varies. Defaults end before
    the 2024 onward final-evaluation period used by the main experiments.
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
    component_grid = _validate_component_grid(
        num_principal_components,
        maximum_components=min(effective_lookback, len(backtest_panel.columns)),
    )
    strategies = {
        f"PCA covariance ({component_count} component{'s' if component_count != 1 else ''}) / Historical Means Sharpe": PCA_Historical_Mean_Strategy(
            risk_free_rate, component_count
        )
        for component_count in component_grid
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
    return XLEPCADimensionExperimentResult(
        daily_returns=outputs.daily_returns,
        cumulative_returns=outputs.cumulative_returns,
        performance_metrics=outputs.performance_metrics,
        implementation_metrics=outputs.implementation_metrics,
        strategy_backtests=strategy_backtests,
    )


def save_xle_pca_dimension_growth_chart(
    result: XLEPCADimensionExperimentResult, output_path: str | Path
) -> None:
    """Write the PCA component-count cumulative-return comparison."""
    ax = result.cumulative_returns.plot(
        figsize=(14, 7), title="XLE PCA component-count comparison"
    )
    ax.set_ylabel("Growth of $1 (gross)")
    ax.set_xlabel("Date")
    ax.figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def save_xle_pca_dimension_experiment_visuals(
    result: XLEPCADimensionExperimentResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write standard growth, trade-off, implementation, and tabular products."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "growth_comparison": output_directory / "growth-comparison.png",
    }
    save_xle_pca_dimension_growth_chart(result, paths["growth_comparison"])
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
    return paths


if __name__ == "__main__":
    experiment = run_xle_pca_dimension_experiment()
    visual_paths = save_xle_pca_dimension_experiment_visuals(
        experiment, "pca_dimension_research_outputs"
    )
    print(experiment.performance_metrics)
    print(experiment.implementation_metrics)
    print(*visual_paths.values(), sep="\n")
