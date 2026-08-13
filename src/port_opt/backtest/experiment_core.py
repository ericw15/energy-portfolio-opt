"""Shared execution and output assembly for XLE research experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_walk_forward_backtest
from .metrics import summarize_implementation, summarize_performance
from .visualizations import save_implementation_comparison, save_risk_return_comparison

WeightEstimator = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ExperimentOutputs:
    """Shared daily returns, summary tables, and strategy audit trails."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtests: dict[str, BacktestResult]


def run_labelled_strategies(
    backtest_panel: pd.DataFrame,
    estimators: Mapping[str, WeightEstimator],
    *,
    lookback_periods: int,
    rebalance_frequency: int,
) -> dict[str, BacktestResult]:
    """Run a labelled strategy map under one common walk-forward protocol."""
    if not estimators:
        raise ValueError("estimators must not be empty")
    return {
        label: run_walk_forward_backtest(
            backtest_panel,
            estimator,
            lookback_periods=lookback_periods,
            rebalance_frequency=rebalance_frequency,
        )
        for label, estimator in estimators.items()
    }


def assemble_experiment_outputs(
    strategy_backtests: dict[str, BacktestResult],
    asset_returns: pd.DataFrame,
    baseline_returns: pd.Series,
    *,
    risk_free_rate: float,
) -> ExperimentOutputs:
    """Verify dates and add common equal-weight and XLE return baselines."""
    if not strategy_backtests:
        raise ValueError("strategy_backtests must not be empty")
    evaluation_index = next(iter(strategy_backtests.values())).portfolio_returns.index
    if not all(
        evaluation_index.equals(backtest.portfolio_returns.index)
        for backtest in strategy_backtests.values()
    ):
        raise RuntimeError("strategies produced different backtest dates")
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
    return ExperimentOutputs(
        daily_returns=daily_returns,
        cumulative_returns=(1.0 + daily_returns).cumprod(),
        performance_metrics=summarize_performance(
            daily_returns, risk_free_rate=risk_free_rate
        ),
        implementation_metrics=summarize_implementation(strategy_backtests),
        strategy_backtests=strategy_backtests,
    )


def save_standard_summary_products(
    outputs: ExperimentOutputs, output_directory: str | Path
) -> dict[str, Path]:
    """Write shared risk/return, implementation, and metric-table products."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "performance_metrics": output_directory / "performance-metrics.csv",
        "implementation_metrics": output_directory / "implementation-metrics.csv",
        "risk_return_comparison": output_directory / "risk-return-comparison.png",
        "implementation_comparison": output_directory / "implementation-comparison.png",
    }
    save_risk_return_comparison(
        outputs.performance_metrics, paths["risk_return_comparison"]
    )
    save_implementation_comparison(
        outputs.implementation_metrics, paths["implementation_comparison"]
    )
    outputs.performance_metrics.to_csv(paths["performance_metrics"])
    outputs.implementation_metrics.to_csv(paths["implementation_metrics"])
    return paths
