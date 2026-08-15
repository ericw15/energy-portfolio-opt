"""Dedicated XLE experiment for the portfolio-objective risk question.

This suite holds the PCA covariance model, historical-mean return estimate,
asset universe, estimation window, and rebalance schedule fixed. It changes
only whether empirical expected-tail-loss is added to the risk denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from port_opt.strategy import (
    PCA_Historical_Mean_Strategy,
    TailAdjustedSharpePCA_Historical_Mean_Strategy,
)

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

DEFAULT_TAIL_LOSS_WEIGHTS = (1.0, 0.1, 0.01)


@dataclass(frozen=True)
class XLETailRiskExperimentResult:
    """Comparable outputs and audit trails for the objective comparison."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtests: dict[str, BacktestResult]


def _validate_tail_loss_weights(
    tail_loss_weights: Sequence[float],
) -> tuple[float, ...]:
    weights = tuple(float(weight) for weight in tail_loss_weights)
    if not weights:
        raise ValueError("tail_loss_weights must not be empty")
    if not np.isfinite(weights).all():
        raise ValueError("tail_loss_weights must contain only finite values")
    if len(set(weights)) != len(weights):
        raise ValueError("tail_loss_weights must not contain duplicates")
    return weights


def run_xle_tail_risk_experiment(
    *,
    training_start: str = "2021-01-01",
    backtest_start: str = "2024-01-01",
    end_date: str = "2026-08-01",
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04 / 252,
    num_principal_components: int | None = None,
    cvar_percentile: float = 0.95,
    tail_loss_weights: Sequence[float] = DEFAULT_TAIL_LOSS_WEIGHTS,
    tail_loss_weight: float | None = None,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> XLETailRiskExperimentResult:
    """Compare ordinary and tail-adjusted PCA maximum-Sharpe portfolios.

    The ordinary strategy maximizes (w' mu - r_f) / sigma(w). Each
    tail-adjusted strategy instead maximizes
    (w' mu - r_f) / (sigma(w) + lambda L_alpha(w)), using one lambda from
    tail_loss_weights. L_alpha(w) is positive empirical expected tail loss of
    in-sample portfolio returns and alpha is cvar_percentile. The singular
    tail_loss_weight argument remains a compatibility override for running
    exactly one tail-adjusted candidate.
    """
    if tail_loss_weight is not None:
        if tuple(tail_loss_weights) != DEFAULT_TAIL_LOSS_WEIGHTS:
            raise ValueError(
                "pass either tail_loss_weight or tail_loss_weights, not both"
            )
        selected_tail_loss_weights = _validate_tail_loss_weights((tail_loss_weight,))
    else:
        selected_tail_loss_weights = _validate_tail_loss_weights(tail_loss_weights)
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
        "PCA covariance / Maximum Sharpe": PCA_Historical_Mean_Strategy(
            risk_free_rate, num_principal_components
        ),
        **{
            f"PCA covariance / Tail-adjusted Sharpe (lambda={weight:g})": TailAdjustedSharpePCA_Historical_Mean_Strategy(
                risk_free_rate,
                tail_loss_weight=weight,
                cvar_percentile=cvar_percentile,
                num_principal_components=num_principal_components,
            )
            for weight in selected_tail_loss_weights
        },
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
    return XLETailRiskExperimentResult(
        daily_returns=outputs.daily_returns,
        cumulative_returns=outputs.cumulative_returns,
        performance_metrics=outputs.performance_metrics,
        implementation_metrics=outputs.implementation_metrics,
        strategy_backtests=strategy_backtests,
    )


def save_xle_tail_risk_growth_chart(
    result: XLETailRiskExperimentResult, output_path: str | Path
) -> None:
    """Write the sole standard visual product for this objective suite."""
    ax = result.cumulative_returns.plot(
        figsize=(12, 6), title="XLE tail-risk objective comparison"
    )
    ax.set_ylabel("Growth of $1 (gross)")
    ax.set_xlabel("Date")
    ax.figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def save_xle_tail_risk_experiment_visuals(
    result: XLETailRiskExperimentResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write growth, risk/return, and implementation comparisons plus tables."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "growth_comparison": output_directory / "growth-comparison.png",
    }
    save_xle_tail_risk_growth_chart(result, paths["growth_comparison"])
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
    experiment = run_xle_tail_risk_experiment(num_principal_components=1)
    visual_paths = save_xle_tail_risk_experiment_visuals(
        experiment, "tail_risk_research_outputs"
    )
    print(experiment.performance_metrics)
    print(experiment.implementation_metrics)
    print(*visual_paths.values(), sep="\n")
