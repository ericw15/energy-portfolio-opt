"""Dedicated XLE experiment for the portfolio-objective risk question.

This suite holds the PCA covariance model, historical-mean return estimate,
asset universe, estimation window, and rebalance schedule fixed. It changes
only whether empirical expected-tail-loss is added to the risk denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from port_opt.strategy import (
    PCA_Historical_Mean_Strategy,
    TailAdjustedSharpePCA_Historical_Mean_Strategy,
)

from .backtest import BacktestResult, run_walk_forward_backtest
from .metrics import summarize_implementation, summarize_performance
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
class XLETailRiskExperimentResult:
    """Comparable outputs and audit trails for the objective comparison."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtests: dict[str, BacktestResult]


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
    tail_loss_weight: float = 1.0,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> XLETailRiskExperimentResult:
    """Compare ordinary and tail-adjusted PCA maximum-Sharpe portfolios.

    The ordinary strategy maximizes ``(w' mu - r_f) / sigma(w)``. The
    tail-adjusted strategy instead maximizes
    ``(w' mu - r_f) / (sigma(w) + lambda L_alpha(w))``. Here ``L_alpha(w)``
    is positive empirical expected tail loss of in-sample portfolio returns,
    ``alpha`` is ``cvar_percentile``, and ``lambda`` is ``tail_loss_weight``.
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
        "PCA covariance / Maximum Sharpe": PCA_Historical_Mean_Strategy(
            risk_free_rate, num_principal_components
        ),
        f"PCA covariance / Tail-adjusted Sharpe (lambda={tail_loss_weight:g})": TailAdjustedSharpePCA_Historical_Mean_Strategy(
            risk_free_rate,
            tail_loss_weight=tail_loss_weight,
            cvar_percentile=cvar_percentile,
            num_principal_components=num_principal_components,
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
        raise RuntimeError("objective strategies produced different backtest dates")
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
    return XLETailRiskExperimentResult(
        daily_returns=daily_returns,
        cumulative_returns=(1.0 + daily_returns).cumprod(),
        performance_metrics=summarize_performance(
            daily_returns, risk_free_rate=risk_free_rate
        ),
        implementation_metrics=summarize_implementation(strategy_backtests),
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
        "performance_metrics": output_directory / "performance-metrics.csv",
        "implementation_metrics": output_directory / "implementation-metrics.csv",
        "risk_return_comparison": output_directory / "risk-return-comparison.png",
        "implementation_comparison": output_directory / "implementation-comparison.png",
    }
    save_xle_tail_risk_growth_chart(result, paths["growth_comparison"])
    save_risk_return_comparison(
        result.performance_metrics, paths["risk_return_comparison"]
    )
    save_implementation_comparison(
        result.implementation_metrics, paths["implementation_comparison"]
    )
    result.performance_metrics.to_csv(paths["performance_metrics"])
    result.implementation_metrics.to_csv(paths["implementation_metrics"])
    return paths


if __name__ == "__main__":
    experiment = run_xle_tail_risk_experiment(num_principal_components=1)
    visual_paths = save_xle_tail_risk_experiment_visuals(
        experiment, "tail_risk_research_outputs"
    )
    print(experiment.performance_metrics)
    print(experiment.implementation_metrics)
    print(*visual_paths.values(), sep="\n")
