"""Compare an XLE constituent portfolio with equal weight and the XLE ETF.

This is the preserved, runnable energy-portfolio research experiment. It uses the
package backtester rather than duplicating its walk-forward logic, while keeping
the original research question and constituent universe visible and editable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from port_opt.strategy import (
    Commodity_Factor_Strategy,
    Markowitz_Portfolio,
    PCA_Commodity_Factor_Strategy,
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
from .visualizations import (
    save_covariance_comparison,
    save_return_histograms,
)
from .xle_data import (
    BASELINE_TICKER,
    DEFAULT_COMMODITY_FACTORS,
    DEFAULT_LOOKBACK_PERIODS,
    DEFAULT_REBALANCE_FREQUENCY,
    XLE_TICKERS,
    load_commodity_factor_returns,
    load_xle_returns,
    select_backtest_panel,
)


@dataclass(frozen=True)
class XLEExperimentResult:
    """Comparable daily and cumulative returns plus the strategy audit trail."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_metrics: pd.DataFrame
    implementation_metrics: pd.DataFrame
    strategy_backtest: BacktestResult
    statistical_tests: pd.DataFrame | None = None
    markowitz_backtest: BacktestResult | None = None
    commodity_factor_backtest: BacktestResult | None = None
    commodity_only_backtest: BacktestResult | None = None
    pca_covariance: pd.DataFrame | None = None
    markowitz_covariance: pd.DataFrame | None = None
    commodity_factor_covariance: pd.DataFrame | None = None
    commodity_only_covariance: pd.DataFrame | None = None
    covariance_as_of_date: pd.Timestamp | None = None


def run_xle_pca_historical_mean_experiment(
    *,
    training_start: str = "2021-01-01",
    backtest_start: str = "2024-01-01",
    end_date: str = "2026-08-01",
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04 / 252,
    num_principal_components: int | None = None,
    covariance_rebalance_index: int = 0,
    commodity_factors: Mapping[str, str] = DEFAULT_COMMODITY_FACTORS,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    hac_lag: int = 20,
) -> XLEExperimentResult:
    """Compare covariance estimators with shared historical-mean returns.

    All portfolio strategies receive the same trailing observations at each
    rebalance and retain historical-mean expected returns. They differ only in
    covariance estimation: PCA factor covariance, observed U.S. commodity
    factors only, PCA plus commodity factors, or sample covariance.
    ``covariance_rebalance_index`` selects the recorded rebalance whose
    in-sample covariances are retained for visual diagnostics.
    ``num_principal_components`` applies identically to the PCA-only and
    PCA-plus-commodity strategies.
    """
    asset_returns, baseline_returns = load_xle_returns(
        training_start,
        end_date,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    commodity_returns = load_commodity_factor_returns(
        training_start,
        end_date,
        commodity_factors=commodity_factors,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    common_index = asset_returns.index.intersection(commodity_returns.index)
    if common_index.empty:
        raise ValueError("equity and commodity returns have no dates in common")
    asset_returns = asset_returns.loc[common_index]
    baseline_returns = baseline_returns.loc[common_index]
    commodity_returns = commodity_returns.loc[common_index]
    # Pass the selected prior history to the generic engine so its first OOS
    # observation is backtest_start (or the next available trading observation).
    backtest_panel, effective_lookback = select_backtest_panel(
        asset_returns, backtest_start, lookback_periods
    )
    backtest_commodity_panel = commodity_returns.loc[backtest_panel.index]
    strategy = PCA_Historical_Mean_Strategy(
        risk_free_rate=risk_free_rate,
        num_principal_components=num_principal_components,
    )
    markowitz_strategy = Markowitz_Portfolio(risk_free_rate=risk_free_rate)
    commodity_strategy = PCA_Commodity_Factor_Strategy(
        risk_free_rate=risk_free_rate,
        num_principal_components=num_principal_components,
    )
    commodity_only_strategy = Commodity_Factor_Strategy(risk_free_rate=risk_free_rate)
    strategy_backtests = run_labelled_strategies(
        backtest_panel,
        {
            "PCA factor / Historical Means Sharpe": strategy.weights_from_returns,
            "PCA + U.S. commodity factors / Historical Means Sharpe": (
                lambda in_sample: commodity_strategy.weights_from_equity_and_factor_returns(
                    in_sample, backtest_commodity_panel.loc[in_sample.index]
                )
            ),
            "U.S. commodity factors only / Historical Means Sharpe": (
                lambda in_sample: commodity_only_strategy.weights_from_equity_and_factor_returns(
                    in_sample, backtest_commodity_panel.loc[in_sample.index]
                )
            ),
            "Markowitz / Historical Means Sharpe": markowitz_strategy.weights_from_returns,
        },
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
    )
    strategy_backtest = strategy_backtests["PCA factor / Historical Means Sharpe"]
    commodity_factor_backtest = strategy_backtests[
        "PCA + U.S. commodity factors / Historical Means Sharpe"
    ]
    commodity_only_backtest = strategy_backtests[
        "U.S. commodity factors only / Historical Means Sharpe"
    ]
    markowitz_backtest = strategy_backtests["Markowitz / Historical Means Sharpe"]
    if covariance_rebalance_index < 0:
        raise ValueError("covariance_rebalance_index must be non-negative")
    try:
        covariance_record = strategy_backtest.records[covariance_rebalance_index]
    except IndexError as error:
        raise ValueError(
            "covariance_rebalance_index is outside the backtest"
        ) from error
    covariance_returns = backtest_panel.loc[
        covariance_record.in_sample_start : covariance_record.in_sample_end
    ]
    outputs = assemble_experiment_outputs(
        strategy_backtests,
        asset_returns,
        baseline_returns,
        risk_free_rate=risk_free_rate,
    )
    statistical_tests = run_pre_specified_return_comparisons(
        outputs.daily_returns,
        {
            "PCA + commodity factors versus PCA": (
                "PCA + U.S. commodity factors / Historical Means Sharpe",
                "PCA factor / Historical Means Sharpe",
            )
        },
        hac_lag=hac_lag,
    )
    return XLEExperimentResult(
        daily_returns=outputs.daily_returns,
        cumulative_returns=outputs.cumulative_returns,
        performance_metrics=outputs.performance_metrics,
        implementation_metrics=outputs.implementation_metrics,
        statistical_tests=statistical_tests,
        strategy_backtest=strategy_backtest,
        markowitz_backtest=markowitz_backtest,
        commodity_factor_backtest=commodity_factor_backtest,
        commodity_only_backtest=commodity_only_backtest,
        pca_covariance=strategy.get_covariance_matrix(None, None, covariance_returns),
        markowitz_covariance=markowitz_strategy.get_covariance_matrix(
            None, None, covariance_returns
        ),
        commodity_factor_covariance=commodity_strategy.get_covariance_matrix_with_factors(
            covariance_returns,
            backtest_commodity_panel.loc[covariance_returns.index],
        ),
        commodity_only_covariance=commodity_only_strategy.get_covariance_matrix_with_factors(
            covariance_returns,
            backtest_commodity_panel.loc[covariance_returns.index],
        ),
        covariance_as_of_date=covariance_record.in_sample_end,
    )


def save_growth_chart(result: XLEExperimentResult, output_path: str | Path) -> None:
    """Write the experiment's cumulative-return comparison to a PNG file."""
    ax = result.cumulative_returns.plot(
        figsize=(12, 6), title="XLE portfolio comparison"
    )
    ax.set_ylabel("Growth of $1 (gross)")
    ax.set_xlabel("Date")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path)
    plt.close(ax.figure)


def save_xle_experiment_visuals(
    result: XLEExperimentResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write all standard charts from one completed XLE experiment result."""
    if (
        result.pca_covariance is None
        or result.markowitz_covariance is None
        or result.commodity_factor_covariance is None
        or result.commodity_only_covariance is None
        or result.covariance_as_of_date is None
    ):
        raise ValueError(
            "result has incomplete covariance diagnostics; "
            "run the historical-mean experiment first"
        )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "growth_comparison": output_directory / "growth-comparison.png",
        "markowitz_covariance_comparison": output_directory
        / "markowitz-covariance-comparison.png",
        "commodity_covariance_comparison": output_directory
        / "commodity-covariance-comparison.png",
        "commodity_only_covariance_comparison": output_directory
        / "commodity-only-covariance-comparison.png",
    }
    save_growth_chart(result, paths["growth_comparison"])
    save_covariance_comparison(
        result.pca_covariance,
        result.markowitz_covariance,
        paths["markowitz_covariance_comparison"],
        as_of_date=result.covariance_as_of_date,
    )
    save_covariance_comparison(
        result.pca_covariance,
        result.commodity_factor_covariance,
        paths["commodity_covariance_comparison"],
        as_of_date=result.covariance_as_of_date,
        comparison_label="PCA + U.S. commodity-factor covariance",
    )
    save_covariance_comparison(
        result.pca_covariance,
        result.commodity_only_covariance,
        paths["commodity_only_covariance_comparison"],
        as_of_date=result.covariance_as_of_date,
        comparison_label="U.S. commodity-factor-only covariance",
    )
    histogram_paths = save_return_histograms(
        result.daily_returns, output_directory / "return-histograms"
    )
    strategy_backtests = {
        "PCA factor / Historical Means Sharpe": result.strategy_backtest,
        "PCA + U.S. commodity factors / Historical Means Sharpe": result.commodity_factor_backtest,
        "U.S. commodity factors only / Historical Means Sharpe": result.commodity_only_backtest,
        "Markowitz / Historical Means Sharpe": result.markowitz_backtest,
    }
    if any(backtest is None for backtest in strategy_backtests.values()):
        raise ValueError("result has incomplete historical-mean strategy backtests")
    summary_outputs = ExperimentOutputs(
        daily_returns=result.daily_returns,
        cumulative_returns=result.cumulative_returns,
        performance_metrics=result.performance_metrics,
        implementation_metrics=result.implementation_metrics,
        strategy_backtests={
            label: backtest
            for label, backtest in strategy_backtests.items()
            if backtest is not None
        },
    )
    paths.update(save_standard_summary_products(summary_outputs, output_directory))
    if result.statistical_tests is not None:
        paths["statistical_tests"] = output_directory / "statistical-tests.csv"
        result.statistical_tests.to_csv(paths["statistical_tests"])
    return {
        **paths,
        **{f"histogram:{label}": path for label, path in histogram_paths.items()},
    }


if __name__ == "__main__":
    experiment = run_xle_pca_historical_mean_experiment(num_principal_components=1)
    visual_paths = save_xle_experiment_visuals(experiment, "research_outputs")
    print(experiment.cumulative_returns.tail())
    print("Performance metrics:")
    print(experiment.performance_metrics)
    print("Implementation metrics:")
    print(experiment.implementation_metrics)
    print("Visual outputs:")
    print(*visual_paths.values(), sep="\n")
