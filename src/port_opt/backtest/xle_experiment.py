"""Compare an XLE constituent portfolio with equal weight and the XLE ETF.

This is the preserved, runnable energy-portfolio research experiment. It uses the
package backtester rather than duplicating its walk-forward logic, while keeping
the original research question and constituent universe visible and editable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from port_opt.strategy import (
    Commodity_Factor_Strategy,
    Markowitz_Portfolio,
    PCA_Commodity_Factor_Strategy,
    PCA_Historical_Mean_Strategy,
    PCA_LightGBM_Strategy,
    get_returns,
)

from .backtest import BacktestResult, run_walk_forward_backtest
from .metrics import summarize_implementation, summarize_performance
from .statistics import run_pre_specified_return_comparisons
from .visualizations import (
    save_covariance_comparison,
    save_implementation_comparison,
    save_return_histograms,
    save_risk_return_comparison,
)

# Original XLE constituent research universe. Review constituent changes before
# interpreting long-horizon results: this static list can introduce survivorship bias.
XLE_TICKERS = [
    "XOM",
    "CVX",
    "COP",
    "PSX",
    "MPC",
    "VLO",
    "SLB",
    "EOG",
    "WMB",
    "BKR",
    "KMI",
    "OXY",
    "HAL",
    "FANG",
    "DVN",
    "TRGP",
    "OKE",
    "APA",
    "NOV",
]
BASELINE_TICKER = "XLE"
DEFAULT_LOOKBACK_PERIODS = 504
DEFAULT_REBALANCE_FREQUENCY = 21
DEFAULT_COMMODITY_FACTORS = {
    "CL=F": "WTI crude oil",
    "NG=F": "Henry Hub natural gas",
    "RB=F": "RBOB gasoline",
}


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


def load_xle_returns(
    start_date: str,
    end_date: str,
    *,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch and align constituent and benchmark daily simple returns once.

    Rows with a missing value for any asset or benchmark are removed deliberately;
    callers should record this complete-case treatment in research outputs.
    """
    if max_download_attempts < 1:
        raise ValueError("max_download_attempts must be positive")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")

    last_error: Exception | None = None
    for attempt in range(max_download_attempts):
        try:
            downloaded_returns = get_returns(
                [*XLE_TICKERS, BASELINE_TICKER],
                start_date=start_date,
                end_date=end_date,
            )
            if not downloaded_returns.empty:
                break
            last_error = ValueError("provider returned no observations")
        except Exception as error:
            last_error = error
        if attempt + 1 < max_download_attempts:
            time.sleep(retry_delay_seconds * (2**attempt))
    else:
        raise RuntimeError(
            f"unable to download non-empty returns after {max_download_attempts} attempts"
        ) from last_error
    aligned_returns = downloaded_returns.dropna(how="any")
    asset_returns = aligned_returns.loc[:, XLE_TICKERS]
    baseline_returns = aligned_returns[BASELINE_TICKER]
    if asset_returns.empty:
        raise ValueError(
            "no complete XLE constituent return observations were downloaded"
        )
    return asset_returns, baseline_returns


def load_commodity_factor_returns(
    start_date: str,
    end_date: str,
    *,
    commodity_factors: Mapping[str, str] = DEFAULT_COMMODITY_FACTORS,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch complete-case U.S.-traded commodity factor returns.

    Mapping keys are provider tickers and mapping values are stable research
    factor labels. The function deliberately returns daily simple returns, not
    price levels, because the covariance model uses factor returns.
    """
    if not commodity_factors:
        raise ValueError("commodity_factors must not be empty")
    if len(set(commodity_factors)) != len(commodity_factors) or len(
        set(commodity_factors.values())
    ) != len(commodity_factors):
        raise ValueError("commodity factor tickers and labels must be unique")
    if max_download_attempts < 1:
        raise ValueError("max_download_attempts must be positive")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")

    last_error: Exception | None = None
    for attempt in range(max_download_attempts):
        try:
            downloaded_returns = get_returns(
                list(commodity_factors), start_date=start_date, end_date=end_date
            )
            if not downloaded_returns.empty:
                break
            last_error = ValueError("provider returned no commodity observations")
        except Exception as error:
            last_error = error
        if attempt + 1 < max_download_attempts:
            time.sleep(retry_delay_seconds * (2**attempt))
    else:
        raise RuntimeError(
            "unable to download non-empty commodity returns after "
            f"{max_download_attempts} attempts"
        ) from last_error

    factor_returns = downloaded_returns.loc[:, list(commodity_factors)].dropna(
        how="any"
    )
    if factor_returns.empty:
        raise ValueError(
            "no complete commodity factor return observations were downloaded"
        )
    return factor_returns.rename(columns=commodity_factors)


def select_backtest_panel(
    asset_returns: pd.DataFrame,
    backtest_start: str,
    lookback_periods: int,
) -> tuple[pd.DataFrame, int]:
    """Select prior observations and return the effective lookback length.

    The requested rolling lookback is a research choice and is never silently
    shortened when data is unavailable.
    """
    requested_start = pd.Timestamp(backtest_start)
    start_position = asset_returns.index.searchsorted(requested_start)
    if start_position >= len(asset_returns):
        raise ValueError("backtest_start falls after the available return observations")
    available_history = start_position
    if available_history < 1:
        raise ValueError(
            "at least one return observation is required before backtest_start"
        )
    if lookback_periods < 1:
        raise ValueError("lookback_periods must be positive")
    if lookback_periods > available_history:
        raise ValueError(
            "lookback_periods exceeds the observations available before backtest_start"
        )
    return (
        asset_returns.iloc[start_position - lookback_periods :],
        lookback_periods,
    )


def run_xle_pca_lightgbm_experiment(
    *,
    training_start: str = "2021-01-01",
    backtest_start: str = "2024-01-01",
    end_date: str = "2026-01-01",
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04 / 252,
    lightgbm_window_size: int = 30,
    lightgbm_num_boost_round: int = 100,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> XLEExperimentResult:
    """Run the forward-looking rolling-window energy strategy and baselines."""
    asset_returns, baseline_returns = load_xle_returns(
        training_start,
        end_date,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    # Pass the selected prior history to the generic engine so its first OOS
    # observation is backtest_start (or the next available trading observation).
    backtest_panel, effective_lookback = select_backtest_panel(
        asset_returns, backtest_start, lookback_periods
    )
    strategy = PCA_LightGBM_Strategy(
        risk_free_rate=risk_free_rate,
        window_size=lightgbm_window_size,
        num_boost_round=lightgbm_num_boost_round,
    )
    strategy_backtest = run_walk_forward_backtest(
        backtest_panel,
        strategy.weights_from_returns,
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
    )

    evaluation_index = strategy_backtest.portfolio_returns.index
    daily_returns = pd.DataFrame(
        {
            "PCA factor + LightGBM": strategy_backtest.portfolio_returns,
            "Equal-weight XLE constituents": asset_returns.loc[evaluation_index].mean(
                axis=1
            ),
            "XLE baseline": baseline_returns.loc[evaluation_index],
        }
    )
    return XLEExperimentResult(
        daily_returns=daily_returns,
        cumulative_returns=(1.0 + daily_returns).cumprod(),
        performance_metrics=summarize_performance(
            daily_returns, risk_free_rate=risk_free_rate
        ),
        implementation_metrics=summarize_implementation(
            {"PCA factor + LightGBM": strategy_backtest}
        ),
        strategy_backtest=strategy_backtest,
    )


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
    strategy_backtest = run_walk_forward_backtest(
        backtest_panel,
        strategy.weights_from_returns,
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
    )
    markowitz_backtest = run_walk_forward_backtest(
        backtest_panel,
        markowitz_strategy.weights_from_returns,
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
    )
    commodity_factor_backtest = run_walk_forward_backtest(
        backtest_panel,
        commodity_strategy.weights_from_equity_and_factor_returns,
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
        factor_returns=backtest_commodity_panel,
    )
    commodity_only_backtest = run_walk_forward_backtest(
        backtest_panel,
        commodity_only_strategy.weights_from_equity_and_factor_returns,
        lookback_periods=effective_lookback,
        rebalance_frequency=rebalance_frequency,
        factor_returns=backtest_commodity_panel,
    )

    evaluation_index = strategy_backtest.portfolio_returns.index
    if not (
        evaluation_index.equals(markowitz_backtest.portfolio_returns.index)
        and evaluation_index.equals(commodity_factor_backtest.portfolio_returns.index)
        and evaluation_index.equals(commodity_only_backtest.portfolio_returns.index)
    ):
        raise RuntimeError("covariance strategies produced different backtest dates")
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
    daily_returns = pd.DataFrame(
        {
            "PCA factor / Historical Means Sharpe": strategy_backtest.portfolio_returns,
            "PCA + U.S. commodity factors / Historical Means Sharpe": (
                commodity_factor_backtest.portfolio_returns
            ),
            "U.S. commodity factors only / Historical Means Sharpe": (
                commodity_only_backtest.portfolio_returns
            ),
            "Markowitz / Historical Means Sharpe": markowitz_backtest.portfolio_returns,
            "Equal-weight XLE constituents": asset_returns.loc[evaluation_index].mean(
                axis=1
            ),
            "XLE baseline": baseline_returns.loc[evaluation_index],
        }
    )
    strategy_backtests = {
        "PCA factor / Historical Means Sharpe": strategy_backtest,
        "PCA + U.S. commodity factors / Historical Means Sharpe": (
            commodity_factor_backtest
        ),
        "U.S. commodity factors only / Historical Means Sharpe": (
            commodity_only_backtest
        ),
        "Markowitz / Historical Means Sharpe": markowitz_backtest,
    }
    statistical_tests = run_pre_specified_return_comparisons(
        daily_returns,
        {
            "PCA + commodity factors versus PCA": (
                "PCA + U.S. commodity factors / Historical Means Sharpe",
                "PCA factor / Historical Means Sharpe",
            )
        },
        hac_lag=hac_lag,
    )
    return XLEExperimentResult(
        daily_returns=daily_returns,
        cumulative_returns=(1.0 + daily_returns).cumprod(),
        performance_metrics=summarize_performance(
            daily_returns, risk_free_rate=risk_free_rate
        ),
        implementation_metrics=summarize_implementation(strategy_backtests),
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
        "risk_return_comparison": output_directory / "risk-return-comparison.png",
        "implementation_comparison": output_directory / "implementation-comparison.png",
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
    save_risk_return_comparison(
        result.performance_metrics, paths["risk_return_comparison"]
    )
    save_implementation_comparison(
        result.implementation_metrics, paths["implementation_comparison"]
    )
    paths["performance_metrics"] = output_directory / "performance-metrics.csv"
    paths["implementation_metrics"] = output_directory / "implementation-metrics.csv"
    result.performance_metrics.to_csv(paths["performance_metrics"])
    result.implementation_metrics.to_csv(paths["implementation_metrics"])
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
