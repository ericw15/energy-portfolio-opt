"""Paired daily-return inference for pre-specified strategy comparisons."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


def paired_hac_return_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    hac_lag: int = 20,
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Test whether two aligned daily return series have equal mean return.

    The paired active return is ``d_t = strategy_return_t - benchmark_return_t``.
    The standard error of its mean uses a Bartlett-kernel Newey--West/HAC
    long-run variance estimate with the supplied maximum lag. The reported
    p-value is two-sided for the null hypothesis ``E[d_t] = 0``.
    """
    if not isinstance(strategy_returns, pd.Series) or not isinstance(
        benchmark_returns, pd.Series
    ):
        raise TypeError("strategy_returns and benchmark_returns must be Series")
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError("return series must have exactly the same index")
    if len(strategy_returns) < 2:
        raise ValueError("at least two paired return observations are required")
    if hac_lag < 0 or hac_lag >= len(strategy_returns):
        raise ValueError("hac_lag must be between zero and observations minus one")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    active_returns = (strategy_returns - benchmark_returns).astype(float)
    if not np.isfinite(active_returns.to_numpy()).all():
        raise ValueError("return series must contain only finite values")
    centered = active_returns.to_numpy() - active_returns.mean()
    observations = len(centered)
    long_run_variance = float(centered @ centered / observations)
    for lag in range(1, hac_lag + 1):
        autocovariance = float(centered[lag:] @ centered[:-lag] / observations)
        long_run_variance += 2.0 * (1.0 - lag / (hac_lag + 1.0)) * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / observations))
    mean_active_return = float(active_returns.mean())
    if standard_error == 0.0:
        test_statistic = (
            np.nan
            if mean_active_return == 0.0
            else np.inf * np.sign(mean_active_return)
        )
        p_value = 1.0 if mean_active_return == 0.0 else 0.0
    else:
        test_statistic = mean_active_return / standard_error
        p_value = float(2.0 * norm.sf(abs(test_statistic)))
    critical_value = float(norm.ppf((1.0 + confidence_level) / 2.0))
    return pd.Series(
        {
            "observations": observations,
            "mean_daily_active_return": mean_active_return,
            "annualized_active_return": mean_active_return * periods_per_year,
            "hac_lag": hac_lag,
            "hac_standard_error": standard_error,
            "test_statistic": test_statistic,
            "confidence_level": confidence_level,
            "confidence_interval_lower": mean_active_return
            - critical_value * standard_error,
            "confidence_interval_upper": mean_active_return
            + critical_value * standard_error,
            "two_sided_p_value": p_value,
        }
    )


def run_pre_specified_return_comparisons(
    daily_returns: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    hac_lag: int = 20,
) -> pd.DataFrame:
    """Run named strategy-versus-benchmark paired HAC tests.

    Each comparison maps a presentation label to ``(strategy_column,
    benchmark_column)``. The output is indexed by the supplied labels, making
    the analysis set explicit rather than implicitly testing all columns.
    """
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("daily_returns must be a non-empty DataFrame")
    if not comparisons:
        raise ValueError("comparisons must not be empty")
    rows: dict[str, pd.Series] = {}
    for label, (strategy_name, benchmark_name) in comparisons.items():
        if strategy_name not in daily_returns or benchmark_name not in daily_returns:
            raise ValueError("comparison names must be columns in daily_returns")
        result = paired_hac_return_test(
            daily_returns[strategy_name], daily_returns[benchmark_name], hac_lag=hac_lag
        )
        result.loc["strategy"] = strategy_name
        result.loc["benchmark"] = benchmark_name
        rows[label] = result
    columns = ["strategy", "benchmark"]
    results = pd.DataFrame.from_dict(rows, orient="index")
    return results.loc[
        :, [*columns, *[c for c in results.columns if c not in columns]]
    ].rename_axis("comparison")
