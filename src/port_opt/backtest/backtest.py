"""Walk-forward portfolio backtesting primitives.

The module deliberately accepts a complete returns panel rather than fetching data.
This keeps data provenance separate from simulation, makes runs reproducible, and
allows the same engine to test any return or covariance estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class WeightEstimator(Protocol):
    """Produces target weights using only the supplied in-sample returns."""

    def __call__(self, in_sample_returns: pd.DataFrame) -> pd.Series: ...


class FactorWeightEstimator(Protocol):
    """Produces target weights from in-sample asset and observed factor returns."""

    def __call__(
        self,
        in_sample_returns: pd.DataFrame,
        in_sample_factor_returns: pd.DataFrame,
    ) -> pd.Series: ...


@dataclass(frozen=True)
class RebalanceRecord:
    """Audit information for one walk-forward rebalance."""

    rebalance_date: pd.Timestamp
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_of_sample_start: pd.Timestamp
    out_of_sample_end: pd.Timestamp
    weights: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Outputs of a backtest, expressed as daily simple returns."""

    portfolio_returns: pd.Series
    weights: pd.DataFrame
    records: tuple[RebalanceRecord, ...]
    turnover: pd.Series

    @property
    def wealth_index(self) -> pd.Series:
        """Growth of one unit of capital, before costs."""
        return (1.0 + self.portfolio_returns).cumprod()

    def sharpe_ratio(
        self, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """Annualized Sharpe ratio from daily simple returns.

        ``risk_free_rate`` must use the same (daily) period as the returns.
        """
        excess_returns = self.portfolio_returns - risk_free_rate
        volatility = excess_returns.std(ddof=1)
        if len(excess_returns) < 2 or np.isclose(volatility, 0.0):
            return np.nan
        return float(np.sqrt(periods_per_year) * excess_returns.mean() / volatility)


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("returns index must be unique and sorted ascending")
    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "returns must contain only finite values; align/drop missing data first"
        )
    return returns.astype(float)


def _validate_weights(
    weights: pd.Series, assets: pd.Index, long_only: bool
) -> pd.Series:
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=assets, dtype=float)
    if not weights.index.is_unique:
        raise ValueError("weight estimator returned duplicate asset labels")
    if set(weights.index) != set(assets):
        raise ValueError("weight estimator must return exactly the return-panel assets")
    weights = weights.reindex(assets).astype(float)
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("weight estimator returned non-finite weights")
    if long_only and (weights < -1e-10).any():
        raise ValueError("long_only backtests do not allow negative weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must sum to one")
    return weights


def run_walk_forward_backtest(
    returns: pd.DataFrame,
    weight_estimator: WeightEstimator | FactorWeightEstimator,
    *,
    lookback_periods: int,
    rebalance_frequency: int = 21,
    long_only: bool = True,
    factor_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a no-look-ahead, periodic-rebalance backtest.

    At each rebalance, the estimator receives the trailing ``lookback_periods``
    observations that end *before* the first out-of-sample date. Its weights are
    held for the next ``rebalance_frequency`` available return observations.
    Frequencies are counts of rows (normally trading days), never calendar days.
    The first out-of-sample return is therefore the first row following training.
    When ``factor_returns`` is supplied, it must share the asset panel's exact
    index and the estimator receives the aligned in-sample factor panel as its
    second argument. Factors remain observable inputs; they are not holdings.
    """
    returns = _validate_returns(returns)
    if factor_returns is not None:
        factor_returns = _validate_returns(factor_returns)
        if not factor_returns.index.equals(returns.index):
            raise ValueError("factor_returns must have exactly the asset return index")
    if lookback_periods < 1:
        raise ValueError("lookback_periods must be positive")
    if rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be positive")
    if len(returns) <= lookback_periods:
        raise ValueError("returns must include at least one out-of-sample observation")

    portfolio_parts: list[pd.Series] = []
    weight_parts: list[pd.DataFrame] = []
    turnover_parts: list[pd.Series] = []
    records: list[RebalanceRecord] = []
    previous_weights: pd.Series | None = None

    for oos_start in range(lookback_periods, len(returns), rebalance_frequency):
        oos_end = min(oos_start + rebalance_frequency, len(returns))
        in_sample = returns.iloc[oos_start - lookback_periods : oos_start]
        holding_returns = returns.iloc[oos_start:oos_end]
        if factor_returns is None:
            estimated_weights = weight_estimator(in_sample.copy())
        else:
            in_sample_factors = factor_returns.iloc[
                oos_start - lookback_periods : oos_start
            ]
            estimated_weights = weight_estimator(
                in_sample.copy(), in_sample_factors.copy()
            )
        weights = _validate_weights(estimated_weights, returns.columns, long_only)

        portfolio_parts.append(holding_returns @ weights)
        weight_parts.append(
            pd.DataFrame(
                np.tile(weights.to_numpy(), (len(holding_returns), 1)),
                index=holding_returns.index,
                columns=returns.columns,
            )
        )
        turnover = (
            0.0
            if previous_weights is None
            else float((weights - previous_weights).abs().sum() / 2)
        )
        turnover_parts.append(pd.Series(turnover, index=[holding_returns.index[0]]))
        records.append(
            RebalanceRecord(
                rebalance_date=holding_returns.index[0],
                in_sample_start=in_sample.index[0],
                in_sample_end=in_sample.index[-1],
                out_of_sample_start=holding_returns.index[0],
                out_of_sample_end=holding_returns.index[-1],
                weights=weights.copy(),
            )
        )
        previous_weights = weights

    return BacktestResult(
        portfolio_returns=pd.concat(portfolio_parts).rename("portfolio_return"),
        weights=pd.concat(weight_parts),
        records=tuple(records),
        turnover=pd.concat(turnover_parts).rename("turnover"),
    )
