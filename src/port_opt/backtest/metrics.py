"""Comparable performance and implementation metrics for backtest outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_TAIL_PROBABILITY = 0.05


def _validate_daily_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("daily_returns must be a non-empty DataFrame")
    if (
        daily_returns.isna().any().any()
        or not np.isfinite(daily_returns.to_numpy()).all()
    ):
        raise ValueError("daily_returns must contain only finite values")
    return daily_returns.astype(float)


def summarize_performance(
    daily_returns: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    tail_probability: float = DEFAULT_TAIL_PROBABILITY,
) -> pd.DataFrame:
    """Summarize common-return, downside-risk, and drawdown metrics.

    All columns are calculated over the supplied daily simple-return observations.
    The tail columns are empirical: the five-percent return quantile and the
    average return at or below that quantile. They do not assume normal returns.
    """
    daily_returns = _validate_daily_returns(daily_returns)
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be between zero and one")

    tail_label = f"{tail_probability * 100:g}pct"
    summaries: dict[str, dict[str, float | int]] = {}
    for name, returns in daily_returns.items():
        observations = len(returns)
        wealth = (1.0 + returns).cumprod()
        cumulative_return = float(wealth.iloc[-1] - 1.0)
        annualized_geometric_return = float(
            wealth.iloc[-1] ** (periods_per_year / observations) - 1.0
        )
        daily_volatility = float(returns.std(ddof=1)) if observations > 1 else np.nan
        annualized_volatility = float(daily_volatility * np.sqrt(periods_per_year))
        excess_returns = returns - risk_free_rate
        sharpe_ratio = (
            float(np.sqrt(periods_per_year) * excess_returns.mean() / daily_volatility)
            if daily_volatility > 0
            else np.nan
        )
        downside_returns = excess_returns.clip(upper=0.0)
        annualized_downside_deviation = float(
            np.sqrt((downside_returns**2).mean()) * np.sqrt(periods_per_year)
        )
        sortino_ratio = (
            float(
                excess_returns.mean() * periods_per_year / annualized_downside_deviation
            )
            if annualized_downside_deviation > 0
            else np.nan
        )
        drawdowns = wealth / wealth.cummax() - 1.0
        tail_return_quantile = float(returns.quantile(tail_probability))
        tail_returns = returns[returns <= tail_return_quantile]
        summaries[name] = {
            "observations": observations,
            "cumulative_return": cumulative_return,
            "mean_daily_return": float(returns.mean()),
            "annualized_arithmetic_return": float(returns.mean() * periods_per_year),
            "annualized_geometric_return": annualized_geometric_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "maximum_drawdown": float(drawdowns.min()),
            "worst_daily_return": float(returns.min()),
            "positive_day_fraction": float((returns > 0.0).mean()),
            f"tail_return_quantile_{tail_label}": tail_return_quantile,
            f"tail_expected_shortfall_{tail_label}": float(tail_returns.mean()),
        }
    return pd.DataFrame.from_dict(summaries, orient="index").rename_axis("series")


def summarize_implementation(
    strategy_backtests: dict[str, BacktestResult],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Summarize rebalance turnover and concentration for modeled strategies.

    Initial allocation is excluded from turnover. Turnover is target-weight
    turnover; it does not include drift between rebalances or trading costs.
    """
    if not strategy_backtests:
        raise ValueError("strategy_backtests must not be empty")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")

    summaries: dict[str, dict[str, float | int]] = {}
    for name, backtest in strategy_backtests.items():
        if not isinstance(backtest, BacktestResult):
            raise TypeError(
                "strategy_backtests values must be BacktestResult instances"
            )
        subsequent_turnover = backtest.turnover.iloc[1:]
        weights = backtest.weights
        effective_asset_count = 1.0 / (weights**2).sum(axis=1)
        summaries[name] = {
            "out_of_sample_observations": len(backtest.portfolio_returns),
            "rebalances": len(backtest.records),
            "total_turnover_ex_initial": float(subsequent_turnover.sum()),
            "mean_rebalance_turnover_ex_initial": float(subsequent_turnover.mean()),
            "annualized_turnover": float(
                subsequent_turnover.sum()
                / len(backtest.portfolio_returns)
                * periods_per_year
            ),
            "mean_max_weight": float(weights.max(axis=1).mean()),
            "maximum_weight": float(weights.max(axis=1).max()),
            "mean_effective_number_assets": float(effective_asset_count.mean()),
        }
    return pd.DataFrame.from_dict(summaries, orient="index").rename_axis("strategy")
