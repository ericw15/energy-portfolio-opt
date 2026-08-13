"""Leakage-free development study for selecting covariance estimators.

This module is intentionally independent of the XLE portfolio experiment.  It
scores covariance forecasts directly, so covariance selection is not conflated
with the historical-mean estimate or a particular portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from port_opt.strategy import PCA_Commodity_Factor_Strategy, PCA_factor_Strategy
from port_opt.strategy.covariance import ewma_covariance


@dataclass(frozen=True)
class CovarianceStudyResult:
    """Aggregate metrics, daily technical scores, and holding-period forecasts."""

    metrics: pd.DataFrame
    forecasts: pd.DataFrame
    holding_period_forecasts: pd.DataFrame
    development_end: pd.Timestamp


def _display_estimator_name(name: str) -> str:
    return name.replace("_covariance", "").replace("_", " ").title()


def _estimator_display_order(name: str) -> tuple[int, int | str]:
    """Keep estimator families adjacent rather than ranking bars by one metric."""
    fixed_order = {
        "sample_covariance": 0,
        "pca_covariance": 3,
        "pca_plus_factor_covariance": 4,
    }
    if name.startswith("ewma_covariance_half_life_"):
        return 2, int(name.rsplit("_", maxsplit=1)[-1])
    return fixed_order.get(name, 5), name


def _estimator_colors(estimators: pd.Index) -> dict[str, tuple[float, ...]]:
    """Use related shades for EWMA candidates and soft distinct family colors."""
    family_colors = plt.get_cmap("Set2")
    ewma_colors = plt.get_cmap("Blues")
    ordered_estimators = sorted(estimators)
    ewma_estimators = [
        estimator
        for estimator in ordered_estimators
        if estimator.startswith("ewma_covariance_half_life_")
    ]
    colors = {
        estimator: family_colors(position % family_colors.N)
        for position, estimator in enumerate(
            estimator
            for estimator in ordered_estimators
            if estimator not in ewma_estimators
        )
    }
    colors.update(
        {
            estimator: ewma_colors(level)
            for estimator, level in zip(
                ewma_estimators,
                np.linspace(0.45, 0.85, len(ewma_estimators)),
                strict=True,
            )
        }
    )
    return colors


def save_fixed_portfolio_variance_comparison(
    result: CovarianceStudyResult, output_path: str | Path
) -> None:
    """Save RMSE and calibration bars for the fixed equal-weight comparison.

    The left panel is lower-is-better variance-forecast error. The right panel
    compares average realized with average predicted variance: one is ideal,
    above one indicates systematic underprediction of variance, and below one
    indicates systematic overprediction.
    """
    required_columns = {
        "root_mean_squared_equal_weight_variance_error",
        "equal_weight_variance_calibration_ratio",
    }
    if not required_columns.issubset(result.metrics.columns):
        raise ValueError(
            "study metrics do not contain fixed-portfolio variance results"
        )
    metrics = result.metrics.loc[
        sorted(result.metrics.index, key=_estimator_display_order)
    ]
    if metrics.empty or not np.isfinite(metrics.to_numpy(dtype=float)).all():
        raise ValueError("study metrics must contain finite values")

    labels = [_display_estimator_name(name) for name in metrics.index]
    positions = np.arange(len(metrics))
    estimator_colors = _estimator_colors(metrics.index)
    colors = [estimator_colors[estimator] for estimator in metrics.index]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    rmse = metrics["root_mean_squared_equal_weight_variance_error"]
    calibration = metrics["equal_weight_variance_calibration_ratio"]

    axes[0].bar(positions, rmse, color=colors)
    axes[0].set_title("Fixed equal-weight variance forecast error")
    axes[0].set_ylabel("Root mean squared variance error (lower is better)")
    axes[1].bar(positions, calibration, color=colors)
    axes[1].axhline(
        1.0, color="black", linewidth=1, linestyle="--", label="Perfect calibration"
    )
    axes[1].set_title("Fixed equal-weight variance calibration")
    axes[1].set_ylabel("Realized variance / predicted variance (target = 1)")
    axes[1].legend()
    for axis, values in zip(axes, (rmse, calibration), strict=True):
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        for position, value in enumerate(values):
            axis.annotate(
                f"{value:.3g}",
                (position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _validate_returns(returns: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError(f"{name} index must be unique and sorted ascending")
    if returns.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return returns.astype(float)


def _positive_definite_for_scoring(
    covariance: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    matrix = covariance.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.trace(matrix) / len(matrix)), 1e-12)
    floor = scale * 1e-10
    adjustment = max(0.0, floor - float(eigenvalues.min()))
    if adjustment:
        matrix = matrix + np.eye(len(matrix)) * adjustment
    return matrix, adjustment


def run_covariance_estimator_study(
    development_returns: pd.DataFrame,
    *,
    factor_returns: pd.DataFrame | None = None,
    lookback_periods: int = 504,
    rebalance_frequency: int = 21,
    ewma_half_lives: Sequence[int] = (63, 126, 252),
    num_principal_components: int | None = None,
) -> CovarianceStudyResult:
    """Compare rolling covariance forecasts on development-only returns.

    At each rebalance, every estimator sees the same trailing return window and
    forecasts covariance for the following ``rebalance_frequency`` available
    observations. The presentation-first comparison is an equal-weight fixed
    portfolio: predicted daily variance is ``w' H w`` and realized daily variance
    is the sample variance of its returns in the following holding block. Lower
    absolute and root mean squared variance errors are better; a calibration
    ratio near one means average predicted and realized variance agree.

    The secondary ``mean_quasi_log_likelihood`` is
    ``logdet(H) + e' H^-1 e`` per future daily return, where ``e`` is the future
    return less the in-window arithmetic mean and ``H`` is the forecast matrix.
    This is a Gaussian quasi-likelihood score, used solely as a proper ranking
    rule for second-moment forecasts; it does not claim returns are Gaussian.

    ``factor_returns`` is optional. When supplied, it must share the exact index
    of the development return panel and adds the PCA-plus-factor covariance
    candidate. It is intended for the separately supplied commodity-factor data.
    """
    returns = _validate_returns(development_returns, "development_returns")
    if lookback_periods < 2:
        raise ValueError("lookback_periods must be at least two")
    if rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be positive")
    if len(returns) <= lookback_periods:
        raise ValueError("development_returns must include out-of-sample observations")
    if not ewma_half_lives or any(half_life < 1 for half_life in ewma_half_lives):
        raise ValueError("ewma_half_lives must contain positive integers")
    if len(set(ewma_half_lives)) != len(ewma_half_lives):
        raise ValueError("ewma_half_lives must not contain duplicates")

    factors: pd.DataFrame | None = None
    if factor_returns is not None:
        factors = _validate_returns(factor_returns, "factor_returns")
        if not factors.index.equals(returns.index):
            raise ValueError(
                "factor_returns must have exactly the development return index"
            )

    pca_strategy = PCA_factor_Strategy(
        risk_free_rate=0.0, num_principal_components=num_principal_components
    )
    pca_factor_strategy = (
        PCA_Commodity_Factor_Strategy(
            risk_free_rate=0.0, num_principal_components=num_principal_components
        )
        if factors is not None
        else None
    )
    equal_weights = np.full(len(returns.columns), 1.0 / len(returns.columns))
    forecast_rows: list[dict[str, object]] = []
    holding_period_rows: list[dict[str, object]] = []

    for forecast_start in range(lookback_periods, len(returns), rebalance_frequency):
        forecast_end = min(forecast_start + rebalance_frequency, len(returns))
        training_returns = returns.iloc[
            forecast_start - lookback_periods : forecast_start
        ]
        training_mean = training_returns.mean().to_numpy()
        covariance_estimates: dict[str, pd.DataFrame] = {
            "sample_covariance": training_returns.cov(),
            "pca_covariance": pca_strategy.get_covariance_matrix(
                None, None, training_returns
            ),
        }
        covariance_estimates.update(
            {
                f"ewma_covariance_half_life_{half_life}": ewma_covariance(
                    training_returns, half_life
                )
                for half_life in ewma_half_lives
            }
        )
        if factors is not None and pca_factor_strategy is not None:
            covariance_estimates["pca_plus_factor_covariance"] = (
                pca_factor_strategy.get_covariance_matrix_with_factors(
                    training_returns,
                    factors.iloc[forecast_start - lookback_periods : forecast_start],
                )
            )

        holding_returns = (
            returns.iloc[forecast_start:forecast_end].to_numpy() @ equal_weights
        )
        realized_equal_weight_variance = (
            float(holding_returns.var(ddof=1)) if len(holding_returns) >= 2 else None
        )

        for estimator, covariance in covariance_estimates.items():
            matrix, regularization = _positive_definite_for_scoring(covariance)
            log_determinant = float(np.linalg.slogdet(matrix)[1])
            condition_number = float(np.linalg.cond(matrix))
            predicted_equal_weight_variance = float(
                equal_weights @ matrix @ equal_weights
            )
            if realized_equal_weight_variance is not None:
                holding_period_rows.append(
                    {
                        "decision_date": training_returns.index[-1],
                        "holding_period_start": returns.index[forecast_start],
                        "holding_period_end": returns.index[forecast_end - 1],
                        "estimator": estimator,
                        "predicted_equal_weight_variance": predicted_equal_weight_variance,
                        "realized_equal_weight_variance": realized_equal_weight_variance,
                        "absolute_equal_weight_variance_error": abs(
                            predicted_equal_weight_variance
                            - realized_equal_weight_variance
                        ),
                        "squared_equal_weight_variance_error": (
                            predicted_equal_weight_variance
                            - realized_equal_weight_variance
                        )
                        ** 2,
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
                        "condition_number": condition_number,
                        "scoring_regularization": regularization,
                    }
                )
            for date, observed_return in returns.iloc[
                forecast_start:forecast_end
            ].iterrows():
                innovation = observed_return.to_numpy() - training_mean
                quasi_log_likelihood = float(
                    log_determinant + innovation @ np.linalg.solve(matrix, innovation)
                )
                equal_weight_innovation = float(equal_weights @ innovation)
                forecast_rows.append(
                    {
                        "decision_date": training_returns.index[-1],
                        "forecast_date": date,
                        "estimator": estimator,
                        "quasi_log_likelihood": quasi_log_likelihood,
                        "predicted_equal_weight_variance": predicted_equal_weight_variance,
                        "realized_equal_weight_squared_innovation": (
                            equal_weight_innovation**2
                        ),
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
                        "condition_number": condition_number,
                        "scoring_regularization": regularization,
                    }
                )

    forecasts = pd.DataFrame(forecast_rows)
    holding_period_forecasts = pd.DataFrame(holding_period_rows)
    holding_period_metrics = holding_period_forecasts.groupby(
        "estimator", sort=True
    ).agg(
        holding_periods=("holding_period_end", "size"),
        mean_predicted_equal_weight_variance=(
            "predicted_equal_weight_variance",
            "mean",
        ),
        mean_realized_equal_weight_variance=(
            "realized_equal_weight_variance",
            "mean",
        ),
        mean_absolute_equal_weight_variance_error=(
            "absolute_equal_weight_variance_error",
            "mean",
        ),
        root_mean_squared_equal_weight_variance_error=(
            "squared_equal_weight_variance_error",
            lambda values: float(np.sqrt(values.mean())),
        ),
        median_condition_number=("condition_number", "median"),
        maximum_condition_number=("condition_number", "max"),
        minimum_eigenvalue=("minimum_eigenvalue", "min"),
        maximum_scoring_regularization=("scoring_regularization", "max"),
    )
    metrics = (
        forecasts.groupby("estimator", sort=True)
        .agg(
            forecasts=("forecast_date", "size"),
            mean_quasi_log_likelihood=("quasi_log_likelihood", "mean"),
        )
        .join(holding_period_metrics)
        .assign(
            equal_weight_variance_calibration_ratio=lambda frame: (
                frame["mean_realized_equal_weight_variance"]
                / frame["mean_predicted_equal_weight_variance"]
            )
        )
        .sort_values("root_mean_squared_equal_weight_variance_error")
    )
    return CovarianceStudyResult(
        metrics=metrics,
        forecasts=forecasts,
        holding_period_forecasts=holding_period_forecasts,
        development_end=returns.index[-1],
    )


if __name__ == "__main__":
    from port_opt.backtest.xle_experiment import XLE_TICKERS
    from port_opt.strategy import get_returns

    returns = get_returns(XLE_TICKERS, start_date="2018-01-01", end_date="2021-01-01")
    results = run_covariance_estimator_study(
        development_returns=returns, num_principal_components=1
    )
    print(results.metrics)
    save_fixed_portfolio_variance_comparison(
        results, "covariance_study_outputs/fixed-portfolio-variance-comparison.png"
    )
