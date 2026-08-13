"""Reproducible visual diagnostics for walk-forward portfolio research."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _validate_covariance_pair(
    pca_covariance: pd.DataFrame, markowitz_covariance: pd.DataFrame
) -> None:
    for covariance in (pca_covariance, markowitz_covariance):
        if not isinstance(covariance, pd.DataFrame) or covariance.empty:
            raise ValueError("covariance matrices must be non-empty DataFrames")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance matrices must be square")
        if covariance.index.tolist() != covariance.columns.tolist():
            raise ValueError("covariance matrix row and column labels must match")
        if not np.isfinite(covariance.to_numpy(dtype=float)).all():
            raise ValueError("covariance matrices must contain only finite values")
    if not pca_covariance.index.equals(markowitz_covariance.index):
        raise ValueError("covariance matrices must use the same asset labels and order")


def save_covariance_comparison(
    pca_covariance: pd.DataFrame,
    comparison_covariance: pd.DataFrame,
    output_path: str | Path,
    *,
    as_of_date: pd.Timestamp | str | None = None,
    pca_label: str = "PCA-factor covariance",
    comparison_label: str = "Markowitz sample covariance",
) -> None:
    """Save side-by-side covariance heatmaps on one comparable color scale.

    Both heatmaps share one common color scale, making their displayed cell
    magnitudes directly comparable. ``as_of_date`` is a display label; callers
    determine the return history used to estimate each matrix.
    """
    _validate_covariance_pair(pca_covariance, comparison_covariance)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    maximum_absolute_covariance = max(
        np.abs(pca_covariance.to_numpy()).max(),
        np.abs(comparison_covariance.to_numpy()).max(),
    )
    minimum_absolute_covariance = min(
        np.abs(pca_covariance.to_numpy()).min(),
        np.abs(comparison_covariance.to_numpy()).min(),
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    date_label = (
        "" if as_of_date is None else f" as of {pd.Timestamp(as_of_date):%Y-%m-%d}"
    )
    heatmap_options = {
        "cmap": "vlag",
        "vmin": minimum_absolute_covariance,
        "vmax": maximum_absolute_covariance,
        "center": np.mean([maximum_absolute_covariance, minimum_absolute_covariance]),
        "square": True,
        "linewidths": 0.2,
        "linecolor": "white",
        "cbar": False,
    }
    pca_image = sns.heatmap(pca_covariance, ax=axes[0], **heatmap_options)
    sns.heatmap(comparison_covariance, ax=axes[1], **heatmap_options)
    axes[0].set_title(f"{pca_label}{date_label}")
    axes[1].set_title(f"{comparison_label}{date_label}")
    for axis in axes:
        axis.tick_params(axis="x", rotation=90)
        axis.tick_params(axis="y", rotation=0)
    figure.colorbar(
        pca_image.collections[0], ax=axes, label="Daily covariance", shrink=0.78
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _filename_component(label: str) -> str:
    component = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return component or "series"


_RISK_RETURN_METRICS = (
    ("annualized_geometric_return", "Annualized geometric return", "higher is better"),
    ("annualized_volatility", "Annualized volatility", "lower is better"),
    ("sharpe_ratio", "Sharpe ratio", "higher is better"),
    ("sortino_ratio", "Sortino ratio", "higher is better"),
    ("maximum_drawdown", "Maximum drawdown", "less negative is better"),
    (
        "tail_expected_shortfall_5pct",
        "5% expected shortfall",
        "less negative is better",
    ),
)
_IMPLEMENTATION_METRICS = (
    ("annualized_turnover", "Annualized target turnover"),
    ("mean_max_weight", "Mean maximum weight"),
    ("mean_effective_number_assets", "Mean effective number of assets"),
)


def _validate_metric_table(
    metrics: pd.DataFrame, required_columns: tuple[str, ...]
) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty DataFrame")
    missing_columns = set(required_columns).difference(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"metrics are missing required columns: {sorted(missing_columns)}"
        )
    selected = metrics.loc[:, required_columns]
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("metrics must contain only finite values")
    return selected


def _bar_colors(labels: pd.Index) -> list[tuple[float, ...]]:
    palette = plt.get_cmap("tab10")
    return [palette(position % palette.N) for position in range(len(labels))]


def save_risk_return_comparison(
    performance_metrics: pd.DataFrame, output_path: str | Path
) -> None:
    """Save six complementary out-of-sample risk/return bar comparisons.

    All series, including baselines, appear in every panel. The figure keeps
    growth separate and concentrates on annualized return, ordinary and downside
    risk-adjusted return, volatility, drawdown, and empirical daily tail loss.
    """
    metric_names = tuple(metric[0] for metric in _RISK_RETURN_METRICS)
    metrics = _validate_metric_table(performance_metrics, metric_names)
    labels = metrics.index
    positions = np.arange(len(labels))
    colors = _bar_colors(labels)
    figure, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for axis, (metric, title, direction) in zip(
        axes.flat, _RISK_RETURN_METRICS, strict=True
    ):
        values = metrics[metric]
        axis.bar(positions, values, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{title} ({direction})")
        axis.set_xticks(positions, labels, rotation=32, ha="right", fontsize=8)
        if metric in {"sharpe_ratio", "sortino_ratio"}:
            axis.set_ylabel("Ratio")
            value_format = ".2f"
        else:
            axis.set_ylabel("Decimal return" if "assets" not in metric else "Assets")
            axis.yaxis.set_major_formatter("{x:.0%}")
            value_format = ".1%"
        for position, value in enumerate(values):
            axis.annotate(
                format(value, value_format),
                (position, value),
                xytext=(0, 3 if value >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_implementation_comparison(
    implementation_metrics: pd.DataFrame, output_path: str | Path
) -> None:
    """Save compact turnover, concentration, and diversification comparisons."""
    metric_names = tuple(metric[0] for metric in _IMPLEMENTATION_METRICS)
    metrics = _validate_metric_table(implementation_metrics, metric_names)
    labels = metrics.index
    positions = np.arange(len(labels))
    colors = _bar_colors(labels)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, _IMPLEMENTATION_METRICS, strict=True):
        values = metrics[metric]
        axis.bar(positions, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=32, ha="right", fontsize=8)
        if metric in {"annualized_turnover", "mean_max_weight"}:
            axis.yaxis.set_major_formatter("{x:.0%}")
            axis.set_ylabel(
                "Target weight turnover"
                if metric == "annualized_turnover"
                else "Weight"
            )
            value_format = ".1%"
        else:
            axis.set_ylabel("Assets")
            value_format = ".2f"
        for position, value in enumerate(values):
            axis.annotate(
                format(value, value_format),
                (position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_return_histograms(
    daily_returns: pd.DataFrame,
    output_directory: str | Path,
    *,
    bins: int = 40,
) -> dict[str, Path]:
    """Save one return-distribution histogram per portfolio or baseline series."""
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("daily_returns must be a non-empty DataFrame")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if (
        daily_returns.isna().any().any()
        or not np.isfinite(daily_returns.to_numpy()).all()
    ):
        raise ValueError("daily_returns must contain only finite values")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for label, series in daily_returns.items():
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.hist(series, bins=bins, density=True, color="#4c78a8", alpha=0.85)
        axis.axvline(0.0, color="black", linewidth=1, label="Zero return")
        axis.axvline(
            series.mean(),
            color="#e45756",
            linewidth=1.5,
            label=f"Mean: {series.mean():.3%}",
        )
        axis.set_title(f"Daily return distribution: {label}")
        axis.set_xlabel("Daily simple return")
        axis.set_ylabel("Density")
        axis.legend()
        path = output_directory / f"return-histogram-{_filename_component(label)}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths[label] = path
    return output_paths
