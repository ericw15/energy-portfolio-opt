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

    Both heatmaps share one symmetric color scale, making their cell magnitudes
    directly comparable. ``as_of_date`` is a display label; callers determine
    the return history used to estimate each matrix.
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
