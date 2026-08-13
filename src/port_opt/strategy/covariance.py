"""Reusable covariance estimators for portfolio strategies and research studies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_covariance(returns: pd.DataFrame, half_life: int) -> pd.DataFrame:
    """Estimate a mean-adjusted EWMA covariance with a trading-day half-life.

    The observation at the end of ``returns`` has the largest weight; an
    observation ``half_life`` trading days older has half that weight. Weights
    are normalized to sum to one, yielding a population-style weighted
    covariance rather than applying a sample ``ddof`` correction.
    """
    if half_life < 1:
        raise ValueError("half_life must be positive")
    values = returns.astype(float)
    ages = np.arange(len(values) - 1, -1, -1)
    weights = np.exp(np.log(0.5) * ages / half_life)
    weights /= weights.sum()
    centered = values.to_numpy() - weights @ values.to_numpy()
    covariance = (centered * weights[:, None]).T @ centered
    return pd.DataFrame(covariance, index=values.columns, columns=values.columns)
