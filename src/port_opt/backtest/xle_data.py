"""Shared XLE research universe, market-data loading, and panel preparation."""

from __future__ import annotations

import time
from typing import Mapping

import pandas as pd

from port_opt.strategy import get_returns

# Static research universe. It is easy to reproduce but introduces survivorship
# bias until it is replaced with a point-in-time constituent history.
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


def _download_returns_with_retries(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    max_download_attempts: int,
    retry_delay_seconds: float,
    description: str,
) -> pd.DataFrame:
    if max_download_attempts < 1:
        raise ValueError("max_download_attempts must be positive")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")

    last_error: Exception | None = None
    for attempt in range(max_download_attempts):
        try:
            downloaded_returns = get_returns(
                tickers, start_date=start_date, end_date=end_date
            )
            if not downloaded_returns.empty:
                return downloaded_returns
            last_error = ValueError("provider returned no observations")
        except Exception as error:
            last_error = error
        if attempt + 1 < max_download_attempts:
            time.sleep(retry_delay_seconds * (2**attempt))
    raise RuntimeError(
        f"unable to download non-empty {description} returns after "
        f"{max_download_attempts} attempts"
    ) from last_error


def load_xle_returns(
    start_date: str,
    end_date: str,
    *,
    max_download_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch complete-case constituent and XLE daily simple returns."""
    downloaded_returns = _download_returns_with_retries(
        [*XLE_TICKERS, BASELINE_TICKER],
        start_date,
        end_date,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
        description="XLE constituent",
    )
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
    """Fetch complete-case U.S.-traded commodity factor daily simple returns."""
    if not commodity_factors:
        raise ValueError("commodity_factors must not be empty")
    if len(set(commodity_factors)) != len(commodity_factors) or len(
        set(commodity_factors.values())
    ) != len(commodity_factors):
        raise ValueError("commodity factor tickers and labels must be unique")
    downloaded_returns = _download_returns_with_retries(
        list(commodity_factors),
        start_date,
        end_date,
        max_download_attempts=max_download_attempts,
        retry_delay_seconds=retry_delay_seconds,
        description="commodity factor",
    )
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
    """Select exactly the requested prior history plus all subsequent returns."""
    requested_start = pd.Timestamp(backtest_start)
    start_position = asset_returns.index.searchsorted(requested_start)
    if start_position >= len(asset_returns):
        raise ValueError("backtest_start falls after the available return observations")
    if start_position < 1:
        raise ValueError(
            "at least one return observation is required before backtest_start"
        )
    if lookback_periods < 1:
        raise ValueError("lookback_periods must be positive")
    if lookback_periods > start_position:
        raise ValueError(
            "lookback_periods exceeds the observations available before backtest_start"
        )
    return asset_returns.iloc[start_position - lookback_periods :], lookback_periods
