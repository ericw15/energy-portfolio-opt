from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize


def get_returns(tickers, start_date, end_date):
    # Download historical data
    data = yf.download(tickers=tickers, start=start_date, end=end_date, progress=False)
    # Calculate daily percentage returns using Adjusted Close
    return data["Close"].pct_change().dropna()


class Portfolio_Strategy:
    def __init__(self, risk_free_rate: float):
        """
        risk_free_rate:float | Should be risk free rate per trading day. At 4% annualized, this might be (4 / 252).
        """
        self.r_free = risk_free_rate

    def optimize_towards_sharpe_ratio(
        self, covariance_matrix, port_expected_returns, equity_names: List[str]
    ):
        """Optimize long-only fully-invested weights for daily Sharpe ratio.

        This remains the compatibility API for the original strategies.  New
        backtests should call :meth:`weights_from_returns`, which returns labelled
        weights and can be passed directly to ``run_walk_forward_backtest``.
        """
        num_equities = len(equity_names)
        covariance = np.asarray(covariance_matrix, dtype=float)
        expected_returns = np.asarray(port_expected_returns, dtype=float).reshape(-1)
        if covariance.shape != (num_equities, num_equities):
            raise ValueError("covariance_matrix must be square with one row per asset")
        if expected_returns.shape != (num_equities,):
            raise ValueError("port_expected_returns must contain one value per asset")
        if not np.isfinite(covariance).all() or not np.isfinite(expected_returns).all():
            raise ValueError(
                "covariance_matrix and port_expected_returns must be finite"
            )

        def negative_sharpe_ratio(weights):
            variance = float(weights.T @ covariance @ weights)
            if variance <= 0:
                return np.inf
            return -(float(weights @ expected_returns) - self.r_free) / np.sqrt(
                variance
            )

        equal_weights = (
            np.ones(num_equities) / num_equities
        )  # initial guess is that we buy everything equally
        mins = optimize.minimize(
            negative_sharpe_ratio,
            equal_weights,
            bounds=[(0, 1)] * num_equities,
            constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1},
        )
        return mins

    def weights_from_returns(self, in_sample_returns: pd.DataFrame) -> pd.Series:
        """Fit this strategy to an in-sample return panel and return target weights.

        The method deliberately does not fetch data. It is the adapter between the
        existing strategy API and the walk-forward backtester.
        """
        if not isinstance(in_sample_returns, pd.DataFrame) or in_sample_returns.empty:
            raise ValueError("in_sample_returns must be a non-empty DataFrame")
        covariance_matrix = self.get_covariance_matrix(None, None, in_sample_returns)
        expected_returns = self.get_expected_returns(None, None, in_sample_returns)
        result = self.optimize_towards_sharpe_ratio(
            covariance_matrix, expected_returns, list(in_sample_returns.columns)
        )
        if not result.success:
            raise RuntimeError(f"portfolio optimization failed: {result.message}")
        return pd.Series(result.x, index=in_sample_returns.columns, name="weight")

    def _get_returns(self, tickers, start_date, end_date):
        return get_returns(tickers, start_date, end_date)

    # Should be overridden
    def get_equity_data(self, start_date, end_date, equity_names):
        return self._get_returns(
            tickers=equity_names, start_date=start_date, end_date=end_date
        )

    def get_covariance_matrix(self, start_date, end_date, equity_data):
        raise NotImplementedError

    def get_expected_returns(self, start_date, end_date, equity_data):
        raise NotImplementedError

    def get_portfolio_weights_sharpe(self, start_date, end_date, equity_names):
        equity_data = self.get_equity_data(start_date, end_date, equity_names)
        covariance_matrix = self.get_covariance_matrix(
            start_date, end_date, equity_data
        )
        port_expected_returns = self.get_expected_returns(
            start_date, end_date, equity_data
        )

        return self.optimize_towards_sharpe_ratio(
            covariance_matrix, port_expected_returns, equity_names
        )
