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

    @staticmethod
    def get_cvar_percentile(
        weights: pd.Series | np.ndarray | list[float],
        in_sample_returns: pd.DataFrame,
        cvar_percentile: float = 0.95,
    ) -> float:
        """Return empirical portfolio expected shortfall as a negative return.

        ``cvar_percentile=0.95`` averages the portfolio's realized returns at
        or below its in-sample fifth-percentile return. The result retains the
        return sign: a more negative value indicates greater tail loss. This is
        a portfolio-level calculation, so the selected tail dates are determined
        after asset returns have been combined using ``weights``.
        """
        if not isinstance(in_sample_returns, pd.DataFrame) or in_sample_returns.empty:
            raise ValueError("in_sample_returns must be a non-empty DataFrame")
        if not 0.0 < cvar_percentile < 1.0:
            raise ValueError("cvar_percentile must be between zero and one")
        if (
            in_sample_returns.columns.has_duplicates
            or in_sample_returns.isna().any().any()
            or not np.isfinite(in_sample_returns.to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                "in_sample_returns must have unique columns and finite values"
            )

        if isinstance(weights, pd.Series):
            if not weights.index.is_unique or set(weights.index) != set(
                in_sample_returns.columns
            ):
                raise ValueError(
                    "labelled weights must match the in_sample_returns columns"
                )
            aligned_weights = weights.reindex(in_sample_returns.columns).astype(float)
        else:
            aligned_weights = pd.Series(weights, index=in_sample_returns.columns)
        if (
            len(aligned_weights) != len(in_sample_returns.columns)
            or not np.isfinite(aligned_weights.to_numpy(dtype=float)).all()
        ):
            raise ValueError("weights must contain one finite value per asset")

        portfolio_returns = in_sample_returns.astype(float) @ aligned_weights
        threshold = portfolio_returns.quantile(1.0 - cvar_percentile)
        return float(portfolio_returns[portfolio_returns <= threshold].mean())

    def optimize_towards_tail_adjusted_sharpe_ratio(
        self,
        covariance_matrix,
        port_expected_returns,
        in_sample_returns: pd.DataFrame,
        equity_names: List[str],
        tail_loss_weight: float = 1.0,
        cvar_percentile: float = 0.95,
    ):
        """Optimize a tail-adjusted daily Sharpe ratio.

        The optimized ratio is ``(w' mu - r_f) / (sigma(w) + lambda L(w))``.
        ``sigma(w)`` is covariance-model volatility, ``L(w)`` is positive
        empirical portfolio expected-tail-loss at ``cvar_percentile``, and
        ``lambda`` is ``tail_loss_weight``. Setting ``lambda`` to zero exactly
        recovers the ordinary Sharpe objective.
        """
        if tail_loss_weight < 0:
            raise ValueError("tail_loss_weight cannot be negative")
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

        def negative_tail_adjusted_sharpe_ratio(weights):
            variance = float(weights.T @ covariance @ weights)
            if variance <= 0:
                return np.inf
            tail_loss = max(
                0.0,
                -self.get_cvar_percentile(weights, in_sample_returns, cvar_percentile),
            )
            denominator = np.sqrt(variance) + tail_loss_weight * tail_loss
            if denominator <= 0:
                return np.inf
            return -(float(weights @ expected_returns) - self.r_free) / denominator

        equal_weights = (
            np.ones(num_equities) / num_equities
        )  # initial guess is that we buy everything equally
        mins = optimize.minimize(
            negative_tail_adjusted_sharpe_ratio,
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
