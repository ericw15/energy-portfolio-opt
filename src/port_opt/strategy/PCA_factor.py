from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .expected_returns import DEFAULT_FEATURE_WINDOWS, get_lightgbm_ER
from .portfolio import Portfolio_Strategy


class PCA_factor_Strategy(Portfolio_Strategy):
    def __init__(self, risk_free_rate, num_principal_components: int | None = None):
        super().__init__(risk_free_rate)
        self.num_principal_components = num_principal_components

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.mean().rename("expected_return")

    def get_covariance_matrix(
        self,
        start_date,
        end_date,
        equity_data,
        num_principal_components: int | None = None,
    ):
        returns = equity_data.astype(float)
        max_components = min(returns.shape)
        if num_principal_components is None:
            num_principal_components = self.num_principal_components
        if num_principal_components is None:
            num_principal_components = min(3, max_components)
        if not 1 <= num_principal_components <= max_components:
            raise ValueError(
                f"num_principal_components must be between 1 and {max_components}"
            )
        pca = PCA(n_components=num_principal_components)
        factor_returns = pca.fit_transform(returns)

        lr = LinearRegression()
        lr.fit(factor_returns, returns)
        predicted_returns = lr.predict(factor_returns)
        residuals = np.var(predicted_returns - returns.to_numpy(), axis=0, ddof=1)

        betas_np = np.array(lr.coef_)
        factor_covariance = np.cov(factor_returns, rowvar=False, ddof=1)
        factor_covariance = np.atleast_2d(factor_covariance)
        asset_covariance_matrix = betas_np @ factor_covariance @ betas_np.T + np.diag(
            residuals
        )
        return pd.DataFrame(
            asset_covariance_matrix, index=returns.columns, columns=returns.columns
        )


class PCA_LightGBM_Strategy(PCA_factor_Strategy):
    """Original PCA covariance prototype paired with its LightGBM return forecast."""

    def __init__(
        self,
        risk_free_rate: float,
        window_size: int = 30,
        num_boost_round: int = 100,
        min_train_samples: int = 20,
        num_principal_components: int | None = None,
        feature_windows: Sequence[int] = DEFAULT_FEATURE_WINDOWS,
    ):
        super().__init__(risk_free_rate, num_principal_components)
        self.window_size = window_size
        self.num_boost_round = num_boost_round
        self.min_train_samples = min_train_samples
        self.feature_windows = feature_windows

    def get_expected_returns(self, start_date, end_date, equity_data):
        return get_lightgbm_ER(
            equity_data,
            window_size=self.window_size,
            num_boost_round=self.num_boost_round,
            min_train_samples=self.min_train_samples,
            feature_windows=self.feature_windows,
        )


class PCA_Historical_Mean_Strategy(PCA_factor_Strategy):
    """PCA covariance paired with full in-sample historical mean returns.

    The mean-return implementation is inherited from ``PCA_factor_Strategy`` so
    its asset labels remain aligned with the covariance matrix and return panel.
    """


class TailAdjustedSharpePCA_Historical_Mean_Strategy(PCA_Historical_Mean_Strategy):
    """PCA covariance with historical means and empirical tail-loss penalty."""

    def __init__(
        self,
        risk_free_rate: float,
        tail_loss_weight: float = 1.0,
        cvar_percentile: float = 0.95,
        num_principal_components: int | None = None,
    ):
        super().__init__(risk_free_rate, num_principal_components)
        if tail_loss_weight < 0:
            raise ValueError("tail_loss_weight cannot be negative")
        if not 0.0 < cvar_percentile < 1.0:
            raise ValueError("cvar_percentile must be between zero and one")
        self.tail_loss_weight = tail_loss_weight
        self.cvar_percentile = cvar_percentile

    def weights_from_returns(self, in_sample_returns: pd.DataFrame) -> pd.Series:
        covariance_matrix = self.get_covariance_matrix(None, None, in_sample_returns)
        expected_returns = self.get_expected_returns(None, None, in_sample_returns)
        result = self.optimize_towards_tail_adjusted_sharpe_ratio(
            covariance_matrix,
            expected_returns,
            in_sample_returns,
            list(in_sample_returns.columns),
            tail_loss_weight=self.tail_loss_weight,
            cvar_percentile=self.cvar_percentile,
        )
        if not result.success:
            raise RuntimeError(f"portfolio optimization failed: {result.message}")
        return pd.Series(result.x, index=in_sample_returns.columns, name="weight")


class EWMAPCA_Historical_Mean_Strategy(PCA_Historical_Mean_Strategy):
    """Historical-mean portfolio with an exponentially weighted PCA covariance.

    Principal directions, factor variances, and residual variances are all fit
    using the same recency weights. This differs from merely applying EWMA to a
    PCA covariance estimated with equally weighted observations.
    """

    def __init__(
        self,
        risk_free_rate: float,
        half_life: int = 63,
        num_principal_components: int | None = None,
    ):
        super().__init__(risk_free_rate, num_principal_components)
        if half_life < 1:
            raise ValueError("half_life must be positive")
        self.half_life = half_life

    def get_covariance_matrix(
        self,
        start_date,
        end_date,
        equity_data,
        num_principal_components: int | None = None,
    ):
        returns = equity_data.astype(float)
        max_components = min(returns.shape)
        component_count = (
            self.num_principal_components
            if num_principal_components is None
            else num_principal_components
        )
        if component_count is None:
            component_count = min(3, max_components)
        if not 1 <= component_count <= max_components:
            raise ValueError(
                f"num_principal_components must be between 1 and {max_components}"
            )

        ages = np.arange(len(returns) - 1, -1, -1)
        weights = np.exp(np.log(0.5) * ages / self.half_life)
        weights /= weights.sum()
        centered_returns = returns.to_numpy() - weights @ returns.to_numpy()
        weighted_covariance = (centered_returns * weights[:, None]).T @ centered_returns
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        loadings = eigenvectors[:, :component_count]
        systematic_covariance = (
            loadings @ np.diag(eigenvalues[:component_count]) @ loadings.T
        )
        residual_variances = np.clip(
            np.diag(weighted_covariance - systematic_covariance), 0.0, None
        )
        covariance = systematic_covariance + np.diag(residual_variances)
        return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)


class Commodity_Factor_Strategy(Portfolio_Strategy):
    """Observed commodity-factor covariance with historical mean returns only."""

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.mean().rename("expected_return")

    @staticmethod
    def _validate_factor_returns(
        equity_returns: pd.DataFrame, commodity_factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        if not isinstance(commodity_factor_returns, pd.DataFrame):
            raise TypeError("commodity_factor_returns must be a DataFrame")
        if commodity_factor_returns.empty:
            raise ValueError("commodity_factor_returns must not be empty")
        if not equity_returns.index.equals(commodity_factor_returns.index):
            raise ValueError(
                "commodity_factor_returns must have exactly the equity return index"
            )
        if commodity_factor_returns.columns.has_duplicates:
            raise ValueError("commodity_factor_returns must have unique columns")
        if (
            commodity_factor_returns.isna().any().any()
            or not np.isfinite(commodity_factor_returns.to_numpy()).all()
        ):
            raise ValueError("commodity_factor_returns must contain only finite values")

        return commodity_factor_returns.astype(float)

    @staticmethod
    def _factor_model_covariance(
        equity_returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        returns = equity_returns.astype(float)
        if (factor_returns.std(ddof=1) == 0).any():
            raise ValueError("all factors must have non-zero variance")
        regression = LinearRegression().fit(factor_returns, returns)
        residuals = returns.to_numpy() - regression.predict(factor_returns)
        idiosyncratic_variance = residuals.var(axis=0, ddof=1)
        covariance = (
            regression.coef_ @ factor_returns.cov().to_numpy() @ regression.coef_.T
            + np.diag(idiosyncratic_variance)
        )
        covariance = (covariance + covariance.T) / 2.0
        return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)

    def get_covariance_matrix_with_factors(
        self,
        equity_returns: pd.DataFrame,
        commodity_factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate covariance using only the observed commodity factors."""
        factors = self._validate_factor_returns(
            equity_returns, commodity_factor_returns
        )
        return self._factor_model_covariance(equity_returns, factors)

    def weights_from_equity_and_factor_returns(
        self,
        in_sample_returns: pd.DataFrame,
        in_sample_factor_returns: pd.DataFrame,
    ) -> pd.Series:
        covariance_matrix = self.get_covariance_matrix_with_factors(
            in_sample_returns, in_sample_factor_returns
        )
        expected_returns = self.get_expected_returns(None, None, in_sample_returns)
        result = self.optimize_towards_sharpe_ratio(
            covariance_matrix, expected_returns, list(in_sample_returns.columns)
        )
        if not result.success:
            raise RuntimeError(f"portfolio optimization failed: {result.message}")
        return pd.Series(result.x, index=in_sample_returns.columns, name="weight")


class PCA_Commodity_Factor_Strategy(Commodity_Factor_Strategy):
    """PCA covariance enriched with observed commodity-return factors."""

    def __init__(self, risk_free_rate, num_principal_components: int | None = None):
        super().__init__(risk_free_rate)
        self.num_principal_components = num_principal_components

    def get_covariance_matrix_with_factors(
        self,
        equity_returns: pd.DataFrame,
        commodity_factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate covariance from the combined PCA and commodity factor panel."""
        commodity_factor_returns = self._validate_factor_returns(
            equity_returns, commodity_factor_returns
        )
        returns = equity_returns.astype(float)
        pca_factor_count = self.num_principal_components
        if pca_factor_count is None:
            pca_factor_count = min(3, min(returns.shape))
        if not 1 <= pca_factor_count <= min(returns.shape):
            raise ValueError(
                f"num_principal_components must be between 1 and {min(returns.shape)}"
            )
        pca_factors = PCA(n_components=pca_factor_count).fit_transform(returns)
        factor_returns = pd.concat(
            [
                pd.DataFrame(
                    pca_factors,
                    index=returns.index,
                    columns=[
                        f"pca_factor_{number}" for number in range(pca_factor_count)
                    ],
                ),
                commodity_factor_returns.astype(float),
            ],
            axis=1,
        )
        return self._factor_model_covariance(returns, factor_returns)
