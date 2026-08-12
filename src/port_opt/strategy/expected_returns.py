"""Expected-return estimators and a leakage-aware model-development study.

All estimators predict the mean *daily* simple return over a future horizon. The
study module is deliberately data-in/data-out: provide a development period that
is excluded from portfolio backtests, then select an approach before backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .portfolio import get_returns

DEFAULT_FEATURE_WINDOWS = (5, 21, 63)


@dataclass(frozen=True)
class ExpectedReturnStudyResult:
    """Out-of-sample development-period predictions and aggregate error metrics."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    development_end: pd.Timestamp


@dataclass
class DiagonalGaussianHMM:
    """Small Gaussian HMM with diagonal state covariances for return regimes."""

    n_states: int = 2
    max_iterations: int = 100
    tolerance: float = 1e-6
    minimum_variance: float = 1e-8

    initial_probabilities: np.ndarray | None = None
    transition_matrix: np.ndarray | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None

    def _log_emissions(self, observations: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.variances is not None
        log_variance = np.log(self.variances)
        squared_distance = (observations[:, None, :] - self.means[None, :, :]) ** 2
        return -0.5 * (
            (np.log(2.0 * np.pi) + log_variance).sum(axis=1)[None, :]
            + (squared_distance / self.variances[None, :, :]).sum(axis=2)
        )

    def _forward_backward(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        assert (
            self.initial_probabilities is not None
            and self.transition_matrix is not None
        )
        log_emissions = self._log_emissions(observations)
        log_initial = np.log(self.initial_probabilities)
        log_transition = np.log(self.transition_matrix)
        log_forward = np.empty_like(log_emissions)
        log_forward[0] = log_initial + log_emissions[0]
        for position in range(1, len(observations)):
            log_forward[position] = log_emissions[position] + logsumexp(
                log_forward[position - 1][:, None] + log_transition, axis=0
            )
        log_likelihood = float(logsumexp(log_forward[-1]))

        log_backward = np.zeros_like(log_emissions)
        for position in range(len(observations) - 2, -1, -1):
            log_backward[position] = logsumexp(
                log_transition
                + log_emissions[position + 1][None, :]
                + log_backward[position + 1][None, :],
                axis=1,
            )
        posterior = np.exp(log_forward + log_backward - log_likelihood)
        transition_posterior = np.empty(
            (len(observations) - 1, self.n_states, self.n_states)
        )
        for position in range(len(observations) - 1):
            transition_posterior[position] = np.exp(
                log_forward[position][:, None]
                + log_transition
                + log_emissions[position + 1][None, :]
                + log_backward[position + 1][None, :]
                - log_likelihood
            )
        return posterior, transition_posterior, log_likelihood

    def fit(self, returns: pd.DataFrame) -> "DiagonalGaussianHMM":
        observations = _validate_returns(returns).to_numpy()
        if self.n_states < 2 or len(observations) < self.n_states * 20:
            raise ValueError("HMM requires at least 20 observations per state")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        first_component = observations @ np.ones(observations.shape[1])
        assignments = pd.qcut(
            first_component, q=self.n_states, labels=False, duplicates="drop"
        )
        if len(np.unique(assignments)) != self.n_states:
            raise ValueError(
                "return history does not support distinct HMM state initialization"
            )
        self.means = np.vstack(
            [
                observations[assignments == state].mean(axis=0)
                for state in range(self.n_states)
            ]
        )
        self.variances = np.vstack(
            [
                np.maximum(
                    observations[assignments == state].var(axis=0),
                    self.minimum_variance,
                )
                for state in range(self.n_states)
            ]
        )
        self.initial_probabilities = np.full(self.n_states, 1.0 / self.n_states)
        self.transition_matrix = np.full(
            (self.n_states, self.n_states), 0.05 / (self.n_states - 1)
        )
        np.fill_diagonal(self.transition_matrix, 0.95)

        previous_likelihood = -np.inf
        for _ in range(self.max_iterations):
            posterior, transitions, likelihood = self._forward_backward(observations)
            self.initial_probabilities = posterior[0] / posterior[0].sum()
            transition_denominator = posterior[:-1].sum(axis=0)[:, None]
            self.transition_matrix = transitions.sum(axis=0) / np.maximum(
                transition_denominator, 1e-12
            )
            self.transition_matrix = np.maximum(self.transition_matrix, 1e-12)
            self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)
            state_weights = posterior.sum(axis=0)
            self.means = posterior.T @ observations / state_weights[:, None]
            residuals = observations[:, None, :] - self.means[None, :, :]
            self.variances = np.maximum(
                (posterior[:, :, None] * residuals**2).sum(axis=0)
                / state_weights[:, None],
                self.minimum_variance,
            )
            if likelihood - previous_likelihood < self.tolerance:
                break
            previous_likelihood = likelihood
        return self

    def forecast_average_return(
        self, observed_returns: pd.DataFrame, horizon: int
    ) -> pd.Series:
        """Forecast mean daily return across the next ``horizon`` observations."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        observations = _validate_returns(observed_returns).to_numpy()
        posterior, _, _ = self._forward_backward(observations)
        assert self.transition_matrix is not None and self.means is not None
        state_probability = posterior[-1]
        expected_returns = np.zeros(self.means.shape[1])
        for _ in range(horizon):
            state_probability = state_probability @ self.transition_matrix
            expected_returns += state_probability @ self.means
        return pd.Series(expected_returns / horizon, index=observed_returns.columns)


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns must contain only finite values")
    return returns.astype(float)


def _minimum_history(feature_windows: Sequence[int]) -> int:
    if not feature_windows or min(feature_windows) < 2:
        raise ValueError("feature_windows must contain integers of at least two")
    return max(max(feature_windows), 26, 14, 5)


def _rsi(returns: pd.Series, period: int = 14) -> float:
    changes = returns.iloc[-period:]
    gains = changes.clip(lower=0).mean()
    losses = -changes.clip(upper=0).mean()
    if np.isclose(losses, 0.0):
        return 100.0 if gains > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + gains / losses))


def _feature_row(
    history: pd.Series, feature_windows: Sequence[int]
) -> dict[str, float]:
    minimum_history = _minimum_history(feature_windows)
    if len(history) < minimum_history:
        raise ValueError("history is shorter than the requested feature requirements")

    features: dict[str, float] = {}
    for lag in range(1, 6):
        features[f"lag_{lag}"] = float(history.iloc[-lag])
    for window in feature_windows:
        trailing = history.iloc[-window:]
        features[f"mean_{window}"] = float(trailing.mean())
        features[f"volatility_{window}"] = float(trailing.std(ddof=1))
        features[f"momentum_{window}"] = float((1.0 + trailing).prod() - 1.0)
    downside = history.iloc[-21:].clip(upper=0)
    features["downside_volatility_21"] = float(downside.std(ddof=1))
    features["rsi_14"] = _rsi(history)

    price_index = (1.0 + history).cumprod()
    fast = price_index.ewm(span=12, adjust=False).mean()
    slow = price_index.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    features["macd_normalized"] = float(macd.iloc[-1] / price_index.iloc[-1])
    features["macd_signal_gap"] = float(
        (macd.iloc[-1] - macd.ewm(span=9, adjust=False).mean().iloc[-1])
        / price_index.iloc[-1]
    )
    return features


def _supervised_samples(
    returns: pd.Series,
    *,
    horizon: int,
    feature_windows: Sequence[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    minimum_history = _minimum_history(feature_windows)
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    dates: list[pd.Timestamp] = []
    for decision_position in range(minimum_history - 1, len(returns) - horizon):
        rows.append(
            _feature_row(returns.iloc[: decision_position + 1], feature_windows)
        )
        targets.append(
            float(
                returns.iloc[
                    decision_position + 1 : decision_position + 1 + horizon
                ].mean()
            )
        )
        dates.append(returns.index[decision_position])
    return (
        pd.DataFrame(rows, index=dates),
        pd.Series(targets, index=dates),
        pd.DatetimeIndex(dates),
    )


def get_lightgbm_ER(
    equity_data: pd.DataFrame,
    use_val_for_hyperparameter_optimization: bool = False,
    window_size: int = 30,
    num_boost_round: int = 100,
    min_train_samples: int = 20,
    feature_windows: Sequence[int] = DEFAULT_FEATURE_WINDOWS,
) -> pd.Series:
    """Forecast per-asset future mean daily returns using technical return features.

    ``window_size`` is retained as the prediction horizon for compatibility. The
    validation flag is retained but tuning is intentionally not performed here;
    use :func:`run_expected_return_study` on a separate development period first.
    """
    equity_data = _validate_returns(equity_data)
    if window_size < 1 or num_boost_round < 1 or min_train_samples < 1:
        raise ValueError(
            "window_size, num_boost_round, and min_train_samples must be positive"
        )

    predictions: dict[str, float] = {}
    for asset in equity_data:
        features, targets, _ = _supervised_samples(
            equity_data[asset], horizon=window_size, feature_windows=feature_windows
        )
        if len(features) < min_train_samples:
            raise ValueError(
                "equity_data is too short for the requested features and training samples"
            )
        model = lgb.train(
            {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1,
            },
            lgb.Dataset(features, label=targets),
            num_boost_round=num_boost_round,
        )
        latest_features = pd.DataFrame(
            [_feature_row(equity_data[asset], feature_windows)]
        )
        predictions[asset] = float(model.predict(latest_features)[0])
    return pd.Series(predictions, name="expected_return")


def run_expected_return_study(
    development_returns: pd.DataFrame,
    *,
    horizon: int = 21,
    validation_fraction: float = 0.25,
    rolling_mean_window: int = 63,
    feature_windows: Sequence[int] = DEFAULT_FEATURE_WINDOWS,
    lightgbm_num_boost_round: int = 100,
    hmm_states: int = 2,
) -> ExpectedReturnStudyResult:
    """Compare expected-return methods on a chronologically purged dev holdout.

    Model training targets end before the validation period begins. The supplied
    panel is a development-only period and should never be included in a later
    portfolio backtest used to report performance.
    """
    returns = _validate_returns(development_returns)
    if horizon < 1 or rolling_mean_window < 1:
        raise ValueError("horizon and rolling_mean_window must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    validation_start = int(len(returns) * (1.0 - validation_fraction))
    if validation_start <= _minimum_history(feature_windows) + horizon:
        raise ValueError(
            "development period is too short for a purged validation split"
        )

    hmm_training_returns = returns.iloc[:validation_start]
    hmm = DiagonalGaussianHMM(n_states=hmm_states).fit(hmm_training_returns)
    hmm_predictions: dict[pd.Timestamp, pd.Series] = {}

    prediction_rows: list[dict[str, object]] = []
    for asset in returns:
        features, targets, decision_dates = _supervised_samples(
            returns[asset], horizon=horizon, feature_windows=feature_windows
        )
        target_end_positions = np.array(
            [returns.index.get_loc(date) + horizon for date in decision_dates]
        )
        train_mask = target_end_positions < validation_start
        validation_mask = np.array(
            [returns.index.get_loc(date) >= validation_start for date in decision_dates]
        )
        train_features, train_targets = (
            features.iloc[train_mask],
            targets.iloc[train_mask],
        )
        validation_features, validation_targets = (
            features.iloc[validation_mask],
            targets.iloc[validation_mask],
        )
        validation_dates = decision_dates[validation_mask]
        if len(train_features) < 20 or validation_features.empty:
            raise ValueError(
                "development period is too short for the requested study split"
            )

        linear = LinearRegression().fit(train_features, train_targets)
        scaler = StandardScaler().fit(train_features)
        standardized_train = scaler.transform(train_features)
        standardized_validation = scaler.transform(validation_features)
        lasso = LassoCV(
            alphas=np.logspace(-5, -2, 7),
            cv=TimeSeriesSplit(n_splits=3, gap=horizon),
            max_iter=100_000,
            tol=1e-6,
            random_state=0,
        ).fit(standardized_train, train_targets)
        direction = (train_targets > 0).astype(int)
        if direction.nunique() == 1:
            logistic_predictions = np.repeat(
                float(train_targets.mean()), len(validation_features)
            )
        else:
            logistic = LogisticRegression(max_iter=1_000, random_state=0).fit(
                standardized_train, direction
            )
            probability_up = logistic.predict_proba(standardized_validation)[:, 1]
            positive_mean = train_targets[train_targets > 0].mean()
            negative_mean = train_targets[train_targets <= 0].mean()
            logistic_predictions = (
                probability_up * positive_mean + (1.0 - probability_up) * negative_mean
            )
        lightgbm_model = lgb.train(
            {"objective": "regression", "metric": "rmse", "verbose": -1},
            lgb.Dataset(train_features, label=train_targets),
            num_boost_round=lightgbm_num_boost_round,
        )

        for position, (date, observed) in enumerate(
            zip(validation_dates, validation_targets)
        ):
            history = returns.loc[:date, asset]
            if date not in hmm_predictions:
                hmm_predictions[date] = hmm.forecast_average_return(
                    returns.loc[:date], horizon
                )
            estimates = {
                "historical_mean": float(history.mean()),
                "rolling_mean": float(history.iloc[-rolling_mean_window:].mean()),
                "linear_regression": float(
                    linear.predict(validation_features.iloc[[position]])[0]
                ),
                "lasso_regression": float(
                    lasso.predict(standardized_validation[[position]])[0]
                ),
                "logistic_regression": float(logistic_predictions[position]),
                "lightgbm_technical": float(
                    lightgbm_model.predict(validation_features.iloc[[position]])[0]
                ),
                "gaussian_hmm": float(hmm_predictions[date][asset]),
            }
            for estimator, prediction in estimates.items():
                prediction_rows.append(
                    {
                        "decision_date": date,
                        "asset": asset,
                        "estimator": estimator,
                        "predicted_return": prediction,
                        "observed_return": float(observed),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    grouped = predictions.groupby("estimator", sort=True)
    metrics = grouped.apply(
        lambda frame: pd.Series(
            {
                "mae": (frame.predicted_return - frame.observed_return).abs().mean(),
                "rmse": np.sqrt(
                    ((frame.predicted_return - frame.observed_return) ** 2).mean()
                ),
                "directional_accuracy": (
                    np.sign(frame.predicted_return) == np.sign(frame.observed_return)
                ).mean(),
                "observations": len(frame),
            }
        ),
        include_groups=False,
    ).sort_values("rmse")
    return ExpectedReturnStudyResult(
        metrics=metrics,
        predictions=predictions,
        development_end=returns.index[-1],
    )


if __name__ == "__main__":
    validation_returns = get_returns(
        "VOO", start_date="2021-01-01", end_date="2026-01-01"
    )
    results = run_expected_return_study(validation_returns, lightgbm_num_boost_round=1)
    breakpoint()
