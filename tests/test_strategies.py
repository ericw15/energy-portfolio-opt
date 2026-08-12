import numpy as np
import pandas as pd
import pytest

from port_opt.backtest import run_walk_forward_backtest
from port_opt.strategy import (
    Markowitz_Portfolio,
    PCA_Historical_Mean_Strategy,
    PCA_Commodity_Factor_Strategy,
    Commodity_Factor_Strategy,
    PCA_LightGBM_Strategy,
    PCA_factor_Strategy,
    Rolling_Markowitz_Portfolio,
    get_lightgbm_ER,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    values = rng.normal(0.0005, 0.01, size=(160, 3))
    return pd.DataFrame(
        values,
        columns=["XOM", "CVX", "NEE"],
        index=pd.date_range("2024-01-02", periods=160, freq="B"),
    )


def test_markowitz_strategy_is_a_walk_forward_weight_estimator(returns):
    strategy = Markowitz_Portfolio(risk_free_rate=0.0)
    result = run_walk_forward_backtest(
        returns,
        strategy.weights_from_returns,
        lookback_periods=20,
        rebalance_frequency=10,
    )

    assert len(result.portfolio_returns) == 140
    assert result.weights.columns.tolist() == returns.columns.tolist()
    assert np.allclose(result.weights.sum(axis=1), 1.0)


def test_rolling_markowitz_uses_only_its_last_rolling_window(returns):
    strategy = Rolling_Markowitz_Portfolio(risk_free_rate=0.0, rolling_days=5)
    expected = returns.iloc[-5:].mean()

    pd.testing.assert_series_equal(
        strategy.get_expected_returns(None, None, returns), expected
    )


def test_pca_covariance_preserves_asset_labels_and_is_symmetric(returns):
    covariance = PCA_factor_Strategy(0.0).get_covariance_matrix(
        None, None, returns, num_principal_components=2
    )

    assert covariance.index.tolist() == returns.columns.tolist()
    assert covariance.columns.tolist() == returns.columns.tolist()
    assert np.allclose(covariance, covariance.T)


def test_pca_historical_mean_strategy_preserves_asset_labels(returns):
    strategy = PCA_Historical_Mean_Strategy(risk_free_rate=0.0)

    expected_returns = strategy.get_expected_returns(None, None, returns)

    pd.testing.assert_series_equal(
        expected_returns, returns.mean().rename("expected_return")
    )
    result = run_walk_forward_backtest(
        returns,
        strategy.weights_from_returns,
        lookback_periods=120,
        rebalance_frequency=10,
    )
    assert result.weights.columns.tolist() == returns.columns.tolist()


def test_pca_commodity_factor_covariance_preserves_asset_labels(returns):
    rng = np.random.default_rng(81)
    commodity_factors = pd.DataFrame(
        rng.normal(0.0002, 0.015, size=(len(returns), 3)),
        columns=["WTI crude oil", "Henry Hub natural gas", "RBOB gasoline"],
        index=returns.index,
    )
    strategy = PCA_Commodity_Factor_Strategy(risk_free_rate=0.0)

    covariance = strategy.get_covariance_matrix_with_factors(returns, commodity_factors)
    result = run_walk_forward_backtest(
        returns,
        strategy.weights_from_equity_and_factor_returns,
        lookback_periods=120,
        rebalance_frequency=10,
        factor_returns=commodity_factors,
    )

    assert covariance.index.tolist() == returns.columns.tolist()
    assert covariance.columns.tolist() == returns.columns.tolist()
    assert np.allclose(covariance, covariance.T)
    assert len(result.records) == 4


def test_commodity_only_factor_covariance_preserves_asset_labels(returns):
    rng = np.random.default_rng(82)
    commodity_factors = pd.DataFrame(
        rng.normal(0.0002, 0.015, size=(len(returns), 3)),
        columns=["WTI crude oil", "Henry Hub natural gas", "RBOB gasoline"],
        index=returns.index,
    )
    strategy = Commodity_Factor_Strategy(risk_free_rate=0.0)

    covariance = strategy.get_covariance_matrix_with_factors(returns, commodity_factors)
    result = run_walk_forward_backtest(
        returns,
        strategy.weights_from_equity_and_factor_returns,
        lookback_periods=120,
        rebalance_frequency=10,
        factor_returns=commodity_factors,
    )

    assert covariance.index.tolist() == returns.columns.tolist()
    assert covariance.columns.tolist() == returns.columns.tolist()
    assert np.allclose(covariance, covariance.T)
    assert len(result.records) == 4


def test_lightgbm_forecast_is_labelled_and_pca_lightgbm_backtests(returns):
    forecasts = get_lightgbm_ER(
        returns, window_size=5, min_train_samples=10, num_boost_round=1
    )
    assert forecasts.index.tolist() == returns.columns.tolist()
    assert np.isfinite(forecasts).all()

    strategy = PCA_LightGBM_Strategy(
        risk_free_rate=0.0,
        window_size=5,
        min_train_samples=10,
        num_boost_round=1,
    )
    result = run_walk_forward_backtest(
        returns,
        strategy.weights_from_returns,
        lookback_periods=120,
        rebalance_frequency=10,
    )
    assert len(result.records) == 4


def test_lightgbm_rejects_insufficient_history(returns):
    with pytest.raises(ValueError, match="too short"):
        get_lightgbm_ER(returns.iloc[:18], window_size=5, min_train_samples=10)


def test_expected_return_study_uses_a_purged_chronological_holdout():
    rng = np.random.default_rng(12)
    development_returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(220, 2)),
        columns=["XOM", "CVX"],
        index=pd.date_range("2019-01-02", periods=220, freq="B"),
    )
    from port_opt.strategy import run_expected_return_study

    result = run_expected_return_study(
        development_returns, horizon=10, lightgbm_num_boost_round=1
    )

    assert set(result.metrics.index) == {
        "historical_mean",
        "gaussian_hmm",
        "lasso_regression",
        "lightgbm_technical",
        "linear_regression",
        "logistic_regression",
        "rolling_mean",
    }
    assert (result.predictions.decision_date > result.development_end).sum() == 0
    assert (result.metrics.observations > 0).all()
