import numpy as np
import pandas as pd
import pytest

from port_opt.backtest import (
    run_covariance_estimator_study,
    save_fixed_portfolio_variance_comparison,
)


@pytest.fixture
def development_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-01-02", periods=100, freq="B")
    return (
        pd.DataFrame(
            rng.normal(0.0005, 0.01, size=(len(dates), 4)),
            index=dates,
            columns=["XOM", "CVX", "COP", "SLB"],
        ),
        pd.DataFrame(
            rng.normal(0.0, 0.015, size=(len(dates), 3)),
            index=dates,
            columns=["WTI", "Henry Hub", "RBOB"],
        ),
    )


def test_covariance_study_is_walk_forward_and_adds_factor_candidate(
    development_data, tmp_path
):
    returns, factors = development_data
    result = run_covariance_estimator_study(
        returns,
        factor_returns=factors,
        lookback_periods=40,
        rebalance_frequency=10,
        ewma_half_lives=(10, 20),
        num_principal_components=2,
    )

    expected_estimators = {
        "sample_covariance",
        "ledoit_wolf_covariance",
        "pca_covariance",
        "pca_plus_factor_covariance",
        "ewma_covariance_half_life_10",
        "ewma_covariance_half_life_20",
    }
    assert set(result.metrics.index) == expected_estimators
    assert set(result.forecasts.estimator) == expected_estimators
    assert len(result.forecasts) == (len(returns) - 40) * len(expected_estimators)
    assert (result.forecasts.forecast_date > result.forecasts.decision_date).all()
    assert len(result.holding_period_forecasts) == 6 * len(expected_estimators)
    assert (
        result.holding_period_forecasts.holding_period_start
        > result.holding_period_forecasts.decision_date
    ).all()
    assert (result.metrics.holding_periods == 6).all()
    assert (result.metrics.equal_weight_variance_calibration_ratio > 0).all()
    assert (result.metrics.forecasts == len(returns) - 40).all()
    assert np.isfinite(result.metrics.to_numpy()).all()
    comparison_path = tmp_path / "fixed-portfolio-variance-comparison.png"
    save_fixed_portfolio_variance_comparison(result, comparison_path)
    assert comparison_path.exists()


def test_covariance_study_requires_exact_factor_alignment(development_data):
    returns, factors = development_data
    with pytest.raises(ValueError, match="exactly"):
        run_covariance_estimator_study(
            returns,
            factor_returns=factors.iloc[1:],
            lookback_periods=40,
        )
