import numpy as np
import pandas as pd
import pytest

from port_opt.backtest import run_walk_forward_backtest


@pytest.fixture
def returns():
    return pd.DataFrame(
        {
            "energy": [0.01, 0.02, -0.01, 0.03, 0.04, -0.02],
            "utility": [0.0, 0.01, 0.02, -0.01, 0.01, 0.02],
        },
        index=pd.date_range("2025-01-02", periods=6, freq="B"),
    )


def test_walk_forward_uses_only_prior_observations_and_holds_weights(returns):
    training_windows = []

    def estimator(train):
        training_windows.append(train.copy())
        return pd.Series({"energy": 0.75, "utility": 0.25})

    result = run_walk_forward_backtest(
        returns, estimator, lookback_periods=2, rebalance_frequency=2
    )

    assert len(training_windows) == 2
    assert training_windows[0].equals(returns.iloc[0:2])
    assert training_windows[1].equals(returns.iloc[2:4])
    expected = 0.75 * returns["energy"].iloc[2:] + 0.25 * returns["utility"].iloc[2:]
    pd.testing.assert_series_equal(
        result.portfolio_returns, expected.rename("portfolio_return")
    )
    assert result.records[0].in_sample_end < result.records[0].out_of_sample_start
    assert (
        result.weights.iloc[0] == pd.Series({"energy": 0.75, "utility": 0.25})
    ).all()


def test_turnover_and_wealth_are_reported(returns):
    calls = iter(
        [
            pd.Series({"energy": 1.0, "utility": 0.0}),
            pd.Series({"energy": 0.5, "utility": 0.5}),
        ]
    )
    result = run_walk_forward_backtest(
        returns, lambda _: next(calls), lookback_periods=2, rebalance_frequency=2
    )

    assert result.turnover.tolist() == [0.0, 0.5]
    assert result.wealth_index.iloc[-1] == pytest.approx(
        (1.0 - 0.01) * (1.0 + 0.03) * (1.0 + 0.025) * (1.0 + 0.0)
    )


def test_walk_forward_passes_only_aligned_prior_factor_observations(returns):
    factor_returns = pd.DataFrame(
        {"WTI": [0.01, 0.0, -0.01, 0.02, 0.01, -0.02]}, index=returns.index
    )
    factor_windows = []

    def estimator(train, factors):
        factor_windows.append(factors.copy())
        assert train.index.equals(factors.index)
        return pd.Series({"energy": 0.5, "utility": 0.5})

    result = run_walk_forward_backtest(
        returns,
        estimator,
        lookback_periods=2,
        rebalance_frequency=2,
        factor_returns=factor_returns,
    )

    assert len(result.records) == 2
    assert factor_windows[0].equals(factor_returns.iloc[:2])
    assert factor_windows[1].equals(factor_returns.iloc[2:4])


@pytest.mark.parametrize(
    "weights",
    [
        pd.Series({"energy": 0.6, "utility": 0.3}),
        pd.Series({"energy": 1.1, "utility": -0.1}),
        pd.Series({"energy": 1.0}),
    ],
)
def test_invalid_weights_are_rejected(returns, weights):
    with pytest.raises(ValueError):
        run_walk_forward_backtest(returns, lambda _: weights, lookback_periods=2)
