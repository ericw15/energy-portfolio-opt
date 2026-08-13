import numpy as np
import pandas as pd

from port_opt.backtest import covariance_experiment


def test_covariance_experiment_runs_all_estimators_and_only_writes_growth(
    monkeypatch, tmp_path
):
    dates = pd.date_range("2023-01-03", periods=50, freq="B")
    columns = [
        *covariance_experiment.XLE_TICKERS,
        covariance_experiment.BASELINE_TICKER,
    ]
    rng = np.random.default_rng(18)
    downloaded = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates), len(columns))),
        index=dates,
        columns=columns,
    )

    def fake_load_xle_returns(*_args, **_kwargs):
        return (
            downloaded[covariance_experiment.XLE_TICKERS],
            downloaded[covariance_experiment.BASELINE_TICKER],
        )

    monkeypatch.setattr(
        covariance_experiment, "load_xle_returns", fake_load_xle_returns
    )
    result = covariance_experiment.run_xle_covariance_experiment(
        training_start="2023-01-01",
        backtest_start="2023-02-14",
        end_date="2023-04-01",
        lookback_periods=30,
        rebalance_frequency=10,
        num_principal_components=2,
        hac_lag=5,
    )

    assert result.daily_returns.columns.tolist() == [
        "Markowitz sample covariance / Historical Means Sharpe",
        "PCA covariance / Historical Means Sharpe",
        "Ledoit-Wolf covariance / Historical Means Sharpe",
        "EWMA covariance (63-day half-life) / Historical Means Sharpe",
        "EWMA-PCA covariance (63-day half-life) / Historical Means Sharpe",
        "Equal-weight XLE constituents",
        "XLE baseline",
    ]
    assert (
        result.performance_metrics.index.tolist()
        == result.daily_returns.columns.tolist()
    )
    assert len(result.implementation_metrics) == 5
    assert result.statistical_tests.index.tolist() == [
        "EWMA-PCA covariance versus PCA covariance"
    ]
    assert (
        result.statistical_tests.loc[
            "EWMA-PCA covariance versus PCA covariance", "hac_lag"
        ]
        == 5
    )
    paths = covariance_experiment.save_xle_covariance_experiment_visuals(
        result, tmp_path / "outputs"
    )
    assert set(paths) == {
        "growth_comparison",
        "performance_metrics",
        "implementation_metrics",
        "risk_return_comparison",
        "implementation_comparison",
        "statistical_tests",
    }
    assert all(path.exists() for path in paths.values())
