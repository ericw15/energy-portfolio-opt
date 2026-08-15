import numpy as np
import pandas as pd

from port_opt.backtest import tail_risk_experiment


def test_tail_risk_experiment_isolates_objective_and_writes_growth(
    monkeypatch, tmp_path
):
    dates = pd.date_range("2023-01-03", periods=50, freq="B")
    columns = [
        *tail_risk_experiment.XLE_TICKERS,
        tail_risk_experiment.BASELINE_TICKER,
    ]
    rng = np.random.default_rng(28)
    downloaded = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates), len(columns))),
        index=dates,
        columns=columns,
    )

    def fake_load_xle_returns(*_args, **_kwargs):
        return (
            downloaded[tail_risk_experiment.XLE_TICKERS],
            downloaded[tail_risk_experiment.BASELINE_TICKER],
        )

    monkeypatch.setattr(tail_risk_experiment, "load_xle_returns", fake_load_xle_returns)
    result = tail_risk_experiment.run_xle_tail_risk_experiment(
        training_start="2023-01-01",
        backtest_start="2023-02-14",
        end_date="2023-04-01",
        lookback_periods=30,
        rebalance_frequency=10,
        num_principal_components=2,
    )

    assert result.daily_returns.columns.tolist() == [
        "PCA covariance / Maximum Sharpe",
        "PCA covariance / Tail-adjusted Sharpe (lambda=1)",
        "PCA covariance / Tail-adjusted Sharpe (lambda=0.1)",
        "PCA covariance / Tail-adjusted Sharpe (lambda=0.01)",
        "Equal-weight XLE constituents",
        "XLE baseline",
    ]
    assert (
        result.performance_metrics.index.tolist()
        == result.daily_returns.columns.tolist()
    )
    assert len(result.implementation_metrics) == 4
    paths = tail_risk_experiment.save_xle_tail_risk_experiment_visuals(
        result, tmp_path / "outputs"
    )
    assert set(paths) == {
        "growth_comparison",
        "performance_metrics",
        "implementation_metrics",
        "risk_return_comparison",
        "implementation_comparison",
    }
    assert all(path.exists() for path in paths.values())
