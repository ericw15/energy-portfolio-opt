import numpy as np
import pandas as pd
import pytest

from port_opt.backtest import pca_dimension_experiment


def test_pca_dimension_experiment_compares_grid_and_writes_standard_products(
    monkeypatch, tmp_path
):
    dates = pd.date_range("2020-01-03", periods=60, freq="B")
    columns = [
        *pca_dimension_experiment.XLE_TICKERS,
        pca_dimension_experiment.BASELINE_TICKER,
    ]
    rng = np.random.default_rng(48)
    downloaded = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates), len(columns))),
        index=dates,
        columns=columns,
    )

    def fake_load_xle_returns(*_args, **_kwargs):
        return (
            downloaded[pca_dimension_experiment.XLE_TICKERS],
            downloaded[pca_dimension_experiment.BASELINE_TICKER],
        )

    monkeypatch.setattr(
        pca_dimension_experiment, "load_xle_returns", fake_load_xle_returns
    )
    result = pca_dimension_experiment.run_xle_pca_dimension_experiment(
        training_start="2020-01-01",
        backtest_start="2020-02-14",
        end_date="2020-04-01",
        lookback_periods=30,
        rebalance_frequency=10,
        num_principal_components=(1, 2, 3),
    )

    assert result.daily_returns.columns.tolist() == [
        "PCA covariance (1 component) / Historical Means Sharpe",
        "PCA covariance (2 components) / Historical Means Sharpe",
        "PCA covariance (3 components) / Historical Means Sharpe",
        "Equal-weight XLE constituents",
        "XLE baseline",
    ]
    assert len(result.implementation_metrics) == 3
    paths = pca_dimension_experiment.save_xle_pca_dimension_experiment_visuals(
        result, tmp_path / "outputs"
    )
    assert set(paths) == {
        "growth_comparison",
        "risk_return_comparison",
        "implementation_comparison",
        "performance_metrics",
        "implementation_metrics",
    }
    assert all(path.exists() for path in paths.values())


@pytest.mark.parametrize("component_grid", [(), (1, 1), (0,), (20,)])
def test_pca_dimension_experiment_rejects_invalid_component_grid(component_grid):
    with pytest.raises(ValueError):
        pca_dimension_experiment._validate_component_grid(component_grid, 19)
