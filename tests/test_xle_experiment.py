import numpy as np
import pandas as pd
import pytest

from port_opt.backtest import xle_experiment
from port_opt.backtest import save_covariance_comparison, save_return_histograms
from port_opt.backtest.xle_experiment import (
    save_xle_experiment_visuals,
    select_backtest_panel,
)


@pytest.fixture
def returns():
    return pd.DataFrame(
        {"XOM": range(6)}, index=pd.date_range("2024-01-02", periods=6, freq="B")
    )


def test_xle_experiment_uses_requested_rolling_history(returns):
    panel, lookback = select_backtest_panel(returns, "2024-01-05", 3)

    assert lookback == 3
    assert panel.index[0] == returns.index[0]
    assert panel.index[-1] == returns.index[-1]


def test_xle_experiment_honors_explicit_lookback(returns):
    panel, lookback = select_backtest_panel(returns, "2024-01-05", 2)

    assert lookback == 2
    assert panel.index[0] == returns.index[1]


def test_xle_experiment_rejects_unavailable_explicit_lookback(returns):
    with pytest.raises(ValueError, match="exceeds"):
        select_backtest_panel(returns, "2024-01-05", 4)


def test_xle_data_download_retries_transient_empty_result(monkeypatch):
    calls = 0

    def fake_get_returns(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame()
        return pd.DataFrame(
            {ticker: [0.01] for ticker in [*xle_experiment.XLE_TICKERS, "XLE"]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )

    monkeypatch.setattr(xle_experiment, "get_returns", fake_get_returns)
    downloaded, _ = xle_experiment.load_xle_returns(
        "2024-01-01",
        "2024-01-03",
        max_download_attempts=2,
        retry_delay_seconds=0,
    )

    assert calls == 2
    assert not downloaded.empty


def test_historical_mean_experiment_compares_pca_and_markowitz(monkeypatch, tmp_path):
    dates = pd.date_range("2023-01-03", periods=50, freq="B")
    columns = [*xle_experiment.XLE_TICKERS, xle_experiment.BASELINE_TICKER]
    rng = np.random.default_rng(8)
    downloaded = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates), len(columns))),
        index=dates,
        columns=columns,
    )

    def fake_load_xle_returns(*_args, **_kwargs):
        return (
            downloaded[xle_experiment.XLE_TICKERS],
            downloaded[xle_experiment.BASELINE_TICKER],
        )

    def fake_load_commodity_factor_returns(*_args, **_kwargs):
        return pd.DataFrame(
            {
                label: downloaded[xle_experiment.XLE_TICKERS[0]]
                for label in xle_experiment.DEFAULT_COMMODITY_FACTORS.values()
            },
            index=downloaded.index,
        )

    monkeypatch.setattr(xle_experiment, "load_xle_returns", fake_load_xle_returns)
    monkeypatch.setattr(
        xle_experiment,
        "load_commodity_factor_returns",
        fake_load_commodity_factor_returns,
    )
    result = xle_experiment.run_xle_pca_historical_mean_experiment(
        training_start="2023-01-01",
        backtest_start="2023-02-14",
        end_date="2023-04-01",
        lookback_periods=30,
        rebalance_frequency=10,
        num_principal_components=2,
    )

    assert result.markowitz_backtest is not None
    assert result.commodity_factor_backtest is not None
    assert result.commodity_only_backtest is not None
    assert result.daily_returns.columns.tolist() == [
        "PCA factor / Historical Means Sharpe",
        "PCA + U.S. commodity factors / Historical Means Sharpe",
        "U.S. commodity factors only / Historical Means Sharpe",
        "Markowitz / Historical Means Sharpe",
        "Equal-weight XLE constituents",
        "XLE baseline",
    ]
    assert result.strategy_backtest.portfolio_returns.index.equals(
        result.markowitz_backtest.portfolio_returns.index
    )
    visual_paths = save_xle_experiment_visuals(result, tmp_path / "outputs")
    assert set(visual_paths) == {
        "growth_comparison",
        "markowitz_covariance_comparison",
        "commodity_covariance_comparison",
        "commodity_only_covariance_comparison",
        "histogram:PCA factor / Historical Means Sharpe",
        "histogram:PCA + U.S. commodity factors / Historical Means Sharpe",
        "histogram:U.S. commodity factors only / Historical Means Sharpe",
        "histogram:Markowitz / Historical Means Sharpe",
        "histogram:Equal-weight XLE constituents",
        "histogram:XLE baseline",
    }
    assert all(path.exists() for path in visual_paths.values())


def test_visual_diagnostics_write_comparable_covariance_and_separate_histograms(
    tmp_path,
):
    labels = ["XOM", "CVX"]
    pca_covariance = pd.DataFrame(
        [[0.02, 0.005], [0.005, 0.03]], index=labels, columns=labels
    )
    markowitz_covariance = pd.DataFrame(
        [[0.025, 0.01], [0.01, 0.035]], index=labels, columns=labels
    )
    covariance_path = tmp_path / "covariance.png"
    save_covariance_comparison(
        pca_covariance,
        markowitz_covariance,
        covariance_path,
        as_of_date="2025-01-02",
    )
    histogram_paths = save_return_histograms(
        pd.DataFrame({"PCA": [0.01, -0.02], "Markowitz": [0.005, 0.01]}),
        tmp_path / "histograms",
        bins=2,
    )

    assert covariance_path.exists()
    assert all(path.exists() for path in histogram_paths.values())
