import pandas as pd
import pytest

from port_opt.backtest import summarize_performance


def test_performance_summary_reports_empirical_risk_and_return_metrics():
    daily_returns = pd.DataFrame(
        {"Strategy": [0.01, -0.02, 0.03, -0.01]},
        index=pd.date_range("2025-01-02", periods=4, freq="B"),
    )

    metrics = summarize_performance(
        daily_returns, periods_per_year=4, tail_probability=0.25
    )

    assert metrics.loc["Strategy", "observations"] == 4
    assert metrics.loc["Strategy", "cumulative_return"] == pytest.approx(
        (1.01 * 0.98 * 1.03 * 0.99) - 1.0
    )
    assert metrics.loc["Strategy", "mean_daily_return"] == pytest.approx(0.0025)
    assert metrics.loc["Strategy", "annualized_arithmetic_return"] == pytest.approx(
        0.01
    )
    assert metrics.loc["Strategy", "maximum_drawdown"] == pytest.approx(-0.02)
    assert metrics.loc["Strategy", "worst_daily_return"] == pytest.approx(-0.02)
    assert metrics.loc["Strategy", "tail_return_quantile_25pct"] == pytest.approx(
        -0.0125
    )
    assert metrics.loc["Strategy", "tail_expected_shortfall_25pct"] == pytest.approx(
        -0.02
    )
