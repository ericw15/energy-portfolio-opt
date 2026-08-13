import pandas as pd
import pytest

from port_opt.backtest.statistics import (
    paired_hac_return_test,
    run_pre_specified_return_comparisons,
)


def test_paired_hac_test_reports_active_return_and_hac_interval():
    index = pd.date_range("2025-01-02", periods=8, freq="B")
    strategy = pd.Series([0.02] * 8, index=index)
    benchmark = pd.Series([0.01] * 8, index=index)

    result = paired_hac_return_test(strategy, benchmark, hac_lag=1)

    assert result["mean_daily_active_return"] == pytest.approx(0.01)
    assert result["annualized_active_return"] == pytest.approx(2.52)
    assert result["hac_lag"] == 1
    assert result["two_sided_p_value"] == 0.0


def test_pre_specified_comparisons_only_test_requested_columns():
    returns = pd.DataFrame(
        {
            "candidate": [0.01, 0.02, 0.01],
            "base": [0.0, 0.01, 0.0],
            "other": [0.2, 0.2, 0.2],
        }
    )

    result = run_pre_specified_return_comparisons(
        returns, {"candidate versus base": ("candidate", "base")}, hac_lag=1
    )

    assert result.index.tolist() == ["candidate versus base"]
    assert result.loc["candidate versus base", "strategy"] == "candidate"
    assert result.loc["candidate versus base", "benchmark"] == "base"
