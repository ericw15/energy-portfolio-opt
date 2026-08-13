import numpy as np
import pandas as pd

from port_opt.backtest.experiment_core import (
    assemble_experiment_outputs,
    run_labelled_strategies,
    save_standard_summary_products,
)


def test_shared_experiment_core_aligns_strategies_and_writes_standard_products(
    tmp_path,
):
    returns = pd.DataFrame(
        {
            "XOM": [0.01, -0.02, 0.01, 0.03, -0.01, 0.02],
            "CVX": [0.00, 0.01, -0.01, 0.02, 0.00, 0.01],
        },
        index=pd.date_range("2025-01-02", periods=6, freq="B"),
    )
    backtests = run_labelled_strategies(
        returns,
        {
            "A": lambda _train: pd.Series({"XOM": 0.5, "CVX": 0.5}),
            "B": lambda _train: pd.Series({"XOM": 1.0, "CVX": 0.0}),
        },
        lookback_periods=2,
        rebalance_frequency=2,
    )
    outputs = assemble_experiment_outputs(
        backtests, returns, returns["XOM"], risk_free_rate=0.0
    )

    assert outputs.daily_returns.columns.tolist() == [
        "A",
        "B",
        "Equal-weight XLE constituents",
        "XLE baseline",
    ]
    paths = save_standard_summary_products(outputs, tmp_path / "outputs")
    assert set(paths) == {
        "performance_metrics",
        "implementation_metrics",
        "risk_return_comparison",
        "implementation_comparison",
    }
    assert all(path.exists() for path in paths.values())
