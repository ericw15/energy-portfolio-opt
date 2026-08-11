from port_opt.strategy import get_returns
from port_opt.strategy.PCA_factor import PCA_factor_Strategy, get_lightgbm_ER
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

XLE_TICKERS = [
    "XOM",
    "CVX",
    "COP",
    "PSX",
    "MPC",
    "VLO",
    "SLB",
    "EOG",
    "WMB",
    "BKR",
    "KMI",
    "OXY",
    "HAL",
    "FANG",
    "DVN",
    "TRGP",
    "OKE",
    "APA",
    "NOV",
]


def get_PCA_factor_portfolio_returns():
    start = "2024-01-01"
    end = "2026-01-01"
    tickers = ["AAPL", "MSFT", "VOO", "LLY", "JPM", "CVX"]
    pca_factor_strat = PCA_factor_Strategy(0.04 / 252)
    equity_data = pca_factor_strat.get_equity_data(
        start_date=start, end_date=end, equity_names=tickers
    )
    expected_returns_lgbm = get_lightgbm_ER(equity_data=equity_data)
    cov_matrix = pca_factor_strat.get_covariance_matrix(
        start_date=start, end_date=end, equity_data=equity_data
    )
    sns.heatmap(
        cov_matrix,
        cmap="coolwarm",
        xticklabels=equity_data.columns,
        yticklabels=equity_data.columns,
    )
    plt.savefig("PCA_factor_covariance.png")
    portfolio = pca_factor_strat.optimize_towards_sharpe_ratio(
        cov_matrix, expected_returns_lgbm, equity_names=equity_data.columns
    )


def main():

    baseline = "XLE"
    start_training_data = datetime(2023, 1, 1)
    start = datetime(2024, 1, 1)
    end = datetime(2026, 1, 1)

    baseline_returns = get_returns(tickers=baseline, start_date=start, end_date=end) + 1
    baseline_portfolio_returns = np.cumprod(baseline_returns)

    equal_weighting = np.full(
        (baseline_portfolio_returns.size, len(XLE_TICKERS)),
        fill_value=1 / len(XLE_TICKERS),
    )
    xle_individual_asset_returns = get_returns(
        tickers=XLE_TICKERS, start_date=start, end_date=end
    )
    equal_weighting_returns_indiv = (
        np.sum(xle_individual_asset_returns * equal_weighting, axis=1) + 1
    )
    # TODO is this +1 in the right place?
    equal_weighting_returns = np.cumprod(equal_weighting_returns_indiv)

    pca_portfolio_balancing = pd.DataFrame(
        np.full(equal_weighting.shape, np.nan), index=equal_weighting_returns.index
    )

    days_step = 30
    step = timedelta(days=days_step)
    lgbm_lookback = 30  # this differs because it numbers trading days, not just days
    current = start
    previous_portfolio_performance = pd.Series([])
    while current < end:
        # This is difficult for two reasons.
        # 1. trading days differ from calendar, so we have to know what days our strategy is in effect for.
        # 2. our window size acts on array elements but the timedelta may be different. Somehow these would have to communicate.
        to_be_replaced = (pca_portfolio_balancing.index >= current) & (
            pca_portfolio_balancing.index < current + step
        )  # only get trading days in that period

        current_equity_data = get_returns(
            tickers=XLE_TICKERS, start_date=start_training_data, end_date=current
        )
        longer_equity_data = get_returns(
            tickers=XLE_TICKERS, start_date=start_training_data, end_date=current
        )
        pca_factor_strat = PCA_factor_Strategy(risk_free_rate=0.04 / 252)
        expected_returns_lgbm = get_lightgbm_ER(
            equity_data=longer_equity_data, window_size=lgbm_lookback
        )
        cov_matrix = pca_factor_strat.get_covariance_matrix(
            start_date=start, end_date=current, equity_data=current_equity_data
        )
        # sns.heatmap(cov_matrix, cmap="coolwarm", xticklabels=current_equity_data.columns, yticklabels=equity_data.columns)
        # plt.savefig("PCA_factor_covariance.png")
        portfolio = pca_factor_strat.optimize_towards_sharpe_ratio(
            cov_matrix, expected_returns_lgbm, equity_names=current_equity_data.columns
        )
        pca_portfolio_balancing[to_be_replaced] = np.tile(
            portfolio.x, (np.sum(to_be_replaced), 1)
        )

        if current + step >= end:
            current = end
        else:
            current += step

    pca_portfolio_balancing.columns = xle_individual_asset_returns.columns
    period_performance = np.sum(
        xle_individual_asset_returns * pca_portfolio_balancing, axis=1
    )
    overall_performance = period_performance + 1
    overall_pca_factor_portfolio_returns = np.cumprod(overall_performance)

    portfolios_return_by_day = {}
    portfolios_return_by_day["Equal Weighting, Equity"] = (
        equal_weighting_returns.to_numpy().flatten()
    )
    portfolios_return_by_day["Baseline (XLE)"] = (
        baseline_portfolio_returns.to_numpy().flatten()
    )
    portfolios_return_by_day["PCA Factor with LightGBM"] = (
        overall_pca_factor_portfolio_returns.to_numpy().flatten()
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=pd.DataFrame(
            portfolios_return_by_day, index=baseline_portfolio_returns.index
        )
    )
    plt.savefig("return_comparison.png")


if __name__ == "__main__":
    main()
