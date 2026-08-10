from .portfolio import Portfolio_Strategy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class PCA_factor_Strategy(Portfolio_Strategy):
    def __init__(self, risk_free_rate):
        super().__init__(risk_free_rate)

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.mean()
    
    def get_covariance_matrix(self, start_date, end_date, equity_data, num_principal_components: int = 3):
        returns = equity_data
        pca = PCA(n_components=num_principal_components)
        returns_per_component = pca.fit_transform(returns.to_numpy()).T

        lr = LinearRegression()
        fit_regression_of_betas = lr.fit(returns_per_component.T,returns)
        predicted_returns = lr.predict(returns_per_component.T)
        residuals = np.var(predicted_returns - returns,axis=0,ddof=1)

        betas_np = np.array(lr.coef_)
        
        returns_per_component_dataframe = pd.DataFrame(returns_per_component.T, columns=[f"PC{i}" for i in range(1,1+num_principal_components)])

        factor_covariance = returns_per_component_dataframe.cov() # covariance
        
        asset_covariance_matrix = betas_np @ factor_covariance.to_numpy() @ betas_np.T + np.diag(v=residuals)
        return asset_covariance_matrix

def get_lightgbm_ER(equity_data, use_val_for_hyperparameter_optimization:bool = False, window_size:int = 30):
    
    model_ER_predictions = []
    for asset in equity_data.columns:
        
        X = np.lib.stride_tricks.sliding_window_view(equity_data[asset], window_shape=window_size)
        # add mean and variance as features
        mean = X.mean(axis=1, keepdims=True)
        var = X.var(axis=1, ddof=1, keepdims=True)
        X = np.hstack((X, mean, var))
        y = equity_data[asset][window_size:]

        # we predict, given the last rolling window, the mean return of the next rolling window
        y_rolling_mean =  y.iloc[::-1].rolling(window=window_size).mean().iloc[::-1].dropna() # The mean of next month
        X = X[:-window_size+1]

        last_sample = X[-1, :]
        aligned_X = X[:-1]
        train_data = lgb.Dataset(aligned_X, label=y_rolling_mean)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #train_data = lgb.Dataset(X_train, label=y_train)
        #test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }
        
        # 5. Train the model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            #valid_sets=[test_data],
            #callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        most_recent_prediction_this_asset = model.predict([last_sample])
        model_ER_predictions.append(most_recent_prediction_this_asset)
    return np.array(model_ER_predictions)


    
if __name__ == "__main__":
    start = "2024-01-01"
    end = "2026-01-01"
    tickers = ["AAPL", "MSFT", "VOO", "LLY", "JPM", "CVX"]
    pca_factor_strat = PCA_factor_Strategy(0.04 / 252)
    equity_data = pca_factor_strat.get_equity_data(start_date=start,end_date=end,equity_names=tickers)
    expected_returns_lgbm = get_lightgbm_ER(equity_data=equity_data)
    cov_matrix = pca_factor_strat.get_covariance_matrix(start_date=start, end_date=end, equity_data=equity_data)
    sns.heatmap(cov_matrix, cmap="coolwarm", xticklabels=equity_data.columns, yticklabels=equity_data.columns)
    plt.savefig("PCA_factor_covariance.png")
    portfolio = pca_factor_strat.optimize_towards_sharpe_ratio(cov_matrix, expected_returns_lgbm, equity_names=equity_data.columns)

    breakpoint()
    