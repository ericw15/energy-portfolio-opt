from portfolio import Portfolio_Strategy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

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

    
