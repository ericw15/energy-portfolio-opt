import numpy as np
from scipy import optimize
import yfinance as yf
from typing import List

def get_returns(tickers, start_date, end_date):
    # Download historical data
    data = yf.download(tickers=tickers, start=start_date, end=end_date)
    # Calculate daily percentage returns using Adjusted Close
    return data["Close"].pct_change().dropna()

class Portfolio_Strategy:
    def __init__(self, risk_free_rate:float):
        """
        risk_free_rate:float | Should be risk free rate per trading day. At 4% annualized, this might be (4 / 252).
        """
        self.r_free = risk_free_rate
        

    def optimize_towards_sharpe_ratio(self, covariance_matrix, port_expected_returns, equity_names:List[str]):
        num_equities = len(equity_names)
        portfolio_variance = lambda x: x.T @ (covariance_matrix.to_numpy() if not isinstance(covariance_matrix, np.ndarray) else covariance_matrix) @ x
        r_portfolio = lambda x: np.dot(x, port_expected_returns)
        negative_sharpe_ratio = lambda x: (r_portfolio(x) - self.r_free) / np.sqrt(portfolio_variance(x)) * -1 # we use -1 because we are minimizing
        equal_weights = np.ones(num_equities) / num_equities # initial guess is that we buy everything equally
        mins = optimize.minimize(negative_sharpe_ratio, 
                                equal_weights, 
                                bounds=[(0,1)]*num_equities,
                                constraints={'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        # TODO make this a dictionary mapping or something with the names for readability
        return mins
    
    def _get_returns(self, tickers, start_date, end_date):
        return get_returns(tickers, start_date, end_date)
    
    # Should be overridden
    def get_equity_data(self, start_date, end_date, equity_names):
        return self._get_returns(tickers=equity_names, start_date=start_date,end_date=end_date)
    
    def get_covariance_matrix(start_date, end_date, equity_data):
        raise NotImplementedError
    
    def get_expected_returns(start_date, end_date, equity_data):
        raise NotImplementedError

    def get_portfolio_weights_sharpe(self, start_date, end_date, equity_names):
        equity_data = self.get_equity_data(start_date, end_date, equity_names)
        covariance_matrix = self.get_covariance_matrix(start_date, end_date, equity_data)
        port_expected_returns = self.get_expected_returns(start_date, end_date, equity_data)

        return self.optimize_towards_sharpe_ratio(covariance_matrix, 
                                             port_expected_returns, 
                                             r_free=self.r_free, 
                                             equity_names=equity_names)
