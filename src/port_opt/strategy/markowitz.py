from portfolio import Portfolio_Strategy

class Markowitz_Portfolio(Portfolio_Strategy):
    def __init__(self, risk_free_rate):
        super().__init__(risk_free_rate)

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.mean()
    
    def get_covariance_matrix(self, start_date, end_date, equity_data):
        return equity_data.cov()
    
class Rolling_Markowitz_Portfolio(Markowitz_Portfolio):
    def __init__(self, risk_free_rate:float, rolling_days:int):
        super().__init__(risk_free_rate)
        self.rolling_days = rolling_days

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.rolling(self.rolling_days).mean()