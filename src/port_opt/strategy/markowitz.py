from .covariance import ewma_covariance, ledoit_wolf_covariance
from .portfolio import Portfolio_Strategy


class Markowitz_Portfolio(Portfolio_Strategy):
    def __init__(self, risk_free_rate):
        super().__init__(risk_free_rate)

    def get_expected_returns(self, start_date, end_date, equity_data):
        return equity_data.mean()

    def get_covariance_matrix(self, start_date, end_date, equity_data):
        return equity_data.cov()


class Rolling_Markowitz_Portfolio(Markowitz_Portfolio):
    def __init__(self, risk_free_rate: float, rolling_days: int):
        super().__init__(risk_free_rate)
        self.rolling_days = rolling_days

    def get_expected_returns(self, start_date, end_date, equity_data):
        if len(equity_data) < self.rolling_days:
            raise ValueError("equity_data must contain rolling_days observations")
        return equity_data.iloc[-self.rolling_days :].mean()


class Ledoit_Wolf_Portfolio(Markowitz_Portfolio):
    """Historical-mean Markowitz portfolio with Ledoit--Wolf covariance."""

    def get_covariance_matrix(self, start_date, end_date, equity_data):
        return ledoit_wolf_covariance(equity_data)


class EWMA_Portfolio(Markowitz_Portfolio):
    """Historical-mean Markowitz portfolio with EWMA covariance."""

    def __init__(self, risk_free_rate: float, half_life: int = 63):
        super().__init__(risk_free_rate)
        if half_life < 1:
            raise ValueError("half_life must be positive")
        self.half_life = half_life

    def get_covariance_matrix(self, start_date, end_date, equity_data):
        return ewma_covariance(equity_data, self.half_life)
