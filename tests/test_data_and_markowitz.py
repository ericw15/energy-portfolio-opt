from port_opt.strategy.portfolio import get_returns


def test_data_import():
    get_returns("VOO", start_date="2024-01-01", end_date="2026-01-01")
