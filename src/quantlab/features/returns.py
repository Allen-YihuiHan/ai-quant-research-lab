import numpy as np
import pandas as pd


def calculate_returns(price: pd.Series) -> pd.Series:
    return price.pct_change()


def calculate_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    total = (1 + r).prod()
    n = len(r)
    if n == 0:
        return float("nan")
    return float(total ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown from peak. Returns a negative number (e.g. -0.30 = -30%)."""
    wealth = 1 + cumulative_returns
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return float(drawdown.min())
