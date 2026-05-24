import numpy as np
import pandas as pd

from quantlab.features.returns import (
    annualized_return,
    annualized_volatility,
    calculate_cumulative_returns,
    max_drawdown,
    sharpe_ratio,
)


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Full performance summary for a return series.

    If benchmark_returns is provided, also computes alpha, beta,
    and information ratio.
    """
    r = returns.dropna()
    cum = calculate_cumulative_returns(r)

    result: dict = {
        "total_return": float(cum.iloc[-1]) if len(cum) > 0 else float("nan"),
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(cum),
        "n_days": int(r.shape[0]),
    }

    if benchmark_returns is not None:
        b = benchmark_returns.dropna()
        aligned_r, aligned_b = r.align(b, join="inner")
        alpha = aligned_r - aligned_b

        result["alpha_annualized"] = annualized_return(alpha, periods_per_year)
        result["beta"] = (
            float(aligned_r.cov(aligned_b) / aligned_b.var())
            if aligned_b.var() != 0
            else float("nan")
        )
        result["information_ratio"] = (
            float(alpha.mean() / alpha.std() * np.sqrt(periods_per_year))
            if alpha.std() != 0
            else float("nan")
        )

    return result
