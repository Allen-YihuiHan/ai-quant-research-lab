import numpy as np
import pandas as pd
import pytest

from quantlab.features.returns import (
    annualized_return,
    annualized_volatility,
    calculate_cumulative_returns,
    calculate_returns,
    max_drawdown,
    sharpe_ratio,
)


# ── returns ──────────────────────────────────────────────────────────────────

def test_calculate_returns_basic():
    prices = pd.Series([100.0, 110.0, 99.0])
    r = calculate_returns(prices)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.1, rel=1e-6)
    assert r.iloc[2] == pytest.approx((99 - 110) / 110, rel=1e-6)


def test_calculate_returns_length():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert len(calculate_returns(prices)) == 4


# ── cumulative returns ────────────────────────────────────────────────────────

def test_cumulative_returns_known():
    r = pd.Series([0.1, -0.1, 0.2])
    cum = calculate_cumulative_returns(r)
    expected = 1.1 * 0.9 * 1.2 - 1
    assert cum.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_cumulative_returns_zero():
    r = pd.Series([0.0, 0.0, 0.0])
    cum = calculate_cumulative_returns(r)
    assert (cum == 0).all()


# ── max drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_no_drawdown():
    cum = pd.Series([0.0, 0.1, 0.2, 0.3])
    assert max_drawdown(cum) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known_50pct():
    # 100 → 150 → 75: drawdown from peak 150 to 75 = -50%
    prices = pd.Series([100.0, 150.0, 75.0])
    r = calculate_returns(prices).dropna()
    cum = calculate_cumulative_returns(r)
    assert max_drawdown(cum) == pytest.approx(-0.5, rel=1e-6)


def test_max_drawdown_recovery():
    # Peak at 1.2, then drops to 0.95
    cum = pd.Series([0.0, 0.1, 0.2, 0.05, -0.05])
    expected = (0.95 - 1.2) / 1.2
    assert max_drawdown(cum) == pytest.approx(expected, rel=1e-6)


def test_max_drawdown_single_element():
    cum = pd.Series([0.05])
    assert max_drawdown(cum) == pytest.approx(0.0, abs=1e-9)


# ── annualized return ─────────────────────────────────────────────────────────

def test_annualized_return_constant():
    r = pd.Series([0.01] * 252)
    assert annualized_return(r) == pytest.approx(1.01**252 - 1, rel=1e-6)


def test_annualized_return_empty():
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))


# ── sharpe ratio ──────────────────────────────────────────────────────────────

def test_sharpe_zero_volatility():
    r = pd.Series([0.01] * 20)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0.001, 0.01, 252))
    assert sharpe_ratio(r) > 0


# ── annualized volatility ─────────────────────────────────────────────────────

def test_annualized_volatility():
    np.random.seed(42)
    daily_vol = 0.01
    r = pd.Series(np.random.normal(0, daily_vol, 500))
    av = annualized_volatility(r)
    assert av == pytest.approx(daily_vol * np.sqrt(252), rel=0.05)
