import pandas as pd


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Price momentum: close / close.shift(window) - 1. No lookahead."""
    return close / close.shift(window) - 1


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def moving_average_ratio(close: pd.Series, short: int, long: int) -> pd.Series:
    return moving_average(close, short) / moving_average(close, long) - 1


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Position within Bollinger Bands: 0 = lower band, 1 = upper band."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    band_width = (upper - lower).replace(0, float("nan"))
    return (close - lower) / band_width


def volume_change(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to rolling mean."""
    return volume / volume.rolling(window).mean() - 1


def add_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical factors in-place on a single-symbol OHLCV DataFrame.

    Input must have columns: close, volume (optional).
    All features use only past data — no lookahead.
    """
    df = df.copy()
    close = df["close"]
    returns = close.pct_change()

    df["momentum_5"] = momentum(close, 5)
    df["momentum_20"] = momentum(close, 20)
    df["momentum_60"] = momentum(close, 60)
    df["volatility_20"] = rolling_volatility(returns, 20)
    df["moving_average_20"] = moving_average(close, 20)
    df["moving_average_60"] = moving_average(close, 60)
    df["moving_average_ratio_5_20"] = moving_average_ratio(close, 5, 20)
    df["moving_average_ratio_20_60"] = moving_average_ratio(close, 20, 60)
    df["rsi_14"] = rsi(close, 14)
    df["macd"], df["macd_signal"] = macd(close)
    df["bollinger_position"] = bollinger_position(close)

    if "volume" in df.columns:
        df["volume_change_20"] = volume_change(df["volume"], 20)

    return df
