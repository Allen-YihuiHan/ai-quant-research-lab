from pathlib import Path
from typing import Union

import pandas as pd
import yfinance as yf

from quantlab.utils.logger import get_logger

logger = get_logger(__name__)


def load_ohlcv(
    symbols: Union[str, list[str]],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Download OHLCV daily data for one or more symbols.

    Returns a MultiIndex DataFrame indexed by (date, symbol).
    Columns: open, high, low, close, volume
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = _load_single(symbol, start_date, end_date, cache_dir)
        df["symbol"] = symbol
        frames.append(df)

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["date", "symbol"]).sort_index()
    return combined


def _load_single(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None],
) -> pd.DataFrame:
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{symbol}_{start_date}_{end_date}.parquet"
        if cache_path.exists():
            logger.info("Cache hit: %s", symbol)
            return pd.read_parquet(cache_path)

    logger.info("Downloading %s from yfinance", symbol)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    df.index = df.index.tz_localize(None)
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached %s → %s", symbol, cache_path)

    return df
