import time
from pathlib import Path
from typing import Union

import pandas as pd
import yfinance as yf

from quantlab.utils.logger import get_logger

logger = get_logger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(
    symbols: Union[str, list[str]],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Download OHLCV daily data for one or more symbols.

    Returns a MultiIndex DataFrame indexed by (date, symbol).
    Columns: open, high, low, close, volume

    Uses Ticker.history() per symbol (more reliable than yf.download()).
    Results are cached as parquet files.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    cached, missing = _split_cache(symbols, start_date, end_date, cache_dir)
    frames: list[pd.DataFrame] = list(cached.values())

    for symbol in missing:
        df = _download_one(symbol, start_date, end_date, retries=3, backoff=15.0)
        if df is None or df.empty:
            logger.warning("No data returned for %s — skipping", symbol)
            continue
        df["symbol"] = symbol
        _save_cache(df.drop(columns="symbol"), cache_dir, symbol, start_date, end_date)
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No data loaded for symbols {symbols}. "
            "All downloads failed and no cache was found."
        )

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["date", "symbol"]).sort_index()
    return combined


# ── internal helpers ──────────────────────────────────────────────────────────

def _download_one(
    symbol: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
    backoff: float = 15.0,
) -> pd.DataFrame | None:
    """Download a single symbol via Ticker.history() with retry."""
    logger.info("Downloading %s", symbol)
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(symbol).history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
            )
            if df.empty:
                raise RuntimeError("empty response")
            df.index = df.index.tz_localize(None)
            df.columns = df.columns.str.lower()
            available = [c for c in _OHLCV_COLS if c in df.columns]
            df = df[available].dropna(how="all")
            df.index.name = "date"
            return df
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, symbol, exc)
            if attempt < retries:
                wait = backoff * attempt
                logger.info("Waiting %.0fs before retry…", wait)
                time.sleep(wait)
    return None


def _split_cache(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        if cache_dir is not None:
            path = _cache_path(cache_dir, symbol, start_date, end_date)
            if path.exists():
                logger.info("Cache hit: %s", symbol)
                df = pd.read_parquet(path)
                df["symbol"] = symbol
                cached[symbol] = df
                continue
        missing.append(symbol)
    return cached, missing


def _save_cache(
    df: pd.DataFrame,
    cache_dir: Union[str, Path, None],
    symbol: str,
    start_date: str,
    end_date: str,
) -> None:
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, symbol, start_date, end_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %s → %s", symbol, path)


def _cache_path(
    cache_dir: Union[str, Path],
    symbol: str,
    start_date: str,
    end_date: str,
) -> Path:
    return Path(cache_dir) / f"{symbol}_{start_date}_{end_date}.parquet"
