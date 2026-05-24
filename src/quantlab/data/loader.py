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

    Uses yf.download() for batch requests to avoid per-ticker rate limits.
    Results are cached per-symbol as parquet files.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    cached, missing = _split_cache(symbols, start_date, end_date, cache_dir)

    frames: list[pd.DataFrame] = list(cached.values())

    if missing:
        downloaded = _batch_download(missing, start_date, end_date, cache_dir)
        frames.extend(downloaded.values())

    if not frames:
        raise RuntimeError(
            f"No data loaded for symbols {symbols}. "
            "All downloads failed and no cache was found. "
            "Check that cache_dir points to the right location, or wait a few minutes before retrying."
        )

    combined = pd.concat(frames)
    combined = combined.reset_index().set_index(["date", "symbol"]).sort_index()
    return combined


# ── internal helpers ──────────────────────────────────────────────────────────

def _split_cache(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return (cached_frames, symbols_to_download)."""
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


def _batch_download(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path, None],
    retries: int = 3,
    backoff: float = 30.0,
) -> dict[str, pd.DataFrame]:
    """Download multiple symbols in a single yf.download() call with retry."""
    logger.info("Batch downloading %d symbol(s): %s", len(symbols), symbols)

    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            break
        except Exception as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                wait = backoff * attempt
                logger.info("Waiting %.0fs before retry…", wait)
                time.sleep(wait)
            else:
                raise

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = _extract_symbol(raw, symbol, symbols)
        if df is None or df.empty:
            logger.warning("No data returned for %s — skipping", symbol)
            continue

        df["symbol"] = symbol
        result[symbol] = df

        if cache_dir is not None:
            path = _cache_path(cache_dir, symbol, start_date, end_date)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns="symbol").to_parquet(path)
            logger.info("Cached %s → %s", symbol, path)

    return result


def _extract_symbol(
    raw: pd.DataFrame,
    symbol: str,
    all_symbols: list[str],
) -> pd.DataFrame | None:
    """Pull a single symbol out of the multi-ticker yf.download() result."""
    try:
        if len(all_symbols) == 1:
            df = raw.copy()
        else:
            df = raw[symbol].copy()
    except KeyError:
        return None

    df.index = df.index.tz_localize(None)
    df.columns = df.columns.str.lower()
    available = [c for c in _OHLCV_COLS if c in df.columns]
    df = df[available].dropna(how="all")
    df.index.name = "date"
    return df


def _cache_path(
    cache_dir: Union[str, Path],
    symbol: str,
    start_date: str,
    end_date: str,
) -> Path:
    return Path(cache_dir) / f"{symbol}_{start_date}_{end_date}.parquet"
