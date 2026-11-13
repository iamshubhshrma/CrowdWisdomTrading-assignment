"""Price bars via yfinance (daily history for the model, hourly bars for post alignment)."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.get_level_values(1) else df.droplevel(1, axis=1)
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def daily_bars(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = yf.download(t, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            log.warning("no daily data for %s", t)
            continue
        df = _flatten(df, t)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[t] = df
    return out


def hourly_bars(tickers: list[str], days: int = 10) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = yf.download(t, period=f"{min(days + 5, 59)}d", interval="60m", progress=False, auto_adjust=True)
        if df.empty:
            continue
        df = _flatten(df, t)
        df.index = pd.to_datetime(df.index).tz_convert("UTC")
        out[t] = df
    return out


def asset_name(ticker: str) -> str | None:
    try:
        return yf.Ticker(ticker).info.get("longName")
    except Exception:  # yfinance info endpoint is flaky; names are in config anyway
        return None
