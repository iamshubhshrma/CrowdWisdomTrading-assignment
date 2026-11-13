"""Align posts/news with price bars and build the daily feature panel.

Alignment rule (no look-ahead):
  A post belongs to trading session d if it was written after the close of d-1
  and before the close of d (exchange local time). Feature row d is therefore
  fully known at the close of d, and is used to predict the close of d+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Exchange
from .sentiment import score_texts

HOLD_WORDS = ("보유", "존버", "물렸", "평단", "들고", "홀딩", "揸住", "坐艇", "持有", "仲揸", "貨尾", "holding", "my position")


def engagement_weight(df: pd.DataFrame) -> pd.Series:
    eng = df["likes"] + df["dislikes"] + 2 * df["replies"] + df["views"] / 100
    return 1 + np.log1p(eng) + 0.5 * np.log1p(df["author_followers"])


def session_dates(ts_utc: pd.Series, ex: Exchange, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map UTC timestamps to the trading session whose close they precede."""
    local = ts_utc.dt.tz_convert(ex.tz)
    day = local.dt.tz_localize(None).dt.normalize()
    day = day.where(local.dt.hour < ex.close_hour, day + pd.Timedelta(days=1))
    days = trading_days.sort_values()
    # posts after the last known close belong to the next (future) session
    idx = days.searchsorted(day.values, side="left")
    future = pd.bdate_range(days[-1] + pd.Timedelta(days=1), periods=1)[0] if len(days) else pd.NaT
    mapped = [days[i] if i < len(days) else future for i in idx]
    return pd.Series(mapped, index=ts_utc.index)


def align_hourly(posts: pd.DataFrame, hourly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach the hourly price bar (bar open time + close) each post falls into or follows."""
    out = []
    for t, g in posts.groupby("ticker"):
        g = g.sort_values("ts_utc")
        h = hourly.get(t)
        if h is None or h.empty:
            out.append(g.assign(bar_ts=pd.NaT, bar_close=np.nan))
            continue
        bars = h[["close"]].rename(columns={"close": "bar_close"}).reset_index(names="bar_ts")
        bars["bar_ts"] = bars["bar_ts"].astype(g["ts_utc"].dtype)
        out.append(pd.merge_asof(g, bars, left_on="ts_utc", right_on="bar_ts", direction="backward"))
    return pd.concat(out, ignore_index=True) if out else posts


def score_posts(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts["sentiment"] = score_texts(posts["text"].fillna("").tolist())
    posts["weight"] = engagement_weight(posts)
    low = posts["text"].fillna("").str.lower()
    posts["holder"] = low.apply(lambda s: any(w in s for w in HOLD_WORDS))
    return posts


def daily_social(posts: pd.DataFrame, ex: Exchange, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    p = posts.copy()
    p["date"] = session_dates(p["ts_utc"], ex, trading_days)
    p["ws"] = p["sentiment"] * p["weight"]
    g = p.groupby(["ticker", "date"])
    d = pd.DataFrame({
        "mentions": g.size(),
        "sent_w": g["ws"].sum() / g["weight"].sum(),
        "bull_share": g["sentiment"].apply(lambda s: (s > 0.05).sum() / max((s.abs() > 0.05).sum(), 1)),
        "engagement": g["weight"].sum(),
    })
    for src, gs in p.groupby("source"):
        d[f"sent_{src}"] = gs.groupby(["ticker", "date"]).apply(lambda x: x["ws"].sum() / x["weight"].sum())
        d[f"n_{src}"] = gs.groupby(["ticker", "date"]).size()
    return d.reset_index()


def daily_news(news: pd.DataFrame, ex: Exchange, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["ticker", "date", "news_sent", "news_n"])
    n = news.copy()
    n["date"] = session_dates(n["ts_utc"], ex, trading_days)
    return (n.groupby(["ticker", "date"])
             .agg(news_sent=("sentiment", "mean"), news_n=("sentiment", "size")).reset_index())


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def price_features(daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in daily.items():
        c = df["close"]
        f = pd.DataFrame(index=df.index)
        f["ticker"] = t
        f["close"] = c
        f["ret_1"] = np.log(c).diff()
        f["ret_5"] = np.log(c).diff(5)
        f["vol_10"] = f["ret_1"].rolling(10).std()
        f["rsi_14"] = (rsi(c) - 50) / 50
        f["dist_sma20"] = c / c.rolling(20).mean() - 1
        f["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
        f["target"] = np.log(c).shift(-1) - np.log(c)  # next-session log return
        rows.append(f.reset_index(names="date"))
    return pd.concat(rows, ignore_index=True)


SOCIAL_COLS = ["mentions_z", "sent_w", "bull_share", "news_sent", "sent_mom"]
PRICE_COLS = ["ret_1", "ret_5", "vol_10", "rsi_14", "dist_sma20", "vol_z"]


def build_panel(price_feat: pd.DataFrame, social: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    panel = price_feat.merge(social, on=["ticker", "date"], how="left").merge(news, on=["ticker", "date"], how="left")
    panel["mentions"] = panel["mentions"].fillna(0)
    for c in ("sent_w", "bull_share", "news_sent", "news_n", "engagement"):
        panel[c] = panel[c].fillna(0.0 if c != "bull_share" else 0.5)
    # mention surprise vs the ticker's own average inside the window (scale-free across tickers)
    has = panel["date"] >= social["date"].min() if not social.empty else panel["date"] < panel["date"].min()
    stats = panel[has].groupby("ticker")["mentions"].agg(["mean", "std"])
    panel = panel.join(stats, on="ticker")
    panel["mentions_z"] = ((panel["mentions"] - panel["mean"]) / panel["std"].replace(0, 1)).fillna(0)
    panel["sent_mom"] = panel.groupby("ticker")["sent_w"].diff().fillna(0)
    panel["in_window"] = has
    return panel.drop(columns=["mean", "std"]).sort_values(["date", "ticker"]).reset_index(drop=True)
