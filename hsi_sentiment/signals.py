"""Per-ticker summary: sentiment, direction, suggested position and crowd positioning."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Exchange


def _label(x: float, hi: float = 0.15) -> str:
    return "Bullish" if x > hi else "Bearish" if x < -hi else "Neutral"


def ticker_summary(ex: Exchange, posts: pd.DataFrame, news: pd.DataFrame, fc: pd.DataFrame,
                   levels: dict[str, dict], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for a in ex.assets:
        t = a.ticker
        p = posts[posts["ticker"] == t]
        n = news[news["ticker"] == t] if not news.empty else news
        f = fc[fc["ticker"] == t]
        lv = levels.get(t)
        if lv is None or f.empty:
            continue
        w = p["weight"].sum()
        social = float((p["sentiment"] * p["weight"]).sum() / w) if w else 0.0
        news_s = float(n["sentiment"].mean()) if len(n) else 0.0
        # recent vs earlier half of the window -> is the mood improving?
        if len(p) >= 6:
            mid = p["ts_utc"].min() + (p["ts_utc"].max() - p["ts_utc"].min()) / 2
            trend = p.loc[p["ts_utc"] >= mid, "sentiment"].mean() - p.loc[p["ts_utc"] < mid, "sentiment"].mean()
        else:
            trend = 0.0
        price = lv["price"]
        pred_ret = float(f["pred_ret"].iloc[0])
        vol = max(lv["atr"] / price, 1e-4)
        model_s = math.tanh(pred_ret / vol)
        composite = 0.45 * social + 0.2 * news_s + 0.35 * model_s
        direction = _label(composite, 0.1)

        s1, r1 = lv["supports"][0], lv["resistances"][0]
        if direction == "Bullish":
            stop = s1 - 0.25 * lv["atr"]
            target = r1
            rr = (target - price) / max(price - stop, 1e-9)
            position = "LONG" if rr >= 1.0 else "WAIT (near resistance)"
        elif direction == "Bearish":
            stop = r1 + 0.25 * lv["atr"]
            target = s1
            rr = (price - target) / max(stop - price, 1e-9)
            position = "SHORT / REDUCE" if rr >= 1.0 else "WAIT (near support)"
        else:
            stop, target, rr, position = np.nan, np.nan, np.nan, "FLAT"

        bulls = int((p["sentiment"] > 0.05).sum())
        bears = int((p["sentiment"] < -0.05).sum())
        crowded = abs(social) > 0.35 and bulls + bears > 20  # one-sided, busy board -> contrarian risk
        rows.append(dict(
            exchange=ex.key, ticker=t, code=a.code, name=names.get(t) or a.name, local_name=a.local_name,
            price=price, mentions=len(p), engagement=float(p["weight"].sum()),
            sources=", ".join(f"{s}:{c}" for s, c in p["source"].value_counts().items()),
            social_sent=social, news_sent=news_s, news_n=len(n), sent_trend=float(trend if pd.notna(trend) else 0),
            bulls=bulls, bears=bears,
            crowd_long_pct=100 * float(p.loc[p["holder"], "sentiment"].ge(0).mean()) if p["holder"].any() else np.nan,
            holder_posts=int(p["holder"].sum()),
            pred_ret=pred_ret, pred_close=float(f["pred_close"].iloc[0]),
            composite=composite, direction=direction, position=position,
            entry=price, stop=stop, target=target, rr=rr,
            supports=lv["supports"], resistances=lv["resistances"], pivots=lv["pivots"],
            crowded=bool(crowded),
            divergence=("Bullish chatter vs falling price" if social > 0.2 and pred_ret < 0 else
                        "Bearish chatter vs rising price" if social < -0.2 and pred_ret > 0 else ""),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mention_share"] = df["mentions"] / max(df["mentions"].sum(), 1)
    # rank: attention x conviction
    df["rank_score"] = df["mention_share"] * (0.5 + df["social_sent"].abs())
    return df.sort_values(["mentions", "social_sent"], ascending=[False, False]).reset_index(drop=True)
