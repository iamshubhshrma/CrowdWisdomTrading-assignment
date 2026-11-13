"""Synthetic posts/news for an offline end-to-end run (`--demo`).

Prices are REAL (yfinance); only the social/news text is generated. Posts are
loosely correlated with the same-day return so the pipeline has something to
learn, which means demo model scores are NOT evidence of real predictive power.
Every demo report is stamped "DEMO DATA".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Exchange
from .sources import to_frame

BULL = {
    "HKEX": ["{n} 今日爆升，揸貨等破頂", "{n} 業績超預期，睇好", "上車 {n}，反彈有排升 🚀", "{n} 回購利好，加注"],
    "KRX": ["{n} 가즈아 떡상 간다 🚀", "{n} 호재 떴다 매수 추천", "{n} 저평가 줍줍 타이밍", "{n} 반등 시작 신고가 기대"],
}
BEAR = {
    "HKEX": ["{n} 又插水，坐艇中", "{n} 睇淡，沽咗先", "{n} 遜預期，止蝕走人", "割韭菜 {n} 接火棒"],
    "KRX": ["{n} 떡락 손절했다", "{n} 물렸다 하락 무섭네", "{n} 악재 매도 탈출", "{n} 설거지 당했다 폭락"],
}
NEUTRAL = {"HKEX": ["{n} 今日點睇？", "有冇人揸住 {n}", "{n} 成交量點解咁少"],
           "KRX": ["{n} 내일 어떨까요", "{n} 보유자분들 계신가요", "{n} 거래량 보세요"]}
NEWS = {1: "{n} shares surge after earnings beat; analysts upgrade", -1: "{n} shares fall as weak guidance prompts downgrade",
        0: "{n} trades flat ahead of results"}


def generate(ex: Exchange, daily: dict[str, pd.DataFrame], days: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    posts, news = [], []
    for a in ex.assets:
        df = daily.get(a.ticker)
        if df is None:
            continue
        rets = np.log(df["close"]).diff().fillna(0)
        base = rng.integers(8, 40)
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            dkey = day.tz_localize(None)
            # mood leans on today's AND (weakly) next session's move
            nxt = rets[rets.index > dkey]
            today = rets[rets.index <= dkey]
            mood = 25 * (today.iloc[-1] if len(today) else 0) + 10 * (nxt.iloc[0] if len(nxt) else 0)
            p_bull = 1 / (1 + np.exp(-mood)) * 0.8 + 0.1
            n = int(rng.poisson(base * (1 + 3 * abs(mood))))
            for i in range(n):
                ts = day + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
                if ts > end:
                    continue
                r = rng.random()
                pool = BULL if r < p_bull * 0.8 else NEUTRAL if r < p_bull * 0.8 + 0.2 else BEAR
                tmpl = pool[ex.key][rng.integers(len(pool[ex.key]))]
                posts.append(dict(
                    post_id=f"demo:{a.ticker}:{ts.value}:{i}", source=ex.sources[i % 2], exchange=ex.key,
                    ticker=a.ticker, ts_utc=ts, text=tmpl.format(n=a.local_name),
                    likes=float(rng.poisson(3)), dislikes=float(rng.poisson(1)), replies=float(rng.poisson(2)),
                    views=float(rng.poisson(150)), author=f"user{rng.integers(1000)}",
                    author_followers=float(rng.poisson(20)), url="",
                ))
            if rng.random() < 0.6:
                k = int(np.sign(mood) if abs(mood) > 0.2 else 0)
                news.append(dict(exchange=ex.key, ticker=a.ticker, ts_utc=day + pd.Timedelta(hours=2),
                                 title=NEWS[k].format(n=a.name), url="", text=NEWS[k].format(n=a.name)))
    news_df = pd.DataFrame(news)
    if not news_df.empty:
        news_df["ts_utc"] = pd.to_datetime(news_df["ts_utc"], utc=True)
    return to_frame(posts, days), news_df
