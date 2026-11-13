import numpy as np
import pandas as pd

from hsi_sentiment import config, features, levels, sources
from hsi_sentiment.sentiment import lexicon_score


def test_lexicon_directions():
    assert lexicon_score("삼성전자 가즈아 떡상 🚀") > 0.3
    assert lexicon_score("하이닉스 물렸다 손절") < -0.3
    assert lexicon_score("騰訊今日爆升，揸貨") > 0.3
    assert lexicon_score("阿里又插水，坐艇中") < -0.3
    assert lexicon_score("Tencent shares surge after earnings beat") > 0.3
    assert lexicon_score("今日天氣好好") == 0.0


def test_negation_flips():
    assert lexicon_score("唔睇好") < 0
    assert lexicon_score("not bullish") < 0


def test_longest_match_wins():
    # "大跌" must not also count the embedded "跌"
    assert abs(lexicon_score("大跌") - np.tanh(-1.5 / 2.5)) < 1e-9


def test_session_alignment_no_lookahead():
    ex = config.KRX  # close 15:00 KST == 06:00 UTC
    days = pd.DatetimeIndex(["2026-11-09", "2026-11-10", "2026-11-11"])
    ts = pd.Series(pd.to_datetime([
        "2026-11-10T05:00Z",  # 14:00 KST on the 10th -> session 10th
        "2026-11-10T07:00Z",  # 16:00 KST on the 10th, after close -> session 11th
        "2026-11-08T03:00Z",  # Sunday -> next trading day (9th)
    ], utc=True))
    got = features.session_dates(ts, ex, days).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2026-11-10", "2026-11-11", "2026-11-09"]


def test_match_asset_numeric_tokens():
    a = sources.match_asset("17000 點，700 今日升", config.HKEX.assets)
    assert a and a.ticker == "0700.HK"
    assert sources.match_asset("恒指 17000", config.HKEX.assets) is None


def test_normalizers_map_to_schema():
    naver = sources.naver_normalize([{"type": "post", "stockCode": "005930", "postId": "1", "title": "떡상",
                                      "text": "가즈아", "writtenAt": "2026-11-10T10:00:00+09:00", "views": 5,
                                      "likes": 2, "isHolderVerified": True}], config.KRX)
    daum = sources.daum_normalize([{"id": 9, "code": "000660", "title": "하락", "content": "",
                                    "createdAt": "2026-11-10 10:00:00", "agreeCount": 1}], config.KRX)
    df = sources.to_frame(naver + daum)
    assert list(df.columns) == sources.POST_COLUMNS
    assert set(df["ticker"]) == {"005930.KS", "000660.KS"}
    assert (df["ts_utc"].dt.hour == 1).all()  # 10:00 KST -> 01:00 UTC


def test_support_below_resistance_above():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(1, 100, 200)},
                      index=pd.bdate_range("2026-01-01", periods=200))
    lv = levels.support_resistance(df)
    assert all(s < lv["price"] for s in lv["supports"])
    assert all(r > lv["price"] for r in lv["resistances"])
