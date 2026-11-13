"""Multilingual retail-trader sentiment scoring.

Default scorer: a hand-built lexicon tuned to retail trading slang in
Cantonese (LIHKG), Korean (Naver/Daum 종목토론실) and English (news / Telegram).
Generic sentiment models miss most of this vocabulary ("坐艇", "떡상", "존버"...).

Optional: set SENTIMENT_MODEL=hf to blend in a multilingual transformer
(cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual). Requires
`uv sync --extra hf`.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

# weight: +2 strong bull, +1 bull, -1 bear, -2 strong bear
LEXICON: dict[str, float] = {
    # ---------------- Korean ----------------
    "떡상": 2, "가즈아": 2, "상한가": 2, "급등": 2, "폭등": 2, "대박": 1.5, "신고가": 1.5, "날아": 1.5,
    "매수": 1, "추매": 1, "불타기": 1, "반등": 1, "상승": 1, "호재": 1.5, "저평가": 1, "목표가 상향": 1.5,
    "줍줍": 1, "존버": 0.5, "오른다": 1, "오르": 0.7, "양봉": 1, "돌파": 1, "흑자": 1, "수주": 1, "기대": 0.7,
    "떡락": -2, "하한가": -2, "폭락": -2, "급락": -2, "개박살": -2, "나락": -2, "망했": -2, "상폐": -2,
    "매도": -1, "손절": -1.5, "물렸": -1.5, "물타기": -0.5, "하락": -1, "악재": -1.5, "고평가": -1,
    "탈출": -1, "음봉": -1, "적자": -1.5, "공매도": -1, "설거지": -1.5, "개미털기": -0.5, "떨어": -1, "내린다": -1,
    "빠진다": -1, "곡소리": -1.5, "한강": -1.5, "거품": -1, "하향": -1,
    "상향": 1, "회복": 1, "호조": 1, "견조": 0.7, "강세": 1, "약세": -1, "부진": -1, "우려": -0.7, "급증": 1,
    # ---------------- Cantonese / Chinese ----------------
    "爆升": 2, "飛天": 2, "破頂": 1.5, "新高": 1.5, "大升": 1.5, "勁升": 1.5, "抽升": 1, "升": 0.6,
    "揸貨": 1, "入貨": 1, "上車": 1, "加注": 1, "撈底": 1, "反彈": 1, "利好": 1.5, "好友": 1, "睇好": 1.2,
    "看好": 1.2, "買入": 1, "增持": 1, "跑贏": 1, "食糊": 1.5, "賺": 1, "牛": 0.8, "回購": 1, "超預期": 1.5,
    "目標價上調": 1.5, "上調": 1, "強勁": 1, "炒上": 1,
    "插水": -2, "爆煲": -2, "瀉": -1.5, "暴跌": -2, "大跌": -1.5, "跌": -0.6, "沽": -1, "放空": -1.5,
    "賣出": -1, "減持": -1, "止蝕": -1.5, "坐艇": -1.5, "接火棒": -1.5, "割韭菜": -1.5, "韭菜": -1, "淡友": -1,
    "睇淡": -1.2, "看淡": -1.2, "利淡": -1.5, "熊": -0.8, "蝕": -1, "跑輸": -1, "落車": -0.8, "下調": -1,
    "遜預期": -1.5, "炒燶": -2, "見底未": -0.5, "高位": -0.5, "唔掂": -1, "死": -1,
    # ---------------- English ----------------
    "beat": 1, "beats": 1, "upgrade": 1.5, "upgraded": 1.5, "outperform": 1.2, "buy": 0.8, "bullish": 1.5,
    "rally": 1.2, "surge": 1.5, "soar": 1.5, "record high": 1.5, "strong": 0.8, "growth": 0.6, "raise": 0.7,
    "raised": 0.7, "buyback": 1, "gain": 0.8, "gains": 0.8, "rebound": 1, "moon": 1.5, "long": 0.5,
    "miss": -1, "misses": -1, "downgrade": -1.5, "downgraded": -1.5, "underperform": -1.2, "sell": -0.8,
    "bearish": -1.5, "plunge": -1.8, "slump": -1.5, "tumble": -1.5, "drop": -0.8, "fall": -0.7, "falls": -0.7,
    "weak": -0.8, "loss": -0.8, "probe": -1, "ban": -1, "tariff": -0.6, "short": -0.5,
    "cut": -0.7, "lawsuit": -1, "sanction": -1, "crash": -2,
}

NEGATORS = ("안 ", "않", "못 ", "아니", "唔", "冇", "不", "未", "not ", "no ", "n't ", "never ")
_TERMS = sorted(LEXICON, key=len, reverse=True)
_ASCII = re.compile(r"^[a-z ]+$")


def lexicon_score(text: str) -> float:
    """Return a sentiment score in [-1, 1]. 0 means neutral / no signal."""
    if not text:
        return 0.0
    t = text.lower()
    total, used = 0.0, [False] * len(t)
    for term in _TERMS:
        pat = rf"\b{re.escape(term)}\b" if _ASCII.match(term) else re.escape(term)
        for m in re.finditer(pat, t):
            s, e = m.span()
            if any(used[s:e]):  # longest match wins ("목표가 상향" before "상향")
                continue
            used[s:e] = [True] * (e - s)
            w = LEXICON[term]
            window = t[max(0, s - 6):s]
            if any(n in window for n in NEGATORS):
                w = -0.6 * w
            total += w
    # emoji / punctuation cues common on boards
    total += 0.5 * (t.count("🚀") + t.count("📈")) - 0.5 * (t.count("📉") + t.count("😭"))
    total += 0.3 * (t.count("ㅋㅋ") > 0) * (1 if total > 0 else -1 if total < 0 else 0)
    return math.tanh(total / 2.5)


@lru_cache(maxsize=1)
def _hf_pipeline():
    from transformers import pipeline  # optional dependency

    return pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", truncation=True)


def hf_score(texts: list[str]) -> list[float]:
    pipe = _hf_pipeline()
    out = []
    for r in pipe([t[:512] or " " for t in texts], batch_size=32):
        lab = r["label"].lower()
        out.append(r["score"] if lab.startswith("pos") else -r["score"] if lab.startswith("neg") else 0.0)
    return out


def score_texts(texts: list[str]) -> list[float]:
    lex = [lexicon_score(t) for t in texts]
    if os.getenv("SENTIMENT_MODEL", "lexicon") == "hf":
        hf = hf_score(texts)
        # lexicon carries the trading slang, transformer the general tone
        return [0.6 * a + 0.4 * b for a, b in zip(lex, hf)]
    return lex
