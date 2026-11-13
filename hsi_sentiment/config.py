"""Static configuration: exchanges, tickers, keyword lists and Apify actor IDs.

Everything that a reviewer needs to reproduce a scrape lives here, so the
actor IDs / inputs are visible in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "output"

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "10"))
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
# Hard cap on spend per actor run (USD) so a misconfigured run can't burn credits.
APIFY_MAX_CHARGE_USD = float(os.getenv("APIFY_MAX_CHARGE_USD", "1.0"))


@dataclass(frozen=True)
class Asset:
    ticker: str            # yfinance symbol, e.g. 0700.HK / 005930.KS
    code: str              # exchange-native code, e.g. 00700 / 005930
    name: str              # English name
    local_name: str        # name as retail traders write it
    keywords: tuple[str, ...]  # ticker/cashtag/keyword searches (local + EN)


@dataclass(frozen=True)
class Exchange:
    key: str
    name: str
    tz: str                # exchange timezone (used for bar alignment)
    close_hour: int        # local hour of the closing auction (posts after it roll to next session)
    sources: tuple[str, ...]
    assets: tuple[Asset, ...] = field(default_factory=tuple)


HKEX = Exchange(
    key="HKEX",
    name="Hong Kong Exchange (Hang Seng)",
    tz="Asia/Hong_Kong",
    close_hour=16,
    sources=("lihkg", "telegram"),
    assets=(
        Asset("0700.HK", "00700", "Tencent", "騰訊", ("騰訊", "腾讯", "00700", "700", "$0700", "Tencent")),
        Asset("9988.HK", "09988", "Alibaba", "阿里巴巴", ("阿里巴巴", "阿里", "09988", "9988", "$9988", "Alibaba")),
        Asset("1810.HK", "01810", "Xiaomi", "小米", ("小米", "01810", "1810", "$1810", "Xiaomi")),
        Asset("3690.HK", "03690", "Meituan", "美團", ("美團", "美团", "03690", "3690", "$3690", "Meituan")),
        Asset("0005.HK", "00005", "HSBC", "匯豐", ("匯豐", "汇丰", "滙豐", "00005", "$0005", "HSBC")),
    ),
)

KRX = Exchange(
    key="KRX",
    name="Korea Exchange (KOSPI)",
    tz="Asia/Seoul",
    close_hour=15,  # 15:30 close; posts at/after 15:00 are treated as close-session
    sources=("naver", "daum"),
    assets=(
        Asset("005930.KS", "005930", "Samsung Electronics", "삼성전자", ("삼성전자", "삼전", "005930", "Samsung Electronics")),
        Asset("000660.KS", "000660", "SK Hynix", "SK하이닉스", ("SK하이닉스", "하이닉스", "하닉", "000660", "SK Hynix")),
        Asset("035420.KS", "035420", "NAVER", "NAVER", ("네이버", "NAVER", "035420")),
        Asset("005380.KS", "005380", "Hyundai Motor", "현대차", ("현대차", "현대자동차", "005380", "Hyundai Motor")),
        Asset("035720.KS", "035720", "Kakao", "카카오", ("카카오", "035720", "Kakao")),
    ),
)

EXCHANGES: dict[str, Exchange] = {e.key: e for e in (HKEX, KRX)}

# ---------------------------------------------------------------------------
# Apify actors (public Store actors; IDs pinned for reproducibility)
# ---------------------------------------------------------------------------
ACTORS = {
    # LIHKG 連登 keyword search -> thread rows with likes / reply counts
    "lihkg": os.getenv("APIFY_ACTOR_LIHKG", "claude_code_reviewer/lihkg-scraper-en"),   # id PJDFRxJdpPknN9nYn
    # Public Telegram channels (t.me/s preview) -> messages with views + channel subscriber count
    "telegram": os.getenv("APIFY_ACTOR_TELEGRAM", "automation-lab/telegram-scraper"),  # id wxRoaBWQSXNzpVH1D
    # Naver 종목토론실 -> posts with views / likes / dislikes / holder-verified flag
    "naver": os.getenv("APIFY_ACTOR_NAVER", "hgservices/naver-stock-discussion-scraper"),  # id xXFbG2oX6UpaPtcTC
    # Daum 토론방 -> official Apify generic scraper hitting Daum's public JSON endpoint
    "daum": os.getenv("APIFY_ACTOR_DAUM", "apify/cheerio-scraper"),
}

# Public HK finance Telegram channels to scan (messages filtered by the asset keywords).
TELEGRAM_CHANNELS = tuple(
    c.strip() for c in os.getenv(
        "TELEGRAM_CHANNELS", "aastocks_news,hkej_news,wallstreetcn,hkstockcn,cls_telegraph"
    ).split(",") if c.strip()
)
