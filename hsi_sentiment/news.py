"""Recent news per asset via Exa (exa.ai) neural search."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from .config import Exchange

log = logging.getLogger(__name__)

NEWS_COLUMNS = ["exchange", "ticker", "ts_utc", "title", "url", "text"]


def _query(asset, ex: Exchange) -> str:
    return f"{asset.name} ({asset.code} {ex.key}) stock news: earnings, guidance, analyst rating, share price move"


def fetch_news(ex: Exchange, days: int, cache_dir, use_cache: bool = True, per_asset: int = 10) -> pd.DataFrame:
    path = cache_dir / f"{ex.key}_exa.json"
    if use_cache and path.exists():
        rows = json.loads(path.read_text())["rows"]
    else:
        if not config.EXA_API_KEY:
            raise RuntimeError("EXA_API_KEY not set (use --demo for an offline run)")
        from exa_py import Exa

        exa = Exa(config.EXA_API_KEY)
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for a in ex.assets:
            try:
                res = exa.search(
                    _query(a, ex), num_results=per_asset, category="news",
                    start_published_date=start,
                    contents={"text": {"max_characters": 1500}, "highlights": {"num_sentences": 3}},
                )
            except Exception as e:
                log.error("Exa failed for %s: %s", a.ticker, e)
                continue
            for r in res.results:
                hl = " ".join(getattr(r, "highlights", None) or [])
                rows.append(dict(
                    exchange=ex.key, ticker=a.ticker, ts_utc=getattr(r, "published_date", None),
                    title=getattr(r, "title", "") or "", url=r.url,
                    text=hl or (getattr(r, "text", "") or "")[:1500],
                ))
            log.info("Exa: %s -> %d articles", a.ticker, len(res.results))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, default=str))
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.drop_duplicates(["ticker", "url"]).dropna(subset=["ts_utc"])
    return df
