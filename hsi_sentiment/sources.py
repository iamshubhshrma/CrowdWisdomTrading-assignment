"""Social post ingestion via Apify actors.

Each source has:
  * build_input(...)  -> the exact actor input sent to Apify (logged for reproducibility)
  * normalize(...)    -> maps raw dataset items into the common Post schema

Common Post schema (one row per post):
  post_id, source, exchange, ticker, ts_utc, text, likes, dislikes, replies, views,
  author, author_followers, url
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

import pandas as pd

from . import config
from .config import Asset, Exchange

log = logging.getLogger(__name__)

POST_COLUMNS = [
    "post_id", "source", "exchange", "ticker", "ts_utc", "text", "likes", "dislikes",
    "replies", "views", "author", "author_followers", "url",
]


def _num(v) -> float:
    try:
        return float(v) if v is not None and v != "" else 0.0
    except (TypeError, ValueError):
        return 0.0


def _to_utc(value, local_tz: str | None = None) -> pd.Timestamp | None:
    """Parse a timestamp; naive values are interpreted in `local_tz` then converted to UTC."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):  # epoch seconds
            return pd.Timestamp(value, unit="s", tz="UTC")
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(local_tz or "UTC")
    return ts.tz_convert("UTC")


def match_asset(text: str, assets: tuple[Asset, ...]) -> Asset | None:
    """Return the asset whose keywords appear most in `text` (None if no match)."""
    best, best_hits = None, 0
    low = text.lower()
    for a in assets:
        hits = 0
        for kw in a.keywords:
            k = kw.lower()
            if k.isdigit() or k.startswith("$"):
                # numeric codes must match as whole tokens (avoid "700" in "17000")
                hits += len(re.findall(rf"(?<![\d.]){re.escape(k)}(?![\d.])", low))
            else:
                hits += low.count(k)
        if hits > best_hits:
            best, best_hits = a, hits
    return best


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

def _since(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


# --- LIHKG (HK) ------------------------------------------------------------
def lihkg_input(ex: Exchange, days: int) -> dict:
    kws = [kw for a in ex.assets for kw in a.keywords[:2]]  # local-language names are what LIHKG users write
    return {
        "mode": "search",
        "keywords": kws,
        "order": "desc_create_time",
        "dateFrom": f"{days}d",
        "maxItems": 150,
        "proxyConfiguration": {"useApifyProxy": True},
    }


def lihkg_normalize(items: list[dict], ex: Exchange) -> list[dict]:
    kw_map = {kw: a for a in ex.assets for kw in a.keywords}
    out = []
    for it in items:
        text = " ".join(filter(None, [it.get("title"), it.get("content") or it.get("own_text")]))
        asset = kw_map.get(it.get("source_query") or "") or match_asset(text, ex.assets)
        if not asset:
            continue
        out.append(dict(
            post_id=f"lihkg:{it.get('thread_id')}:{it.get('floor', 0)}",
            source="lihkg", exchange=ex.key, ticker=asset.ticker,
            ts_utc=_to_utc(it.get("created_at"), ex.tz), text=text,
            likes=_num(it.get("like_count")), dislikes=_num(it.get("dislike_count")),
            replies=_num(it.get("reply_count")), views=_num(it.get("view_count")),
            author=it.get("author"), author_followers=_num(it.get("author_level")),
            url=it.get("thread_url"),
        ))
    return out


# --- Telegram (HK) ---------------------------------------------------------
def telegram_input(ex: Exchange, days: int) -> dict:
    return {
        "channels": list(config.TELEGRAM_CHANNELS),
        "lastNDays": days,
        "maxMessages": 500,
        "keywords": [kw for a in ex.assets for kw in a.keywords],
        "includeChannelInfo": True,
    }


def telegram_normalize(items: list[dict], ex: Exchange) -> list[dict]:
    subs: dict[str, float] = {}
    for it in items:
        if it.get("rowType") == "profile":
            n = next((v for k, v in it.items() if "subscriber" in k.lower() and isinstance(v, (int, float))), 0)
            subs[(it.get("username") or it.get("channelUsername") or "").lower()] = float(n)
    out = []
    for it in items:
        if it.get("rowType") not in ("message", "comment"):
            continue
        text = it.get("text") or ""
        asset = match_asset(text, ex.assets)
        if not asset:
            continue
        ch = (it.get("channelUsername") or "").lower()
        reactions = it.get("reactions")
        likes = sum(_num(r.get("count")) for r in reactions) if isinstance(reactions, list) else _num(reactions)
        out.append(dict(
            post_id=f"telegram:{ch}:{it.get('messageId')}", source="telegram", exchange=ex.key,
            ticker=asset.ticker, ts_utc=_to_utc(it.get("date")), text=text, likes=likes, dislikes=0.0,
            replies=_num(it.get("commentsCount") or it.get("replies")), views=_num(it.get("views")),
            author=ch, author_followers=subs.get(ch, 0.0), url=it.get("url"),
        ))
    return out


# --- Naver 종목토론실 (KR) ---------------------------------------------------
def naver_input(ex: Exchange, days: int) -> dict:
    return {
        "stocks": [a.code for a in ex.assets],
        "dateFrom": (_since(days) + timedelta(hours=9)).strftime("%Y-%m-%d"),
        "maxResults": 400,
    }


def naver_normalize(items: list[dict], ex: Exchange) -> list[dict]:
    by_code = {a.code: a for a in ex.assets}
    out = []
    for it in items:
        if it.get("type") != "post" or it.get("stockCode") not in by_code:
            continue
        a = by_code[it["stockCode"]]
        out.append(dict(
            post_id=f"naver:{it.get('postId')}", source="naver", exchange=ex.key, ticker=a.ticker,
            ts_utc=_to_utc(it.get("writtenAt"), ex.tz),
            text=" ".join(filter(None, [it.get("title"), it.get("text")])),
            likes=_num(it.get("likes")), dislikes=_num(it.get("dislikes")), replies=0.0,
            views=_num(it.get("views")), author=it.get("author"),
            # Naver exposes no follower count; holder-verified authors get a credibility boost instead
            author_followers=10.0 if it.get("isHolderVerified") else 0.0, url=it.get("url"),
        ))
    return out


# --- Daum 토론방 (KR) — official apify/cheerio-scraper with a custom pageFunction ---
DAUM_PAGE_FUNCTION = r"""
async function pageFunction(context) {
    const { request, json, body } = context;
    const data = json || JSON.parse(body.toString());
    const posts = (data.data && data.data.posts) || [];
    const cutoff = new Date(request.userData.cutoff).getTime();
    let oldest = Date.now();
    for (const p of posts) {
        const t = new Date(p.createdAt.replace(' ', 'T') + '+09:00').getTime();
        oldest = Math.min(oldest, t);
    }
    const page = request.userData.page;
    if (posts.length && oldest >= cutoff && page < data.totalPages && page < request.userData.maxPages) {
        const next = request.url.replace(/page=\d+/, 'page=' + (page + 1));
        await context.enqueueRequest({ url: next, headers: request.headers,
            userData: { ...request.userData, page: page + 1 } });
    }
    return posts.map(p => ({ ...p, code: request.userData.code }));
}
"""


def daum_input(ex: Exchange, days: int) -> dict:
    cutoff = _since(days).isoformat()
    return {
        "startUrls": [{
            "url": f"https://finance.daum.net/content/debates/A{a.code}?page=1&perPage=50",
            "headers": {"Referer": f"https://finance.daum.net/quotes/A{a.code}"},
            "userData": {"code": a.code, "page": 1, "cutoff": cutoff, "maxPages": 20},
        } for a in ex.assets],
        "pageFunction": DAUM_PAGE_FUNCTION,
        "additionalMimeTypes": ["application/json"],
        "proxyConfiguration": {"useApifyProxy": True},
        "maxConcurrency": 5,
    }


def daum_normalize(items: list[dict], ex: Exchange) -> list[dict]:
    by_code = {a.code: a for a in ex.assets}
    out = []
    for it in items:
        code = it.get("code") or (it.get("symbolCode") or "").lstrip("A")
        if code not in by_code:
            continue
        out.append(dict(
            post_id=f"daum:{it.get('id')}", source="daum", exchange=ex.key, ticker=by_code[code].ticker,
            ts_utc=_to_utc(it.get("createdAt"), ex.tz),
            text=" ".join(filter(None, [it.get("title"), it.get("content")])),
            likes=_num(it.get("agreeCount")), dislikes=_num(it.get("disagreeCount")),
            replies=_num(it.get("commentCount")), views=_num(it.get("readCount")),
            author=it.get("daumName"), author_followers=0.0,
            url=f"https://finance.daum.net/quotes/A{code}#talks",
        ))
    return out


SOURCES: dict[str, tuple[Callable, Callable]] = {
    "lihkg": (lihkg_input, lihkg_normalize),
    "telegram": (telegram_input, telegram_normalize),
    "naver": (naver_input, naver_normalize),
    "daum": (daum_input, daum_normalize),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_actor(actor_id: str, run_input: dict) -> list[dict]:
    from apify_client import ApifyClient

    client = ApifyClient(config.APIFY_TOKEN)
    log.info("Apify: starting %s", actor_id)
    run = client.actor(actor_id).call(
        run_input=run_input,
        max_total_charge_usd=Decimal(str(config.APIFY_MAX_CHARGE_USD)),
    )
    if run is None:
        raise RuntimeError(f"Apify run for {actor_id} returned nothing")
    ds = getattr(run, "default_dataset_id", None) or run["defaultDatasetId"]
    status = getattr(run, "status", None) or run.get("status")
    items = list(client.dataset(ds).iterate_items())
    log.info("Apify: %s finished (%s) with %d items, dataset %s", actor_id, status, len(items), ds)
    return items


def fetch_posts(ex: Exchange, days: int, cache_dir, use_cache: bool = True) -> pd.DataFrame:
    """Run every configured source for an exchange; raw items are cached as JSON."""
    rows: list[dict] = []
    for src in ex.sources:
        build_input, normalize = SOURCES[src]
        raw_path = cache_dir / f"{ex.key}_{src}.json"
        if use_cache and raw_path.exists():
            items = json.loads(raw_path.read_text())["items"]
            log.info("%s/%s: loaded %d cached items", ex.key, src, len(items))
        else:
            if not config.APIFY_TOKEN:
                raise RuntimeError("APIFY_TOKEN not set (use --demo for an offline run)")
            run_input = build_input(ex, days)
            try:
                items = run_actor(config.ACTORS[src], run_input)
            except Exception as e:  # one dead source must not kill the pipeline
                log.error("%s/%s failed: %s", ex.key, src, e)
                continue
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(
                {"actor": config.ACTORS[src], "input": run_input, "items": items},
                ensure_ascii=False, default=str))
        rows.extend(normalize(items, ex))
    return to_frame(rows, days)


def to_frame(rows: list[dict], days: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=POST_COLUMNS)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.dropna(subset=["ts_utc"]).drop_duplicates("post_id")
    if days:
        df = df[df["ts_utc"] >= pd.Timestamp(_since(days))]
    return df.sort_values("ts_utc").reset_index(drop=True)
