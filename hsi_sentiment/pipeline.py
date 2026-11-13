"""End-to-end orchestration + prediction ledger ("predict n+1 and check")."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, demo, features, levels, model, news as news_mod, prices, signals, sources
from .config import EXCHANGES, Exchange
from .sentiment import score_texts

log = logging.getLogger(__name__)
LEDGER = config.DATA_DIR / "predictions.csv"


def run_exchange(ex: Exchange, days: int, cache_dir, use_cache: bool, is_demo: bool) -> dict:
    tickers = [a.ticker for a in ex.assets]
    daily = prices.daily_bars(tickers)
    hourly = prices.hourly_bars(tickers, days)
    if not daily:
        raise RuntimeError(f"no price data for {ex.key}")
    trading_days = pd.DatetimeIndex(sorted(set().union(*[d.index for d in daily.values()])))

    if is_demo:
        posts, news = demo.generate(ex, daily, days)
    else:
        posts = sources.fetch_posts(ex, days, cache_dir, use_cache)
        try:
            news = news_mod.fetch_news(ex, days, cache_dir, use_cache)
        except RuntimeError as e:
            log.error("%s news skipped: %s", ex.key, e)
            news = pd.DataFrame(columns=news_mod.NEWS_COLUMNS)
    log.info("%s: %d posts, %d news articles", ex.key, len(posts), len(news))
    if posts.empty:
        raise RuntimeError(f"{ex.key}: no posts collected — check Apify runs in data/raw")

    posts = features.align_hourly(features.score_posts(posts), hourly)
    if not news.empty:
        news = news.copy()
        news["sentiment"] = score_texts((news["title"].fillna("") + ". " + news["text"].fillna("")).tolist())

    social_d = features.daily_social(posts, ex, trading_days)
    news_d = features.daily_news(news, ex, trading_days)
    panel = features.build_panel(features.price_features(daily), social_d, news_d)

    oos = model.walk_forward(panel)
    metrics = model.evaluate(oos)
    fc = model.forecast(panel, use_sentiment=model.sentiment_helps(oos))
    lv = {t: levels.support_resistance(df) for t, df in daily.items()}
    names = {a.ticker: a.name for a in ex.assets}
    summary = signals.ticker_summary(ex, posts, news, fc, lv, names)
    return dict(exchange=ex, daily=daily, hourly=hourly, posts=posts, news=news, social_daily=social_d,
                panel=panel, oos=oos, metrics=metrics, forecast=fc, levels=lv, summary=summary)


def record_predictions(results: list[dict], run_ts: str, is_demo: bool) -> pd.DataFrame:
    rows = []
    for r in results:
        for _, f in r["forecast"].iterrows():
            rows.append(dict(run_ts=run_ts, exchange=r["exchange"].key, ticker=f["ticker"],
                             asof_date=f["date"].date().isoformat(), last_close=f["close"],
                             pred_close=f["pred_close"], pred_ret=f["pred_ret"], demo=is_demo,
                             actual_date="", actual_close=np.nan))
    new = pd.DataFrame(rows)
    if LEDGER.exists():
        old = pd.read_csv(LEDGER, dtype={"actual_date": str})
        # one prediction per (ticker, asof_date, demo) — the newest run wins
        new = pd.concat([old, new]).drop_duplicates(["ticker", "asof_date", "demo"], keep="last")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(LEDGER, index=False)
    return new


def check_predictions() -> pd.DataFrame:
    """Fill in realised closes for past n+1 predictions and score them."""
    if not LEDGER.exists():
        raise SystemExit("no predictions recorded yet — run the pipeline first")
    led = pd.read_csv(LEDGER, dtype={"actual_date": str})
    daily = prices.daily_bars(sorted(led["ticker"].unique()), period="3mo")
    for i, r in led.iterrows():
        df = daily.get(r["ticker"])
        if df is None:
            continue
        after = df[df.index > pd.Timestamp(r["asof_date"])]
        if len(after):
            led.at[i, "actual_date"] = after.index[0].date().isoformat()
            led.at[i, "actual_close"] = float(after["close"].iloc[0])
    led.to_csv(LEDGER, index=False)
    done = led.dropna(subset=["actual_close"]).copy()
    if done.empty:
        return done
    done["err_pct"] = 100 * (done["pred_close"] - done["actual_close"]).abs() / done["actual_close"]
    done["rw_err_pct"] = 100 * (done["last_close"] - done["actual_close"]).abs() / done["actual_close"]
    done["dir_hit"] = np.sign(done["pred_ret"]) == np.sign(np.log(done["actual_close"] / done["last_close"]))
    return done


def run(exchanges: list[str], days: int, use_cache: bool, is_demo: bool) -> tuple[list[dict], str]:
    from . import report

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_dir = config.RAW_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results = []
    for key in exchanges:
        try:
            results.append(run_exchange(EXCHANGES[key], days, cache_dir, use_cache, is_demo))
        except Exception as e:
            log.exception("%s failed: %s", key, e)
    if not results:
        raise SystemExit("all exchanges failed")
    record_predictions(results, run_ts, is_demo)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        r["summary"].to_json(config.OUT_DIR / f"summary_{r['exchange'].key}.json", orient="records",
                             force_ascii=False, indent=1, date_format="iso")
        r["posts"].drop(columns=["text"]).to_csv(config.OUT_DIR / f"posts_{r['exchange'].key}.csv", index=False)
    path = report.render(results, run_ts, is_demo, check_predictions_safe())
    (config.OUT_DIR / "run_meta.json").write_text(json.dumps(
        {"run_ts": run_ts, "demo": is_demo, "days": days, "actors": config.ACTORS,
         "telegram_channels": config.TELEGRAM_CHANNELS}, indent=1))
    return results, str(path)


def check_predictions_safe() -> pd.DataFrame:
    try:
        return check_predictions()
    except (SystemExit, Exception) as e:  # report must render even if the ledger check fails
        log.warning("prediction check skipped: %s", e)
        return pd.DataFrame()
