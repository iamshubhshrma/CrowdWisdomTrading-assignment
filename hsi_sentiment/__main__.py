"""CLI.

  uv run python -m hsi_sentiment run            # live: Apify + Exa + yfinance
  uv run python -m hsi_sentiment run --demo     # offline: synthetic posts, real prices
  uv run python -m hsi_sentiment check          # score past n+1 predictions against realised closes
"""
from __future__ import annotations

import argparse
import logging

from . import config, pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="hsi_sentiment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="scrape, score, model, report")
    r.add_argument("--exchanges", nargs="+", default=list(config.EXCHANGES), choices=list(config.EXCHANGES))
    r.add_argument("--days", type=int, default=config.LOOKBACK_DAYS)
    r.add_argument("--no-cache", action="store_true", help="re-run Apify/Exa even if today's raw cache exists")
    r.add_argument("--demo", action="store_true", help="synthetic social/news data (no API keys needed)")
    sub.add_parser("check", help="evaluate recorded n+1 predictions")
    tg = sub.add_parser("telegram-check", help="verify Telegram channels exist and are active (cheap Apify call)")
    tg.add_argument("channels", nargs="*", help="defaults to TELEGRAM_CHANNELS")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("yfinance", "urllib3", "peewee", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.cmd == "run":
        results, path = pipeline.run(a.exchanges, a.days, not a.no_cache, a.demo)
        for res in results:
            print(f"\n=== {res['exchange'].name} ===")
            cols = ["ticker", "name", "price", "mentions", "social_sent", "direction", "pred_close", "position"]
            print(res["summary"][cols].round(3).to_string(index=False))
            if not res["metrics"].empty:
                print("\nWalk-forward OOS:")
                print(res["metrics"].round(2).to_string(index=False))
        print(f"\nReport: {path}")
    elif a.cmd == "telegram-check":
        from .sources import run_actor

        chans = a.channels or list(config.TELEGRAM_CHANNELS)
        keys = [k for asset in config.HKEX.assets for k in asset.keywords]
        items = run_actor(config.ACTORS["telegram"], {"channels": chans, "maxMessages": 50, "lastNDays": 10,
                                                      "includeChannelInfo": True})
        for ch in chans:
            rows = [i for i in items if (i.get("channelUsername") or i.get("username") or "").lower() == ch.lower()]
            prof = next((i for i in rows if i.get("rowType") == "profile"), {})
            msgs = [i for i in rows if i.get("rowType") == "message"]
            subs = next((v for k, v in prof.items() if "subscriber" in k.lower()), "?")
            hits = sum(any(k.lower() in (m.get("text") or "").lower() for k in keys) for m in msgs)
            last = max((m.get("date") or "" for m in msgs), default="-")
            ok = "OK " if msgs and hits else "LOW" if msgs else "BAD"
            print(f"{ok} {ch:<24} subs={subs!s:<8} msgs_10d={len(msgs):<4} ticker_hits={hits:<4} last={last[:16]}")
    else:
        done = pipeline.check_predictions()
        if done.empty:
            print("No predictions have a realised close yet.")
            return
        cols = ["ticker", "asof_date", "actual_date", "last_close", "pred_close", "actual_close", "err_pct",
                "rw_err_pct", "dir_hit", "demo"]
        print(done[cols].round(3).to_string(index=False))
        print(f"\nMAPE model {done['err_pct'].mean():.2f}% vs random walk {done['rw_err_pct'].mean():.2f}% | "
              f"direction hit {100 * done['dir_hit'].mean():.0f}%")


if __name__ == "__main__":
    main()
