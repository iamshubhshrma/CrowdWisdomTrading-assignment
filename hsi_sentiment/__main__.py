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
