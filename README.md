# HK & KR Retail Sentiment → n+1 Price Prediction

A Python pipeline that scrapes retail-investor chatter for **5 Hang Seng (HKEX) and 5 KOSPI (KRX) stocks**, adds recent news from **Exa**, aligns everything to price bars and predicts each stock's **next-session (n+1) close**. It validates that prediction with a strict walk-forward test and renders a visual, ranked report for each exchange.

| Exchange | Tickers | Social source 1 | Social source 2 | News |
|---|---|---|---|---|
| HKEX | 0700 Tencent · 9988 Alibaba · 1810 Xiaomi · 3690 Meituan · 0005 HSBC | **LIHKG** 連登 (keyword search) | **Telegram** public channels | Exa |
| KRX | 005930 Samsung · 000660 SK Hynix · 035420 NAVER · 005380 Hyundai · 035720 Kakao | **Naver 종목토론실** | **Daum 토론방** | Exa |

## Quick start

```bash
uv sync
cp .env.example .env            # add APIFY_TOKEN and EXA_API_KEY
uv run python -m hsi_sentiment run          # live run -> output/report.html
uv run python -m hsi_sentiment check        # after the next close: score the n+1 predictions
uv run python -m hsi_sentiment run --demo   # offline: synthetic posts, real prices (no keys needed)
uv run pytest
```

Useful flags: `--exchanges HKEX`, `--days 30` (more walk-forward history), `--no-cache` (re-run Apify/Exa even though today's raw cache already exists).

## Apify actors used (for reproduction)

| Source | Actor | Actor ID | Input (built in `hsi_sentiment/sources.py`) |
|---|---|---|---|
| LIHKG | `claude_code_reviewer/lihkg-scraper-en` | `PJDFRxJdpPknN9nYn` | `mode: search`, Traditional-Chinese / English name keywords, `dateFrom: 10d`, `maxItems: 40` per keyword |
| Telegram | `automation-lab/telegram-scraper` | `wxRoaBWQSXNzpVH1D` | `channels: [rthk_new_c, StockIntelligenceNews, StockIntelligence]`, `lastNDays: 10`, ticker `keywords`, `includeChannelInfo` (subscriber counts) |
| Naver 종목토론실 | `apify/cheerio-scraper` (official) | `YrQuEkowkNCLdk4j2` | Naver's discussion JSON API `m.stock.naver.com/front-api/discussion/list?itemCode={code}&pageSize=100`, custom `pageFunction` that follows the `offset` cursor back 10 days (max 300 pages per stock) |
| Daum 토론방 | `apify/cheerio-scraper` (official) | `YrQuEkowkNCLdk4j2` | Daum's public JSON endpoint `finance.daum.net/content/debates/A{code}`, custom `pageFunction` that paginates back 10 days |

Every run writes the exact actor ID, input and raw dataset items to `data/raw/<date>/<EXCHANGE>_<source>.json`. Actor IDs can be overridden through env vars (see `.env.example`).

- **Permissions:** `apify/cheerio-scraper` needs a one-time permission approval in the Apify console before its first run.
- **Why cheerio for Naver:** Naver is scraped with cheerio rather than a per-post-priced Store actor. Samsung and SK Hynix get about 4–5k posts a day, so 10 days through a per-post actor would cost tens of dollars. Through cheerio it costs cents of compute.
- **Checking Telegram:** `uv run python -m hsi_sentiment telegram-check [channels…]` verifies Telegram channels with one cheap call. It prints subscribers, messages in the last 10 days, ticker hits and the last post time.

## Live run (2026-11-13)

`output/report.html` is the report from the live run.

| Exchange | Posts | By source | News (Exa) |
|---|---|---|---|
| HKEX | 347 | LIHKG 18 threads · Telegram ~330 messages | 50 |
| KRX | 72,783 | Naver 72,524 · Daum 259 | 50 |

Walk-forward out-of-sample results (20 predictions per exchange):

| Exchange | Model | MAPE | Direction hit rate |
|---|---|---|---|
| HKEX | Price + sentiment | 2.20% | 55% |
| HKEX | Price-only | 2.23% | 70% |
| HKEX | Random walk | 2.27% | – |
| KRX | Price + sentiment | 2.43% | 55% |
| KRX | Price-only | 2.40% | 55% |
| KRX | Random walk | 2.45% | – |

Sentiment gives a marginal gain on HKEX error and none on KRX. With this sample size the honest reading is "no proven edge yet". That is why the deployed forecast is gated on out-of-sample performance.

## How it works

```
Apify actors ──► normalise to one Post schema ──► score sentiment ──► align to bars ──► daily panel ──► walk-forward model ──► report
Exa search  ──► news per ticker ───────────────┘                                    yfinance (1y daily + 10d hourly) ─┘
```

1. **Ingestion** (`sources.py`, `news.py`, `prices.py`). Each post is normalised to `post_id, source, ticker, ts_utc, text, likes, dislikes, replies, views, author, author_followers, url`. Timestamps given in local time (KST/HKT) are converted to UTC.
2. **Sentiment** (`sentiment.py`). Generic models miss most retail slang (坐艇, 插水, 떡상, 존버, 물렸다…), so the scorer is a multilingual lexicon for Cantonese, Korean and English, with negation and longest-match handling. Setting `SENTIMENT_MODEL=hf` blends in XLM-RoBERTa. Each post is weighted by engagement and author reach.
3. **Alignment** (`features.py`). A post belongs to the trading session whose close it **precedes**, so the features for day *d* are fully known at *d*'s close and are used to predict *d+1*. Posts are also matched to the hourly bar they fall in, which drives the "posts per hour" charts.
4. **Model** (`model.py`). This is a two-stage ridge regression:
   - A price model fits technical features over 1 year of history.
   - A sentiment model fits the price model's *residuals* over the 10-day window, using mention surprise, weighted sentiment, bull share, news sentiment and sentiment momentum.

   Both stages are refit at every step of an expanding walk-forward and benchmarked against a **random walk** and the **price-only** model.

   A **leak-free gate** keeps the sentiment overlay only while its *past* out-of-sample error is lower than the price-only model's. Sentiment is therefore never trusted just because it is there.
5. **Per-ticker output** (`levels.py`, `signals.py`):
   - ticker, name and current price
   - social and news sentiment, plus the sentiment trend
   - direction: a composite of social 45%, news 20% and model 35%
   - suggested position (LONG / SHORT / WAIT / FLAT), with entry, stop, target and reward:risk
   - support and resistance from clustered swing pivots, volume-profile nodes and floor pivots
   - "crowd positioning": the share of self-declared holders (보유/존버/揸住/坐艇…) who are still positive
   - flags for crowded trades and for sentiment/price divergence
6. **Report** (`report.py`). An HTML report with one tab per exchange. Each tab has the ranked table (sorted by mentions, then sentiment), an attention-vs-conviction bubble chart and a daily sentiment heatmap. It also shows per-ticker candles with S/R and the n+1 forecast star, hourly price with post volume, the walk-forward metrics and a predicted-vs-realised scatter.
7. **Checking n+1** (`pipeline.py`). Every run appends its forecasts to `data/predictions.csv`. `check` fills in the realised close and reports the model's error against a random walk, plus its directional hit rate.

## Limitations

- 10 days of sentiment covers about 7 trading sessions, so the walk-forward has only a few test sessions per exchange. Treat the metrics as a sanity check, not proof of an edge. `--days 30` gives a more meaningful test.
- Samsung and SK Hynix reach the 300-page (30k-post) Naver cap and cover the last 7–8 days instead of 10. The oldest heatmap cells for those two hold only a handful of posts.
- Retail traders rarely discuss these names in LIHKG thread titles (about 18 threads in 10 days), and most active HK Telegram chatter is in private groups. HKEX volume is therefore thin and leans on news-style channels.
- Naver and Daum don't expose follower counts. Naver's holder-verified flag is used as the credibility signal instead, and LIHKG uses author level.
- The lexicon scores a large share of Korean board posts (memes, off-topic) as neutral, which pulls the average sentiment towards 0.
- Demo mode (`--demo`) generates synthetic text on top of real prices. Its scores mean nothing and the report is stamped accordingly.
- This is research output, not investment advice.
