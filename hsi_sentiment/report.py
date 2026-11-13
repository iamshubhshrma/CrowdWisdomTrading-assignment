"""Self-contained visual HTML report (Plotly)."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version as _pjs
from plotly.subplots import make_subplots

from . import config

GREEN, RED, GREY, BLUE = "#16a34a", "#dc2626", "#6b7280", "#2563eb"
DIR_COLOR = {"Bullish": GREEN, "Bearish": RED, "Neutral": GREY}

def _fig_html(fig: go.Figure, height: int = 380) -> str:
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white",
                      font=dict(family="Inter, system-ui, sans-serif", size=12))
    # plotly.js is loaded once in <head>, so figure order in the page doesn't matter
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:+.{nd}f}%" if pct else f"{x:,.{nd}f}"


def _sent_rgb(v: float) -> str:
    """Grey at 0, fading to green (+) / red (-) at |v| >= 0.6."""
    a = min(abs(v) / 0.6, 1.0)
    end = (22, 163, 74) if v > 0 else (220, 38, 38)
    r, g, b = (round(209 + (e - 209) * a) for e in end)
    return f"rgb({r},{g},{b})"


def _sent_cell(v: float) -> str:
    w = min(abs(v), 1) * 50
    color = GREEN if v > 0 else RED
    left = 50 if v >= 0 else 50 - w
    return (f'<div class="bar"><span style="left:{left}%;width:{w}%;background:{color}"></span>'
            f'<em>{v:+.2f}</em></div>')


def ranking_table(s: pd.DataFrame) -> str:
    rows = []
    for i, r in s.iterrows():
        flags = []
        if r["crowded"]:
            flags.append('<span class="flag">crowded</span>')
        if r["divergence"]:
            flags.append(f'<span class="flag">{html.escape(r["divergence"])}</span>')
        crowd = f'{r["crowd_long_pct"]:.0f}% long ({r["holder_posts"]})' if pd.notna(r["crowd_long_pct"]) else "–"
        rows.append(f"""<tr>
<td>{i + 1}</td><td><b>{r['ticker']}</b><br><small>{html.escape(r['name'])} · {html.escape(r['local_name'])}</small></td>
<td class="num">{_fmt(r['price'])}</td>
<td class="num">{r['mentions']}<br><small>{r['mention_share'] * 100:.0f}% · {html.escape(r['sources'])}</small></td>
<td>{_sent_cell(r['social_sent'])}<small>{r['bulls']}▲ / {r['bears']}▼ · trend {r['sent_trend']:+.2f}</small></td>
<td>{_sent_cell(r['news_sent'])}<small>{r['news_n']} articles</small></td>
<td><span class="badge" style="background:{DIR_COLOR[r['direction']]}">{r['direction']}</span><br><small>score {r['composite']:+.2f}</small></td>
<td class="num">{_fmt(r['pred_close'])}<br><small>{_fmt(r['pred_ret'], pct=True)}</small></td>
<td><b>{r['position']}</b><br><small>stop {_fmt(r['stop'])} · tgt {_fmt(r['target'])} · R:R {_fmt(r['rr'], 1)}</small></td>
<td><small>S: {', '.join(_fmt(x) for x in r['supports'])}<br>R: {', '.join(_fmt(x) for x in r['resistances'])}<br>Pivot {_fmt(r['pivots']['P'])}</small></td>
<td><small>{crowd}</small></td><td>{''.join(flags)}</td></tr>""")
    return f"""<table class="rank"><thead><tr><th>#</th><th>Ticker</th><th>Price</th><th>Mentions</th>
<th>Social sentiment</th><th>News sentiment</th><th>Direction</th><th>n+1 forecast</th><th>Position</th>
<th>Support / Resistance</th><th>Crowd positioning</th><th>Flags</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def bubble(s: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for d, g in s.groupby("direction"):
        fig.add_trace(go.Scatter(
            x=g["social_sent"], y=g["mentions"], mode="markers+text", text=g["name"], textposition="top center",
            marker=dict(size=12 + 40 * g["engagement"] / max(s["engagement"].max(), 1), color=DIR_COLOR[d],
                        opacity=0.75, line=dict(width=1, color="white")),
            name=d, customdata=np.stack([g["pred_ret"] * 100, g["news_sent"]], axis=1),
            hovertemplate="%{text}<br>sentiment %{x:.2f}<br>mentions %{y}<br>n+1 %{customdata[0]:+.2f}%"
                          "<br>news %{customdata[1]:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=GREY)
    fig.update_layout(title="Attention vs conviction", xaxis_title="Engagement-weighted sentiment",
                      yaxis_title="Mentions (10d)")
    return fig


def heatmap(social_d: pd.DataFrame, names: dict) -> go.Figure:
    pv = social_d.pivot(index="ticker", columns="date", values="sent_w")
    cnt = social_d.pivot(index="ticker", columns="date", values="mentions").reindex_like(pv)
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=[d.strftime("%m-%d") for d in pv.columns], y=[names.get(t, t) for t in pv.index],
        zmid=0, zmin=-0.6, zmax=0.6, colorscale=[[0, RED], [0.5, "#f3f4f6"], [1, GREEN]],
        text=cnt.fillna(0).astype(int).values, texttemplate="%{text}",
        hovertemplate="%{y} %{x}<br>sentiment %{z:.2f}<br>posts %{text}<extra></extra>"))
    fig.update_layout(title="Daily sentiment per session (cell = posts)", xaxis_type="category")
    return fig


def ticker_chart(t: str, name: str, daily: pd.DataFrame, hourly: pd.DataFrame | None, posts: pd.DataFrame,
                 summ: pd.Series) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, column_widths=[0.55, 0.45], row_heights=[0.65, 0.35],
                        specs=[[{"rowspan": 2}, {}], [None, {}]], horizontal_spacing=0.08, vertical_spacing=0.12,
                        subplot_titles=("Daily · 90d with S/R and n+1 forecast", "Hourly price (10d)",
                                        "Posts per hour aligned to bars"))
    d = daily.tail(90)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="price", showlegend=False, increasing_line_color=GREEN,
                                 decreasing_line_color=RED), row=1, col=1)
    for lvl in summ["supports"]:
        fig.add_hline(y=lvl, line=dict(color=GREEN, dash="dash", width=1), row=1, col=1)
    for lvl in summ["resistances"]:
        fig.add_hline(y=lvl, line=dict(color=RED, dash="dash", width=1), row=1, col=1)
    nxt = pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), periods=1)[0]
    fig.add_trace(go.Scatter(x=[d.index[-1], nxt], y=[d["close"].iloc[-1], summ["pred_close"]], mode="lines+markers",
                             line=dict(color=BLUE, dash="dot"), marker=dict(size=[0, 10], symbol="star"),
                             name="n+1 forecast", showlegend=False), row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)

    if hourly is not None and not hourly.empty:
        h = hourly[hourly.index >= posts["ts_utc"].min() - pd.Timedelta(hours=6)] if len(posts) else hourly
        # break the line across non-trading gaps (overnight / weekend) instead of drawing a diagonal
        gap = h.index.to_series().diff() > pd.Timedelta(hours=2)
        h = pd.concat([h, pd.DataFrame({"close": np.nan}, index=h.index[gap] - pd.Timedelta(minutes=1))]).sort_index()
        fig.add_trace(go.Scatter(x=h.index, y=h["close"], mode="lines+markers", marker=dict(size=3), line=dict(color=BLUE, width=1.5),
                                 showlegend=False, name="hourly"), row=1, col=2)
    if len(posts):
        p = posts.assign(hour=posts["ts_utc"].dt.floor("h"))
        g = p.groupby("hour").agg(n=("sentiment", "size"), s=("sentiment", "mean"))
        fig.add_trace(go.Bar(x=g.index, y=g["n"], marker_color=[_sent_rgb(v) for v in g["s"]], showlegend=False, name="posts",
                             hovertemplate="%{x}<br>%{y} posts<extra></extra>"), row=2, col=2)
    fig.update_layout(title=f"{name} ({t}) — {summ['direction']} · {summ['position']}")
    return fig


def model_section(res: dict) -> str:
    m, oos = res["metrics"], res["oos"]
    if m.empty:
        return "<p>Not enough sentiment-window days for walk-forward validation yet.</p>"
    best = m.loc[m["mape_pct"].idxmin(), "model"]
    tbl = m.round(2).to_html(index=False, classes="metrics", border=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oos["target"] * 100, y=oos["pred_full"] * 100, mode="markers", text=oos["ticker"],
                             marker=dict(color=BLUE, size=8, opacity=0.7), name="price+sentiment"))
    fig.add_trace(go.Scatter(x=oos["target"] * 100, y=oos["pred_price"] * 100, mode="markers", text=oos["ticker"],
                             marker=dict(color=GREY, size=6, symbol="x"), name="price-only"))
    lim = max(oos["target"].abs().max(), 0.01) * 100
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(dash="dot", color=GREY),
                             showlegend=False))
    fig.update_layout(title="Out-of-sample: predicted vs realised next-session return (%)",
                      xaxis_title="realised %", yaxis_title="predicted %")
    used = bool(res["forecast"]["sentiment_used"].iloc[0])
    gate = ("<b>ON</b> — the sentiment overlay beat price-only out of sample, so today's n+1 forecast uses it."
            if used else "<b>OFF</b> — the sentiment overlay has not beaten price-only out of sample yet, so today's "
            "n+1 forecast falls back to price-only. Sentiment still drives the direction / position columns.")
    return (f"<p>Expanding-window walk-forward over {oos['date'].nunique()} sessions × {oos['ticker'].nunique()} "
            f"tickers ({len(oos)} OOS predictions). Lowest MAPE: <b>{best}</b>. Sentiment gate: {gate}</p>"
            f"{tbl}{_fig_html(fig, 360)}")


def ledger_section(done: pd.DataFrame) -> str:
    if done.empty:
        return "<p>No live n+1 predictions have matured yet. Run <code>python -m hsi_sentiment check</code> after the next session closes.</p>"
    cols = ["exchange", "ticker", "asof_date", "actual_date", "last_close", "pred_close", "actual_close", "err_pct",
            "rw_err_pct", "dir_hit", "demo"]
    return (f"<p>Matured predictions: {len(done)} · model MAPE {done['err_pct'].mean():.2f}% vs random walk "
            f"{done['rw_err_pct'].mean():.2f}% · direction hit {100 * done['dir_hit'].mean():.0f}%</p>"
            + done[cols].round(3).to_html(index=False, classes="metrics", border=0))


CSS = """
body{font-family:Inter,system-ui,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
header{background:#0f172a;color:white;padding:24px 32px} header h1{margin:0;font-size:24px}
header p{margin:6px 0 0;color:#cbd5e1} main{padding:16px 32px;max-width:1600px;margin:auto}
section{background:white;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 1px 3px #0001}
h2{margin-top:0} h3{margin-bottom:4px} .demo{background:#f59e0b;color:#111;padding:10px 32px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{padding:8px;border-bottom:1px solid #e5e7eb;vertical-align:top;text-align:left}
th{background:#f1f5f9;position:sticky;top:0} .num{text-align:right;font-variant-numeric:tabular-nums}
small{color:#64748b} .badge{color:white;padding:2px 8px;border-radius:999px;font-size:12px}
.flag{display:inline-block;background:#fef3c7;color:#92400e;padding:1px 6px;border-radius:6px;font-size:11px;margin:1px}
.bar{position:relative;height:16px;background:#f1f5f9;border-radius:4px;min-width:110px}
.bar span{position:absolute;top:0;height:100%;border-radius:4px;opacity:.8}
.bar em{position:absolute;right:4px;top:0;font-size:11px;font-style:normal}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px} .metrics{max-width:720px}
.tabs button{border:0;background:#e2e8f0;padding:8px 16px;border-radius:8px;margin-right:6px;cursor:pointer;font-weight:600}
.tabs button.on{background:#0f172a;color:white} .ex{display:none} .ex.on{display:block}
@media(max-width:900px){.grid{grid-template-columns:1fr} main{padding:8px}}
"""


def render(results: list[dict], run_ts: str, is_demo: bool, ledger: pd.DataFrame):
    tabs, panes = [], []
    for k, res in enumerate(results):
        ex, s = res["exchange"], res["summary"]
        names = {a.ticker: a.name for a in ex.assets}
        tabs.append(f'<button class="{"on" if k == 0 else ""}" onclick="show({k})">{ex.key}</button>')
        charts = "".join(
            f"<div>{_fig_html(ticker_chart(r['ticker'], r['name'], res['daily'][r['ticker']], res['hourly'].get(r['ticker']), res['posts'][res['posts']['ticker'] == r['ticker']], r), 420)}</div>"
            for _, r in s.iterrows())
        src_counts = res["posts"]["source"].value_counts().to_dict()
        panes.append(f"""<div class="ex {'on' if k == 0 else ''}" id="ex{k}">
<section><h2>{html.escape(ex.name)} — ranked by social mentions</h2>
<p>{len(res['posts'])} posts ({', '.join(f'{a}: {b}' for a, b in src_counts.items())}) · {len(res['news'])} Exa news articles ·
last {res['posts']['ts_utc'].min():%Y-%m-%d} → {res['posts']['ts_utc'].max():%Y-%m-%d %H:%M} UTC</p>
{ranking_table(s)}</section>
<section class="grid"><div>{_fig_html(bubble(s))}</div><div>{_fig_html(heatmap(res['social_daily'], names))}</div></section>
<section><h2>Per-ticker view</h2>{charts}</section>
<section><h2>n+1 model — walk-forward validation</h2>{model_section(res)}</section></div>""")

    demo_banner = ('<div class="demo">DEMO DATA — social posts and news are synthetic (prices are real). '
                   'Model scores below are not evidence of predictive power.</div>') if is_demo else ""
    plotly_js_version = _pjs()
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HK & KR Sentiment Report</title>
<script src="https://cdn.plot.ly/plotly-{plotly_js_version}.min.js" charset="utf-8"></script><style>{CSS}</style></head><body>
<header><h1>Retail Sentiment → n+1 Price Report</h1>
<p>HKEX (LIHKG + Telegram) · KRX (Naver 종목토론실 + Daum) · news via Exa · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · run {run_ts}</p></header>
{demo_banner}<main><div class="tabs">{''.join(tabs)}</div>{''.join(panes)}
<section><h2>Prediction ledger — did n+1 come true?</h2>{ledger_section(ledger)}</section>
<section><h2>Method</h2><ul>
<li><b>Alignment:</b> post timestamps → UTC → exchange local time; a post belongs to the session whose close it precedes, so features for day d are known at d's close and predict d+1. Posts are also snapped to the hourly bar they fall in.</li>
<li><b>Sentiment:</b> multilingual lexicon for Cantonese / Korean trader slang + English news, with negation handling; weighted by engagement (likes, replies, views) and author reach.</li>
<li><b>Model:</b> ridge on technicals over 1y, plus a ridge on sentiment features fit to its residuals over the 10-day window; refit at every step of an expanding walk-forward, benchmarked against a random walk.</li>
<li><b>Levels:</b> clustered 5-bar swing pivots + volume-profile nodes + floor pivots.</li>
<li><b>Position:</b> composite of social (45%), news (20%) and model (35%) scores; stop beyond the nearest level, target at the next level, only taken if reward:risk ≥ 1.</li>
</ul><p><small>Research output, not investment advice.</small></p></section></main>
<script>function show(k){{document.querySelectorAll('.ex').forEach((e,i)=>e.classList.toggle('on',i===k));
document.querySelectorAll('.tabs button').forEach((e,i)=>e.classList.toggle('on',i===k));window.dispatchEvent(new Event('resize'));}}</script>
</body></html>"""
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUT_DIR / ("report_demo.html" if is_demo else "report.html")
    path.write_text(doc, encoding="utf-8")
    return path
