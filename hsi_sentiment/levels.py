"""Support / resistance detection.

Combines three independent methods and keeps the levels where they agree:
  * swing pivots  — fractal highs/lows (5-bar) over ~6 months, clustered within 0.5 ATR,
                    scored by number of touches and recency
  * floor pivots  — classic P / R1 / S1 from the last session
  * volume nodes  — price bins with the highest traded volume (volume profile)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _swing_levels(df: pd.DataFrame, k: int = 2) -> list[tuple[float, float]]:
    h, l = df["high"].values, df["low"].values
    pts = []
    n = len(df)
    for i in range(k, n - k):
        recency = 0.5 + 0.5 * i / n
        if h[i] == h[i - k:i + k + 1].max():
            pts.append((h[i], recency))
        if l[i] == l[i - k:i + k + 1].min():
            pts.append((l[i], recency))
    return pts


def _cluster(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    pts = sorted(points)
    clusters: list[list[tuple[float, float]]] = []
    for p in pts:
        if clusters and p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        w = np.array([x[1] for x in c])
        price = float(np.average([x[0] for x in c], weights=w))
        out.append((price, float(w.sum())))
    return out


def _volume_nodes(df: pd.DataFrame, bins: int = 30, top: int = 3) -> list[float]:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    hist, edges = np.histogram(typical, bins=bins, weights=df["volume"])
    idx = np.argsort(hist)[-top:]
    return [float((edges[i] + edges[i + 1]) / 2) for i in idx]


def support_resistance(df: pd.DataFrame, lookback: int = 126) -> dict:
    d = df.tail(lookback)
    price = float(d["close"].iloc[-1])
    a = atr(d)
    levels = _cluster(_swing_levels(d), tol=0.5 * a)
    for v in _volume_nodes(d):  # volume nodes add confluence weight
        levels.append((v, 1.0))
    levels = _cluster(levels, tol=0.5 * a)

    last = d.iloc[-1]
    pivot = (last["high"] + last["low"] + last["close"]) / 3
    pivots = {"P": pivot, "R1": 2 * pivot - last["low"], "S1": 2 * pivot - last["high"],
              "R2": pivot + (last["high"] - last["low"]), "S2": pivot - (last["high"] - last["low"])}

    sup = sorted([lv for lv in levels if lv[0] < price - 0.1 * a], key=lambda x: (-x[0]))
    res = sorted([lv for lv in levels if lv[0] > price + 0.1 * a], key=lambda x: x[0])
    # nearest two levels on each side, preferring stronger ones among the closest four
    pick = lambda lv: sorted(sorted(lv[:4], key=lambda x: -x[1])[:2], key=lambda x: abs(x[0] - price))
    supports = [round(x[0], 2) for x in pick(sup)] or [round(pivots["S1"], 2)]
    resistances = [round(x[0], 2) for x in pick(res)] or [round(pivots["R1"], 2)]
    return {
        "price": price, "atr": a, "supports": supports, "resistances": resistances,
        "pivots": {k: round(float(v), 2) for k, v in pivots.items()},
        "high_52w": float(df["high"].tail(252).max()), "low_52w": float(df["low"].tail(252).min()),
    }
