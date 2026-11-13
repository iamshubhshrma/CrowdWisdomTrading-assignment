"""n+1 price model with strict walk-forward (expanding window) out-of-sample validation.

Two-stage design, because sentiment exists for ~10 days but prices for a year:
  1. price model   — Ridge on technical features, trained on the 1y pooled panel
  2. residual model — Ridge on social/news features, trained on the sentiment window,
                      predicting what the price model gets wrong
  final prediction  = price_pred + residual_pred   (log-return of the next session)

At every test date t both stages are refit on rows whose *target* was known
before t (row date < t), so no future information leaks into a prediction.
Benchmarks reported side by side: random walk (pred = 0) and price-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PRICE_COLS, SOCIAL_COLS

MIN_TRAIN_DAYS = 3   # sentiment-window days needed before the first OOS prediction
CLIP = 0.10          # cap |predicted log-return| at 10%


def _ridge(alpha: float, intercept: bool = True):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=intercept))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.dropna(subset=PRICE_COLS + ["target"])
    pm = _ridge(10.0).fit(tr[PRICE_COLS], tr["target"])
    price_pred = pm.predict(test[PRICE_COLS].fillna(0))

    win = tr[tr["in_window"]]
    resid_pred = np.zeros(len(test))
    if len(win) >= 10:
        resid = win["target"] - pm.predict(win[PRICE_COLS])
        # heavy shrinkage (tiny sample) and no intercept: a free intercept would just
        # memorise the last few days' market-wide drift and extrapolate it
        rm = _ridge(100.0, intercept=False).fit(win[SOCIAL_COLS], resid)
        resid_pred = rm.predict(test[SOCIAL_COLS].fillna(0))
    return np.clip(price_pred, -CLIP, CLIP), np.clip(price_pred + resid_pred, -CLIP, CLIP)


def walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one OOS prediction row per (ticker, test date)."""
    p = panel.dropna(subset=PRICE_COLS).copy()
    win_dates = sorted(p.loc[p["in_window"] & p["target"].notna(), "date"].unique())
    out = []
    se_price = se_full = 0.0  # cumulative OOS squared errors seen *before* each test date
    for t in win_dates[MIN_TRAIN_DAYS:]:
        train = p[(p["date"] < t) & p["target"].notna()]
        test = p[p["date"] == t]
        price_pred, full_pred = _fit_predict(train, test)
        # gate: use the sentiment overlay only if it has beaten price-only out of sample so far
        use_full = bool(out) and se_full < se_price
        out.append(test[["date", "ticker", "close", "target"]].assign(
            pred_rw=0.0, pred_price=price_pred, pred_full=full_pred,
            pred_gated=full_pred if use_full else price_pred, gate_on=use_full))
        se_price += float(((price_pred - test["target"]) ** 2).sum())
        se_full += float(((full_pred - test["target"]) ** 2).sum())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(oos: pd.DataFrame) -> pd.DataFrame:
    """MAE on price, RMSE on return, directional hit-rate for every model."""
    if oos.empty:
        return pd.DataFrame()
    actual_px = oos["close"] * np.exp(oos["target"])
    rows = []
    for name, col in [("Random walk", "pred_rw"), ("Price-only ridge", "pred_price"),
                      ("Price + sentiment", "pred_full"), ("Gated (deployed)", "pred_gated")]:
        pred_px = oos["close"] * np.exp(oos[col])
        err_pct = (pred_px - actual_px).abs() / actual_px
        nz = oos[col] != 0
        hit = (np.sign(oos.loc[nz, col]) == np.sign(oos.loc[nz, "target"])).mean() if nz.any() else np.nan
        rows.append(dict(model=name, n=len(oos), mape_pct=100 * err_pct.mean(),
                         rmse_ret_bp=1e4 * np.sqrt(((oos[col] - oos["target"]) ** 2).mean()),
                         hit_rate_pct=100 * hit if pd.notna(hit) else np.nan))
    return pd.DataFrame(rows)


def sentiment_helps(oos: pd.DataFrame) -> bool:
    """Deploy the sentiment overlay only if it beat price-only over the whole walk-forward."""
    if oos.empty:
        return False
    return bool(((oos["pred_full"] - oos["target"]) ** 2).sum() < ((oos["pred_price"] - oos["target"]) ** 2).sum())


def forecast(panel: pd.DataFrame, use_sentiment: bool = True) -> pd.DataFrame:
    """Fit on everything known and predict the next session for each ticker's latest row."""
    p = panel.dropna(subset=PRICE_COLS)
    last = p["date"].max()
    test = p[p["date"] == last]
    train = p[p["target"].notna()]
    price_pred, full_pred = _fit_predict(train, test)
    pred = full_pred if use_sentiment else price_pred
    return test[["date", "ticker", "close"]].assign(
        pred_ret=pred, pred_ret_price=price_pred, pred_ret_sent=full_pred, sentiment_used=use_sentiment,
        pred_close=test["close"].values * np.exp(pred)).reset_index(drop=True)
