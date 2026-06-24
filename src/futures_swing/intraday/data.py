"""Intraday price layer — 5-minute ES bars + session VWAP/ATR (MVP, free yfinance).

yfinance serves ~60 days of 5-min history (enough to prototype + a feasibility
probe, not to validate). Cached to data/intraday/<KEY>_5m.parquet. All timestamps
are tz-aware US/Eastern so RTH (09:30-16:00 ET) and the session VWAP anchor align.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data_loader

CACHE_DIR = data_loader.REPO_ROOT / "data" / "intraday"
ET = "America/New_York"
RTH_START, RTH_END = 9 * 60 + 30, 16 * 60      # minutes since ET midnight


def _flatten_5m(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)   # yfinance intraday is tz-aware
    df.index = idx; df.index.name = "ts"
    keep = ["open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[keep].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]


def refresh_5m(key="ES", yf_symbol="ES=F", period="60d") -> pd.DataFrame:
    """Pull 5-min bars and merge into the cache (keeps prior history yfinance drops)."""
    import yfinance as yf
    raw = yf.download(yf_symbol, period=period, interval="5m", progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no 5m data for {yf_symbol}")
    new = _flatten_5m(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}_5m.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new])
        new = new[~new.index.duplicated(keep="last")].sort_index()
    new.to_parquet(path)
    return new


def load_5m(key="ES") -> pd.DataFrame:
    path = CACHE_DIR / f"{key}_5m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run intraday.data.refresh_5m() first")
    return pd.read_parquet(path)


def add_session_features(df: pd.DataFrame, *, rth_only=True, atr_window=14) -> pd.DataFrame:
    """Add session_date, rth flag, cumulative-to-t session VWAP, and a rolling
    intraday ATR. All strictly causal (cumsum/rolling use only bars <= t)."""
    df = df.copy()
    mins = df.index.hour * 60 + df.index.minute
    df["rth"] = (mins >= RTH_START) & (mins < RTH_END)
    df["session_date"] = df.index.tz_convert(ET).normalize().tz_localize(None)

    # rolling intraday ATR (true range over the continuous 5-min series)
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_window, min_periods=atr_window).mean()

    rth = df[df["rth"]] if rth_only else df
    tp = (rth["high"] + rth["low"] + rth["close"]) / 3.0
    vol = rth["volume"].where(rth["volume"] > 0)
    g = rth.groupby("session_date")
    cum_pv = (tp * vol).groupby(rth["session_date"]).cumsum()
    cum_v = vol.groupby(rth["session_date"]).cumsum()
    vwap = cum_pv / cum_v
    # fallback when volume is missing/zero: expanding mean of typical price
    vwap = vwap.fillna(tp.groupby(rth["session_date"]).expanding().mean().reset_index(level=0, drop=True))
    df["vwap"] = vwap.reindex(df.index)
    return df
