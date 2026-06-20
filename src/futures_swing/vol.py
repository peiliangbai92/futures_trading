"""Volatility estimators + ATR.

The four range-based daily estimators (Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang) are vendored from QuantitativeResearch's
``option_research/realized_vol.py`` (pure numpy/pandas). Yang-Zhang is the
default — it is the only one that captures both intraday range and overnight
gaps. Added here: ``close_to_close_volatility`` (the forecast-vol denominator
for the Sharpe signal) and ``atr`` (true-range stops/targets).

All annualized estimators use 252 trading days and return *annualized* sigma.
``atr`` is in *price units* (not annualized).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _validate_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns: {missing}")
    work = frame[list(required)].astype(float)
    if (work["high"] < work["low"]).any():
        raise ValueError("OHLC frame contains rows where high < low")
    if (work[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("OHLC frame contains non-positive prices")
    return work


def parkinson_volatility(frame: pd.DataFrame, *, window: int = 21) -> pd.Series:
    """Parkinson (1980) high-low range estimator, annualized."""
    work = _validate_ohlc(frame)
    log_hl_sq = np.log(work["high"] / work["low"]) ** 2
    daily_var = log_hl_sq.rolling(window=window, min_periods=window).mean() / (4.0 * math.log(2.0))
    return np.sqrt(daily_var * TRADING_DAYS_PER_YEAR).rename("parkinson_vol")


def garman_klass_volatility(frame: pd.DataFrame, *, window: int = 21) -> pd.Series:
    """Garman-Klass (1980), annualized."""
    work = _validate_ohlc(frame)
    log_hl_sq = np.log(work["high"] / work["low"]) ** 2
    log_co_sq = np.log(work["close"] / work["open"]) ** 2
    daily = 0.5 * log_hl_sq - (2.0 * math.log(2.0) - 1.0) * log_co_sq
    daily_var = daily.rolling(window=window, min_periods=window).mean()
    return np.sqrt(daily_var * TRADING_DAYS_PER_YEAR).rename("garman_klass_vol")


def rogers_satchell_volatility(frame: pd.DataFrame, *, window: int = 21) -> pd.Series:
    """Rogers-Satchell (1991), annualized (drift-robust)."""
    work = _validate_ohlc(frame)
    log_ho = np.log(work["high"] / work["open"])
    log_lo = np.log(work["low"] / work["open"])
    log_co = np.log(work["close"] / work["open"])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    daily_var = rs.rolling(window=window, min_periods=window).mean()
    return np.sqrt(daily_var * TRADING_DAYS_PER_YEAR).rename("rogers_satchell_vol")


def yang_zhang_volatility(frame: pd.DataFrame, *, window: int = 21) -> pd.Series:
    """Yang-Zhang (2000), annualized — captures overnight gaps. Default."""
    if window < 2:
        raise ValueError("Yang-Zhang requires window >= 2")
    work = _validate_ohlc(frame)
    prev_close = work["close"].shift(1)
    log_oc_prev = np.log(work["open"] / prev_close)        # overnight
    log_co = np.log(work["close"] / work["open"])           # intraday
    log_ho = np.log(work["high"] / work["open"])
    log_lo = np.log(work["low"] / work["open"])
    rs_daily = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    sigma2_on = log_oc_prev.rolling(window=window, min_periods=window).var(ddof=1)
    sigma2_oc = log_co.rolling(window=window, min_periods=window).var(ddof=1)
    sigma2_rs = rs_daily.rolling(window=window, min_periods=window).sum() / (window - 1)

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    daily_var = sigma2_on + k * sigma2_oc + (1.0 - k) * sigma2_rs
    return np.sqrt(daily_var * TRADING_DAYS_PER_YEAR).rename("yang_zhang_vol")


def close_to_close_volatility(close: pd.Series, *, window: int = 21) -> pd.Series:
    """Annualized stdev of close-to-close log returns.

    Simple, robust forecast-vol denominator for the Sharpe signal. Uses only
    closes so it works for index/ETF series that have no clean OHLC.
    """
    log_ret = np.log(close.astype(float)).diff()
    daily = log_ret.rolling(window=window, min_periods=window).std(ddof=1)
    return (daily * math.sqrt(TRADING_DAYS_PER_YEAR)).rename("c2c_vol")


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range (price units): max(H-L, |H-prevC|, |L-prevC|)."""
    work = _validate_ohlc(frame)
    prev_close = work["close"].shift(1)
    tr = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rename("true_range")


def atr(frame: pd.DataFrame, *, window: int = 14) -> pd.Series:
    """Average True Range (price units), simple rolling mean of true range."""
    tr = true_range(frame)
    return tr.rolling(window=window, min_periods=window).mean().rename("atr")


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "close_to_close_volatility",
    "true_range",
    "atr",
]
