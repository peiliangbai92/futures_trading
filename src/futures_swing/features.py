"""Feature engineering — point-in-time feature matrix per instrument (V1).

Every feature at date t is computed from data <= t (rolling stats, ffilled macro
levels, regime). The forward-return *target* is deliberately kept out of the
feature matrix (see ``forward_log_return``) so it can never leak into training.

V1 features (yfinance-derivable only):
  trend   ret_{5,20,60,120}, ma_dist_{20,50,100,200}
  vol     yz_vol (Yang-Zhang), atr_pct, vol_chg20
  macro   cross-asset levels / returns / changes (per-symbol set)
  regime  ordinal code of the market regime (ES+VIX)

Excluded in V1 (-> V2): option-flow (dealer gamma / GEX / put-call; no history)
and FRED macro (2Y, real yield, breakeven, HY/IG OAS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FRED_SERIES, INSTRUMENTS, MACRO_SYMBOLS, data_loader, regime
from . import vol as volmod

RETURN_WINDOWS = (5, 20, 60, 120)
MA_WINDOWS = (20, 50, 100, 200)
YZ_WINDOW = 21
ATR_WINDOW = 14
VOL_CHG_WINDOW = 20
WARMUP = max(MA_WINDOWS) + 5  # rows before this lack the longest MA


# --------------------------------------------------------------------- blocks


def _trend_block(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for n in RETURN_WINDOWS:
        out[f"ret_{n}"] = np.log(close / close.shift(n))
    for n in MA_WINDOWS:
        ma = close.rolling(n, min_periods=n).mean()
        out[f"ma_dist_{n}"] = close / ma - 1.0
    return out


def _vol_block(ohlc: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ohlc.index)
    yz = volmod.yang_zhang_volatility(ohlc, window=YZ_WINDOW)
    out["yz_vol"] = yz
    out["vol_chg20"] = yz / yz.shift(VOL_CHG_WINDOW) - 1.0
    out["atr_pct"] = volmod.atr(ohlc, window=ATR_WINDOW) / ohlc["close"]
    return out


def _macro_block(symbol: str, idx: pd.DatetimeIndex, macros: dict[str, pd.Series], *, include_fred: bool = True) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)

    def lvl(key: str) -> pd.Series:
        return macros[key].reindex(idx, method="ffill")

    def ret(key: str, n: int) -> pd.Series:
        s = macros[key].reindex(idx, method="ffill").where(lambda x: x > 0)
        return np.log(s / s.shift(n))

    def chg(key: str, n: int) -> pd.Series:
        return macros[key].reindex(idx, method="ffill").diff(n)

    if symbol == "ES":
        out["vix_level"] = lvl("VIX")
        out["vix_chg5"] = chg("VIX", 5)
        out["vvix_level"] = lvl("VVIX")
        out["dxy_ret20"] = ret("DXY", 20)
        out["ust10y_level"] = lvl("UST10Y")
        out["ust10y_chg20"] = chg("UST10Y", 20)
        out["tip_ret20"] = ret("TIP", 20)
        out["oil_ret20"] = ret("OIL", 20)
        if include_fred:
            out["ust2y_level"] = lvl("UST2Y")
            out["ust2y_chg20"] = chg("UST2Y", 20)
            out["curve_2s10s_level"] = lvl("CURVE_2S10S")
            out["curve_2s10s_chg20"] = chg("CURVE_2S10S", 20)
            out["breakeven_chg20"] = chg("BREAKEVEN", 20)
            out["real_yield_chg20"] = chg("REAL_YIELD", 20)
            out["credit_hy_ig_ret20"] = ret("HYG", 20) - ret("LQD", 20)  # risk appetite
    elif symbol == "GC":
        out["ust10y_level"] = lvl("UST10Y")
        out["ust10y_chg20"] = chg("UST10Y", 20)
        out["tip_ret20"] = ret("TIP", 20)   # real-yield proxy
        out["dxy_ret20"] = ret("DXY", 20)
        out["oil_ret20"] = ret("OIL", 20)
        out["vix_level"] = lvl("VIX")
        if include_fred:
            out["real_yield_level"] = lvl("REAL_YIELD")   # key gold driver
            out["real_yield_chg20"] = chg("REAL_YIELD", 20)
            out["breakeven_level"] = lvl("BREAKEVEN")
            out["breakeven_chg20"] = chg("BREAKEVEN", 20)
            out["ust2y_level"] = lvl("UST2Y")
            out["ust2y_chg20"] = chg("UST2Y", 20)
    else:
        raise KeyError(f"no macro feature set defined for {symbol}")
    return out


# --------------------------------------------------------------------- public


def build_feature_matrix(
    symbol: str, *, dropna: bool = True, include_fred: bool = False, regime_mode: str = "auto",
    hmm_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Assemble the point-in-time feature matrix for an instrument (ES / GC).

    ``include_fred`` toggles the V1.1 FRED rates/inflation + credit features.
    Default OFF: the V1.1 study found they do not improve OOS IC and widen the
    IS-OOS gap (the strong-looking yield *levels* are non-stationary). Kept
    available behind the flag for further research (``model --compare-fred``).

    ``regime_mode`` selects the regime feature(s): ``"rule"`` -> a single
    ``regime_code`` (V1); ``"hmm"`` -> causal HMM filtered-state posteriors
    (V1.2, ``model --compare-regime``); ``"auto"`` -> the per-instrument default
    in ``INSTRUMENTS`` (ES: hmm, GC: rule).
    """
    if symbol not in INSTRUMENTS:
        raise KeyError(f"{symbol} not in INSTRUMENTS")
    if regime_mode == "auto":
        regime_mode = INSTRUMENTS[symbol].get("regime", "rule")
    ohlc = data_loader.load_ohlc_model(symbol)
    macro_keys = list(MACRO_SYMBOLS) + (list(FRED_SERIES) if include_fred else [])
    macros = {k: data_loader.load_close(k) for k in macro_keys}

    feats = pd.concat(
        [_trend_block(ohlc["close"]), _vol_block(ohlc)],
        axis=1,
    )
    feats = feats.join(_macro_block(symbol, feats.index, macros, include_fred=include_fred))

    # market-wide regime (ES + VIX), reindexed onto this instrument's calendar
    es_close, vix_close = data_loader.load_close("ES"), data_loader.load_close("VIX")
    if regime_mode == "hmm":
        hf = regime.hmm_features_for(symbol, **(hmm_kwargs or {})).reindex(feats.index, method="ffill")
        feats = feats.join(hf)
    elif regime_mode == "rule":
        reg = regime.classify(es_close, vix_close)
        feats["regime_code"] = regime.code_series(reg.reindex(feats.index, method="ffill"))
    else:
        raise ValueError(f"unknown regime_mode {regime_mode!r}")

    feats = feats.sort_index()
    if dropna:
        # require the longest-lookback trend feature so the matrix starts clean;
        # remaining sporadic NaNs (e.g. early VVIX) are left for LightGBM.
        feats = feats[feats[f"ma_dist_{max(MA_WINDOWS)}"].notna()]
    return feats


def feature_columns(symbol: str) -> list[str]:
    """Column names of the feature matrix (no target)."""
    return list(build_feature_matrix(symbol, dropna=True).columns)


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Target: forward log return over ``horizon`` sessions, aligned to entry t.

    target[t] = log(close[t+h] / close[t]). The last ``horizon`` rows are NaN
    (no future yet) and must be dropped before training.
    """
    close = close.astype(float).sort_index()
    return np.log(close.shift(-horizon) / close).rename(f"fwd_ret_{horizon}")
