"""Point-in-time intraday feature frame for the gamma-regime MVP.

For each RTH 5-min ES bar at time t on session date D, using ONLY information
available by t:
  regime     sign of net dealer gamma from the most recent profile dated < D
  flip_es    that profile's zero-gamma level mapped into ES units (basis-robust %)
  dist_flip  (close - flip_es) / atr        — how far above/below the flip, in ATRs
  dist_vwap  (close - session_vwap) / atr    — extension from the session VWAP

LOOK-AHEAD GUARDS (we just got burned by one in the daily model):
  * gamma profile dated D reflects D's evening OI, so session D may use only a
    profile dated STRICTLY BEFORE D (prof_date < D) — never D's own.
  * VWAP/ATR are causal (cumsum / rolling over bars <= t, from data.py).
  * The ES->SPY basis is cancelled by mapping the flip as a RATIO to spot_ref,
    using the ES daily close on the profile date as the reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data_loader
from . import data as idata
from . import gamma as gmod


def _session_profile_map(session_dates, prof: pd.DataFrame, es_daily: pd.Series) -> pd.DataFrame:
    """For each session date D, attach the latest gamma profile dated < D and the
    ES reference close, mapping the SPY flip into ES price units."""
    rows = []
    pidx = prof.index
    for D in session_dates:
        prior = pidx[pidx < D]
        if len(prior) == 0:
            rows.append(dict(session_date=D, prof_date=pd.NaT, regime=np.nan, flip_es=np.nan))
            continue
        dprof = prior[-1]
        p = prof.loc[dprof]
        es_ref = es_daily.asof(dprof)                       # ES close on/before the profile date
        flip_es = (p["flip"] / p["spot_ref"]) * es_ref if pd.notna(p["flip"]) and es_ref else np.nan
        rows.append(dict(session_date=D, prof_date=dprof, regime=p["regime"], flip_es=flip_es))
    return pd.DataFrame(rows).set_index("session_date")


def build_features(key="ES", gamma_symbol="SPY", *, rth_only=True) -> pd.DataFrame:
    es = idata.add_session_features(idata.load_5m(key), rth_only=rth_only)
    prof = gmod.profile_frame(gamma_symbol)
    if prof.empty:
        raise RuntimeError(f"no gamma profiles for {gamma_symbol} — run gamma.ensure_profiles() first")
    es_daily = data_loader.load_ohlc(key)["close"]

    rth = es[es["rth"]].copy() if rth_only else es.copy()
    smap = _session_profile_map(sorted(rth["session_date"].unique()), prof, es_daily)
    rth = rth.join(smap[["prof_date", "regime", "flip_es"]], on="session_date")

    rth["dist_vwap"] = (rth["close"] - rth["vwap"]) / rth["atr"]
    rth["dist_flip"] = (rth["close"] - rth["flip_es"]) / rth["atr"]
    return rth


def coverage(key="ES", gamma_symbol="SPY") -> dict:
    """Quick feasibility read on how many usable session-days the free data gives."""
    f = build_features(key, gamma_symbol)
    by_day = f.dropna(subset=["regime"]).groupby("session_date")
    days = by_day.ngroups
    pos = sum(g["regime"].iloc[0] > 0 for _, g in by_day)
    flip_days = sum(g["flip_es"].notna().any() for _, g in by_day)
    return dict(usable_days=days, positive_gamma_days=pos, days_with_flip=flip_days,
                first=str(f.index.min()), last=str(f.index.max()), bars=len(f))
