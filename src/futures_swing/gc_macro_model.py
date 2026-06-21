"""Candidate GC redesign — a LONG-HORIZON (60d) LINEAR (ridge) macro model.

Feature mining showed GC's macro drivers express LINEARLY over ~60 days, not at
the current 10d horizon (where only weak price-reversion exists). A ridge on
*stationary* gold-driver features at h60 reached OOS IC ~+0.12 (vs the 10d
LightGBM's +0.070). This builds + validates that candidate. STATIONARY features
only (z-scores / changes — avoid the V1.1 non-stationary-level trap). Mirror of
ES: both linear, opposite time scale (ES short reversion / GC long macro).

A-priori gold-driver feature set (NOT cherry-picked from the IC table):
  realyield_z   10Y TIPS real yield (DFII10) causal z-score   — the key gold driver
  dxy_z         US dollar index causal z-score                — inverse driver
  breakeven_chg 60d change in 10Y breakeven inflation          — inflation expectations
  gc_ext        z-score of close/SMA200-1                      — price extension / reversion
  gc_mom        z-score of 120d return                         — slow momentum / reversion
  gc_vol        Yang-Zhang realized vol                        — vol regime

NOT production — pending forward validation.

CLI:  python -m futures_swing.gc_macro_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data_loader, model
from . import vol as volmod
from .diagnostics import block_bootstrap_ic

SYM = "GC"
HORIZON = 60
Z = 252
RIDGE_ALPHA = 10.0


def _z(s, w=Z):
    return (s - s.rolling(w, min_periods=60).mean()) / s.rolling(w, min_periods=60).std()


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    ohlc = data_loader.load_ohlc(SYM)
    close = ohlc["close"]
    idx = close.index

    def S(key):
        return data_loader.load_close(key).reindex(idx, method="ffill")

    RY, DXY, BE = S("REAL_YIELD"), S("DXY"), S("BREAKEVEN")
    X = pd.DataFrame(index=idx)
    X["realyield_z"] = _z(RY)
    X["dxy_z"] = _z(DXY)
    X["breakeven_chg60"] = BE.diff(60)
    X["gc_ext"] = _z(close / close.rolling(200, min_periods=200).mean() - 1.0)
    X["gc_mom"] = _z(np.log(close / close.shift(120)))
    X["gc_vol"] = volmod.yang_zhang_volatility(ohlc, window=21)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X[X[["realyield_z", "dxy_z", "gc_ext", "gc_mom"]].notna().all(axis=1)]  # require core feats
    y = np.log(close.shift(-HORIZON) / close).reindex(X.index).rename("fwd")
    valid = y.notna()
    return X[valid], y[valid]


def _ridge():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("rdg", Ridge(alpha=RIDGE_ALPHA))])


def walk_forward():
    X, y = build_features()
    folds = model.purged_walk_forward_folds(len(X), horizon=HORIZON)
    oos = pd.Series(np.nan, index=X.index, dtype=float)
    rows = []
    for tr, te in folds:
        m = _ridge()
        m.fit(X.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop])
        p_te = pd.Series(m.predict(X.iloc[te.start:te.stop]), index=X.index[te.start:te.stop])
        oos.iloc[te.start:te.stop] = p_te.to_numpy()
        rows.append(dict(test_start=str(X.index[te.start].date()),
                         oos_ic=model._ic(p_te, y.iloc[te.start:te.stop])))
    return X, y, oos, pd.DataFrame(rows)


def main():
    X, y, oos, fm = walk_forward()
    fic = fm["oos_ic"].dropna().to_numpy()
    pooled = model._ic(oos, y)
    bb = block_bootstrap_ic(oos, y, HORIZON, n_boot=3000, seed=0)
    tstat = fic.mean() / (fic.std(ddof=1) / np.sqrt(len(fic)))
    # sub-period (forward check): early vs late half
    mid = oos.index[len(oos) // 2]
    ic_e = model._ic(oos[oos.index < mid], y[oos.index < mid])
    ic_l = model._ic(oos[oos.index >= mid], y[oos.index >= mid])

    print(f"\n=== GC h{HORIZON} ridge macro model ({X.shape[1]} stationary features, "
          f"{len(X)} rows, eff_N~{len(X)//HORIZON}) ===")
    print(f"features: {list(X.columns)}")
    print(f"\npooled OOS IC = {pooled:+.4f}   (current GC: 10d lgbm +0.070)")
    print(f"  block-bootstrap 95% CI = [{bb['lo95']:+.4f}, {bb['hi95']:+.4f}]  P(IC<=0) = {bb['p_le0']:.3f}")
    print(f"  per-fold: mean {fic.mean():+.3f}, t = {tstat:+.2f}, {len(fic)} folds, {(fic>0).mean()*100:.0f}% positive")
    print(f"\nforward/sub-period stability (the key generalization check):")
    print(f"  early-half OOS IC = {ic_e:+.4f}   late-half OOS IC = {ic_l:+.4f}")
    print("\nper-fold OOS IC:")
    print(fm.to_string(index=False))


if __name__ == "__main__":
    main()
