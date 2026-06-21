"""Overfit-gap diagnostics for the alpha model (V1.3 follow-up study).

The walk-forward report flags a large IS-OOS IC gap (ES +0.335, GC +0.425) with
"=> overfit". This module rigorously decomposes that gap to separate two very
different questions the single number conflates:

  1. Is the **out-of-sample edge real** (distinguishable from zero)?
  2. Is the **gap harmful overfitting**, or just the expected IS>OOS artifact of
     a flexible learner fit on a small *effective* sample (n_days / horizon)?

A large IS-OOS gap is structurally guaranteed for any high-capacity learner — IS
IC is measured on the very rows the model fit — so the gap alone does NOT imply
the OOS forecast is broken. What matters is whether OOS IC survives honest
significance testing, and whether the IS IC is mostly noise-fitting capacity.

Experiments (all reproducible, seeded; folds are ~independent because the
purge+embargo gap separates each test block from its training set):

  fold       per-fold OOS IC mean / std / t-stat / two-sided p (fold-level test)
             + fraction of folds positive and a sign-test p
  bootstrap  circular block-bootstrap CI of the *pooled* OOS IC, block length =
             horizon, to respect overlapping-label autocorrelation
  capacity   OOS IC / IS IC / gap across model capacity (stump .. deep LightGBM)
             plus a linear Ridge baseline — does the gap shrink while OOS edge
             survives? (gap that collapses with capacity == benign capacity
             artifact, not a leak)
  null       shuffled-label control: refit on permuted targets to get the null
             distribution of IS IC (how much IS IC is *pure noise-fitting*) and
             of pooled OOS IC (leak check + empirical p for the real OOS IC)
  robust     drop-top-k folds — is the edge concentrated in a few lucky years?

CLI:
    python -m futures_swing.diagnostics --symbol GC --experiment all
    python -m futures_swing.diagnostics --symbol ES --experiment null --n-perm 50
    python -m futures_swing.diagnostics --symbol ES --experiment all --json reports/diagnostics/ES.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, t as student_t

from . import INSTRUMENTS, model

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- generic WF


def _lgbm_factory(params: dict | None):
    def make():
        return model._new_model(params)
    return make


def _ridge_factory(alpha: float = 10.0):
    """Linear baseline with median-impute + standardize (LightGBM eats NaNs
    natively; a linear model can't, so impute first)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def make():
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("rdg", Ridge(alpha=alpha)),
            ]
        )
    return make


def walk_forward_generic(
    symbol: str,
    make_estimator,
    *,
    regime_mode: str = "auto",
    include_fred: bool = False,
    embargo: int | None = None,
    y_override: np.ndarray | None = None,
):
    """Purged walk-forward with a pluggable estimator factory. Mirrors
    ``model.walk_forward`` but lets us swap LightGBM <-> Ridge and inject a
    permuted target (``y_override``) for the shuffled-label null."""
    X, y, horizon = model.make_dataset(symbol, include_fred=include_fred, regime_mode=regime_mode)
    if y_override is not None:
        y = pd.Series(y_override, index=y.index)
    folds = model.purged_walk_forward_folds(len(X), horizon=horizon, embargo=embargo)
    if not folds:
        raise RuntimeError(f"{symbol}: no walk-forward folds")
    oos_pred = pd.Series(np.nan, index=X.index, dtype=float)
    rows = []
    for i, (tr, te) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop]
        X_te, y_te = X.iloc[te.start:te.stop], y.iloc[te.start:te.stop]
        est = make_estimator()
        est.fit(X_tr, y_tr)
        pred_tr = pd.Series(np.asarray(est.predict(X_tr)), index=X_tr.index)
        pred_te = pd.Series(np.asarray(est.predict(X_te)), index=X_te.index)
        oos_pred.iloc[te.start:te.stop] = pred_te.to_numpy()
        rows.append(dict(fold=i, is_ic=model._ic(pred_tr, y_tr), oos_ic=model._ic(pred_te, y_te)))
    return oos_pred, y, pd.DataFrame(rows), horizon


# ------------------------------------------------------------------ fold test


def fold_stats(fold_metrics: pd.DataFrame) -> dict:
    """Treat each fold's OOS IC as one (approx-independent) observation and test
    whether the mean differs from zero."""
    oos = fold_metrics["oos_ic"].dropna().to_numpy()
    n = len(oos)
    mean = float(oos.mean())
    std = float(oos.std(ddof=1))
    se = std / np.sqrt(n)
    tstat = mean / se if se > 0 else np.nan
    p_t = float(2 * student_t.sf(abs(tstat), df=n - 1)) if np.isfinite(tstat) else np.nan
    n_pos = int((oos > 0).sum())
    p_sign = float(binomtest(n_pos, n, 0.5).pvalue)
    is_mean = float(fold_metrics["is_ic"].dropna().mean())
    return dict(
        n_folds=n, oos_ic_mean=mean, oos_ic_std=std, oos_ic_se=se, t_stat=float(tstat),
        p_two_sided=p_t, n_pos=n_pos, frac_pos=n_pos / n, p_sign=p_sign,
        is_ic_mean=is_mean, gap=is_mean - mean,
    )


# ------------------------------------------------------------------ bootstrap


def block_bootstrap_ic(pred: pd.Series, actual: pd.Series, horizon: int, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Circular block-bootstrap CI of pooled OOS IC. Block length = horizon so
    each resampled block carries one ~independent label, respecting the overlap
    that makes the naive 1/sqrt(N) SE too small."""
    mask = pred.notna() & actual.notna()
    p = pred[mask].to_numpy()
    a = actual[mask].to_numpy()
    n = len(p)
    L = max(int(horizon), 1)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / L))
    offs = np.arange(L)
    ics = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offs[None, :]).ravel() % n)[:n]
        rho, _ = spearmanr(p[idx], a[idx])
        ics[b] = rho
    return dict(
        point=float(model._ic(pred, actual)), boot_mean=float(np.nanmean(ics)),
        lo95=float(np.nanpercentile(ics, 2.5)), hi95=float(np.nanpercentile(ics, 97.5)),
        p_le0=float(np.mean(ics <= 0)), n_eff_blocks=n_blocks,
    )


# ------------------------------------------------------------------ capacity


CAPACITY_GRID = {
    "stump":   dict(num_leaves=2, max_depth=1, n_estimators=100, min_child_samples=200,
                    learning_rate=0.05, reg_alpha=1.0, reg_lambda=5.0),
    "shallow": None,  # == model.DEFAULT_PARAMS (the production model)
    "deep":    dict(num_leaves=127, max_depth=-1, n_estimators=600, min_child_samples=10,
                    learning_rate=0.05, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
                    colsample_bytree=1.0),
}


def capacity_sweep(symbol: str, *, regime_mode: str = "auto") -> pd.DataFrame:
    rows = []
    for name, params in CAPACITY_GRID.items():
        oos_pred, y, fm, horizon = walk_forward_generic(
            symbol, _lgbm_factory(params), regime_mode=regime_mode)
        rows.append(_cap_row(name, oos_pred, y, fm))
    # linear baseline
    oos_pred, y, fm, horizon = walk_forward_generic(symbol, _ridge_factory(), regime_mode=regime_mode)
    rows.append(_cap_row("ridge", oos_pred, y, fm))
    return pd.DataFrame(rows)


def _cap_row(name: str, oos_pred: pd.Series, y: pd.Series, fm: pd.DataFrame) -> dict:
    oos_ic = float(model._ic(oos_pred, y))
    is_ic = float(fm["is_ic"].mean())
    fold_oos = float(fm["oos_ic"].mean())
    return dict(model=name, oos_ic_pooled=oos_ic, oos_ic_foldmean=fold_oos,
                is_ic=is_ic, gap=is_ic - fold_oos)


# ------------------------------------------------------------------ null ctrl


def shuffled_null(symbol: str, *, n_perm: int = 50, seed: int = 0, regime_mode: str = "auto") -> dict:
    """Refit on permuted targets. The null IS IC is how much in-sample IC this
    model's capacity manufactures from PURE NOISE; the null pooled-OOS IC band
    is a leak check (should straddle 0) and gives an empirical p for the real
    OOS IC."""
    real_oos_pred, real_y, real_fm, horizon = walk_forward_generic(
        symbol, _lgbm_factory(None), regime_mode=regime_mode)
    real_oos = float(model._ic(real_oos_pred, real_y))
    real_is = float(real_fm["is_ic"].mean())

    rng = np.random.default_rng(seed)
    yv = real_y.to_numpy()
    null_oos = np.empty(n_perm)
    null_is = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(yv)
        oos_pred, y_p, fm, _ = walk_forward_generic(
            symbol, _lgbm_factory(None), regime_mode=regime_mode, y_override=perm)
        null_oos[k] = model._ic(oos_pred, y_p)
        null_is[k] = fm["is_ic"].mean()
    p_oos = float((null_oos >= real_oos).mean())
    return dict(
        real_oos_ic=real_oos, real_is_ic=real_is,
        null_oos_mean=float(null_oos.mean()), null_oos_std=float(null_oos.std(ddof=1)),
        null_oos_lo=float(np.percentile(null_oos, 2.5)), null_oos_hi=float(np.percentile(null_oos, 97.5)),
        null_is_mean=float(null_is.mean()), null_is_std=float(null_is.std(ddof=1)),
        p_oos_empirical=p_oos, n_perm=n_perm,
        is_ic_noise_share=float(null_is.mean() / real_is) if real_is else np.nan,
    )


# ------------------------------------------------------------------ robustness


def drop_top_k(fold_metrics: pd.DataFrame, ks=(0, 1, 2, 3, 5)) -> pd.DataFrame:
    oos = fold_metrics["oos_ic"].dropna().sort_values(ascending=False)
    rows = []
    for k in ks:
        kept = oos.iloc[k:]
        rows.append(dict(dropped_best=k, n=len(kept), mean_oos_ic=float(kept.mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ runner


def run_all(symbol: str, *, n_perm: int = 50, n_boot: int = 2000, seed: int = 0) -> dict:
    res = model.walk_forward(symbol)
    fs = fold_stats(res.fold_metrics)
    bb = block_bootstrap_ic(res.oos_pred, res.oos_actual, res.horizon, n_boot=n_boot, seed=seed)
    cap = capacity_sweep(symbol)
    nul = shuffled_null(symbol, n_perm=n_perm, seed=seed)
    rob = drop_top_k(res.fold_metrics)
    return dict(
        symbol=symbol, horizon=res.horizon, effective_n=res.effective_n,
        n_features=res.n_features,
        fold=fs, bootstrap=bb,
        capacity=cap.to_dict(orient="records"),
        null=nul, robust=rob.to_dict(orient="records"),
        per_fold_oos_ic=[None if pd.isna(v) else float(v) for v in res.fold_metrics["oos_ic"]],
    )


def _print_report(out: dict) -> None:
    s = out["symbol"]
    print(f"\n================ {s} (horizon={out['horizon']}d, eff_N={out['effective_n']:.0f}, "
          f"feats={out['n_features']}) ================")
    f = out["fold"]
    print("\n[fold-level test]  (each fold's OOS IC = one obs)")
    print(f"  folds={f['n_folds']}  OOS IC mean={f['oos_ic_mean']:+.4f}  std={f['oos_ic_std']:.4f}  "
          f"se={f['oos_ic_se']:.4f}")
    print(f"  t={f['t_stat']:+.2f}  p(two-sided)={f['p_two_sided']:.3f}  | "
          f"pos {f['n_pos']}/{f['n_folds']} ({f['frac_pos']:.0%})  sign-p={f['p_sign']:.3f}")
    print(f"  IS IC mean={f['is_ic_mean']:+.3f}  gap={f['gap']:+.3f}")
    b = out["bootstrap"]
    print("\n[block bootstrap]  pooled OOS IC, block=horizon")
    print(f"  point={b['point']:+.4f}  95% CI=[{b['lo95']:+.4f}, {b['hi95']:+.4f}]  "
          f"P(IC<=0)={b['p_le0']:.3f}")
    print("\n[capacity sweep]")
    print(f"  {'model':8s} {'OOS(pool)':>10s} {'OOS(fold)':>10s} {'IS':>8s} {'gap':>8s}")
    for r in out["capacity"]:
        print(f"  {r['model']:8s} {r['oos_ic_pooled']:>+10.4f} {r['oos_ic_foldmean']:>+10.4f} "
              f"{r['is_ic']:>+8.3f} {r['gap']:>+8.3f}")
    n = out["null"]
    print("\n[shuffled-label null]  (model capacity on pure noise)")
    print(f"  real:  IS IC={n['real_is_ic']:+.3f}   OOS IC={n['real_oos_ic']:+.4f}")
    print(f"  null:  IS IC={n['null_is_mean']:+.3f}±{n['null_is_std']:.3f}   "
          f"OOS IC={n['null_oos_mean']:+.4f}±{n['null_oos_std']:.4f} "
          f"(95% [{n['null_oos_lo']:+.4f},{n['null_oos_hi']:+.4f}])")
    print(f"  => {n['is_ic_noise_share']:.0%} of IS IC is pure noise-fitting; "
          f"empirical p(OOS) = {n['p_oos_empirical']:.3f}")
    print("\n[robustness — drop best-k folds]")
    for r in out["robust"]:
        print(f"  drop {r['dropped_best']}: mean OOS IC={r['mean_oos_ic']:+.4f} (n={r['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Overfit-gap diagnostics")
    ap.add_argument("--symbol", required=True, choices=list(INSTRUMENTS))
    ap.add_argument("--experiment", default="all",
                    choices=["all", "fold", "bootstrap", "capacity", "null", "robust"])
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None, help="write full results JSON here")
    args = ap.parse_args()

    if args.experiment == "all":
        out = run_all(args.symbol, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
        _print_report(out)
    else:
        res = model.walk_forward(args.symbol)
        out = {"symbol": args.symbol, "horizon": res.horizon}
        if args.experiment == "fold":
            out["fold"] = fold_stats(res.fold_metrics)
            print(json.dumps(out["fold"], indent=2))
        elif args.experiment == "bootstrap":
            out["bootstrap"] = block_bootstrap_ic(res.oos_pred, res.oos_actual, res.horizon,
                                                  n_boot=args.n_boot, seed=args.seed)
            print(json.dumps(out["bootstrap"], indent=2))
        elif args.experiment == "capacity":
            print(capacity_sweep(args.symbol).to_string(index=False))
        elif args.experiment == "null":
            out["null"] = shuffled_null(args.symbol, n_perm=args.n_perm, seed=args.seed)
            print(json.dumps(out["null"], indent=2))
        elif args.experiment == "robust":
            print(drop_top_k(res.fold_metrics).to_string(index=False))

    if args.json:
        full = out if args.experiment == "all" else run_all(
            args.symbol, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(full, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
