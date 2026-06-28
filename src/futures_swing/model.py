"""Alpha model — LightGBM forward-return forecast with purged walk-forward CV.

The make-or-break risk here is **overlapping-return overfitting**: a 5D/10D
forward target makes consecutive labels highly autocorrelated, so the *effective*
sample size is ~ n_days / horizon (far smaller than the row count) while the
feature count is large. Guards baked in:

  * Purged + embargoed walk-forward: training rows whose label window
    [t, t+horizon] would overlap the test block are dropped, plus an extra
    ``embargo`` (>= horizon) gap. Train is always strictly before test.
  * Parsimony + regularization: shallow trees, high min_child_samples, strong
    L1/L2, low learning rate, subsampling (see ``DEFAULT_PARAMS``).
  * Overfit report: in-sample vs out-of-sample IC (rank correlation of forecast
    vs realized forward return). A large IS>>OOS gap means shrink the model.

Predictions are *expected forward log returns*; signal.py risk-adjusts them.

CLI:
    python -m futures_swing.model --symbol ES   # walk-forward metrics
    python -m futures_swing.model --symbol GC
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import INSTRUMENTS, features

# Regularized, parsimonious defaults (overridable via configs in V1.1).
DEFAULT_PARAMS = dict(
    objective="huber",        # robust to fat-tailed return outliers
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=15,
    max_depth=3,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

MIN_TRAIN = 1000   # ~4 trading years before the first test fold
TEST_SIZE = 252    # 1-year out-of-sample blocks


@dataclass
class WalkForwardResult:
    symbol: str
    horizon: int
    oos_pred: pd.Series                       # out-of-sample forecasts, by date
    oos_actual: pd.Series                     # realized forward returns, by date
    fold_metrics: pd.DataFrame                # per-fold IS/OOS IC + hit rate
    effective_n: float
    n_features: int
    feature_cols: list[str] = field(default_factory=list)

    @property
    def oos_ic(self) -> float:
        return _ic(self.oos_pred, self.oos_actual)

    @property
    def oos_hit(self) -> float:
        return _hit(self.oos_pred, self.oos_actual)

    @property
    def is_oos_gap(self) -> float:
        """Mean in-sample IC minus mean out-of-sample IC (overfit signal)."""
        return float(self.fold_metrics["is_ic"].mean() - self.fold_metrics["oos_ic"].mean())


# ------------------------------------------------------------------ metrics


def _ic(pred: pd.Series, actual: pd.Series) -> float:
    """Spearman rank IC between forecast and realized return."""
    mask = pred.notna() & actual.notna()
    if mask.sum() < 10:
        return float("nan")
    rho, _ = spearmanr(pred[mask], actual[mask])
    return float(rho)


def _hit(pred: pd.Series, actual: pd.Series) -> float:
    mask = pred.notna() & actual.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(actual[mask])).mean())


# ------------------------------------------------------------------ dataset


def make_dataset(symbol: str, *, include_fred: bool = False, regime_mode: str = "auto", hmm_kwargs: dict | None = None) -> tuple[pd.DataFrame, pd.Series, int]:
    """Aligned (X, y, horizon). Drops rows without a realized forward target;
    keeps feature NaNs (LightGBM handles them natively)."""
    horizon = INSTRUMENTS[symbol]["horizon"]
    X = features.build_feature_matrix(symbol, dropna=True, include_fred=include_fred, regime_mode=regime_mode, hmm_kwargs=hmm_kwargs)
    from . import data_loader

    close = data_loader.load_ohlc_model(symbol)["close"]
    y = features.forward_log_return(close, horizon).reindex(X.index)
    valid = y.notna()
    return X[valid], y[valid], horizon


# ------------------------------------------------------------------ CV folds


def purged_walk_forward_folds(
    n: int, *, horizon: int, embargo: int | None = None,
    min_train: int = MIN_TRAIN, test_size: int = TEST_SIZE,
) -> list[tuple[range, range]]:
    """Expanding-window folds with a purge+embargo gap between train and test.

    Train is positions [0, train_end); test is [start, start+test_size). The gap
    ``horizon + embargo`` removed from the end of train guarantees no training
    label window overlaps (or sits adjacent to) the test block.
    """
    embargo = horizon if embargo is None else embargo
    gap = horizon + embargo
    folds: list[tuple[range, range]] = []
    start = max(min_train, gap + 1)
    while start < n:
        train_end = start - gap
        if train_end <= min_train // 2:
            start += test_size
            continue
        test_end = min(start + test_size, n)
        folds.append((range(0, train_end), range(start, test_end)))
        start += test_size
    return folds


# ------------------------------------------------------------------ fit/predict


def _new_model(params: dict | None = None):
    from lightgbm import LGBMRegressor

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return LGBMRegressor(**p)


# --- per-symbol alpha dispatch (V1.4) -----------------------------------------
# The gap diagnosis showed the model class should be per-symbol: ES is a linear
# short-horizon mean-reversion problem (ridge on ret_5/ret_20; the 23-feature
# LightGBM dilutes the signal to noise), while GC carries real nonlinearity that
# only LightGBM captures. ``INSTRUMENTS[sym]["alpha"]`` selects kind + features.
DEFAULT_ALPHA = {"kind": "lgbm", "features": "all"}


def _resolve_alpha(symbol: str, override: dict | None) -> dict:
    if override is not None:
        return override
    return INSTRUMENTS[symbol].get("alpha", DEFAULT_ALPHA)


def _make_estimator(alpha_spec: dict, params: dict | None = None):
    """LightGBM (default) or a regularized linear pipeline (ridge)."""
    if alpha_spec.get("kind", "lgbm") == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("rdg", Ridge(alpha=alpha_spec.get("ridge_alpha", 10.0))),
        ])
    return _new_model(params)


def _select_features(X: pd.DataFrame, alpha_spec: dict) -> pd.DataFrame:
    feats = alpha_spec.get("features", "all")
    if feats == "all":
        return X
    keep = [c for c in feats if c in X.columns]
    if not keep:
        raise ValueError(f"alpha feature set {feats} not found in feature matrix")
    return X[keep]


def walk_forward(symbol: str, *, params: dict | None = None, embargo: int | None = None, include_fred: bool = False, regime_mode: str = "auto", hmm_kwargs: dict | None = None, alpha_spec: dict | None = None) -> WalkForwardResult:
    """Run purged walk-forward CV and collect OOS forecasts + IS/OOS metrics.

    The estimator + feature set come from the per-symbol alpha spec
    (``INSTRUMENTS[sym]["alpha"]``, overridable via ``alpha_spec``): ES uses a
    ridge mean-reversion sleeve, GC the full-feature LightGBM (V1.4)."""
    alpha = _resolve_alpha(symbol, alpha_spec)
    X, y, horizon = make_dataset(symbol, include_fred=include_fred, regime_mode=regime_mode, hmm_kwargs=hmm_kwargs)
    X = _select_features(X, alpha)
    folds = purged_walk_forward_folds(len(X), horizon=horizon, embargo=embargo)
    if not folds:
        raise RuntimeError(f"{symbol}: not enough history for any walk-forward fold")

    oos_pred = pd.Series(np.nan, index=X.index, dtype=float)
    rows = []
    for i, (tr, te) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop]
        X_te, y_te = X.iloc[te.start:te.stop], y.iloc[te.start:te.stop]
        model = _make_estimator(alpha, params)
        model.fit(X_tr, y_tr)
        pred_te = pd.Series(np.asarray(model.predict(X_te)), index=X_te.index)
        pred_tr = pd.Series(np.asarray(model.predict(X_tr)), index=X_tr.index)
        oos_pred.iloc[te.start:te.stop] = pred_te.to_numpy()
        rows.append(
            dict(
                fold=i,
                test_start=str(X.index[te.start].date()),
                test_end=str(X.index[te.stop - 1].date()),
                n_train=len(X_tr),
                n_test=len(X_te),
                is_ic=_ic(pred_tr, y_tr),
                oos_ic=_ic(pred_te, y_te),
                oos_hit=_hit(pred_te, y_te),
            )
        )
    return WalkForwardResult(
        symbol=symbol,
        horizon=horizon,
        oos_pred=oos_pred,
        oos_actual=y,
        fold_metrics=pd.DataFrame(rows),
        effective_n=len(X) / horizon,
        n_features=X.shape[1],
        feature_cols=list(X.columns),
    )


def fit_full(symbol: str, *, params: dict | None = None, embargo: int | None = None, alpha_spec: dict | None = None):
    """Train on all rows that have a realized target (minus the final purge gap),
    for live prediction. Returns (model, feature_cols, last_train_date)."""
    alpha = _resolve_alpha(symbol, alpha_spec)
    X, y, horizon = make_dataset(symbol)
    X = _select_features(X, alpha)
    embargo = horizon if embargo is None else embargo
    cut = len(X)  # all targets already realized; purge nothing extra at the tail
    model = _make_estimator(alpha, params)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    return model, list(X.columns), X.index[cut - 1]


def predict_latest(symbol: str, model, feature_cols: list[str]) -> tuple[pd.Timestamp, float]:
    """Forecast the most recent date's forward return (features available today)."""
    X = features.build_feature_matrix(symbol, dropna=True)[feature_cols]
    last_date = X.index[-1]
    pred = float(model.predict(X.iloc[[-1]])[0])
    return last_date, pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward CV for the alpha model")
    ap.add_argument("--symbol", required=True, choices=list(INSTRUMENTS))
    ap.add_argument("--embargo", type=int, default=None, help="extra gap days (default = horizon)")
    ap.add_argument("--compare-fred", action="store_true",
                    help="run with and without FRED features and report the OOS IC delta")
    ap.add_argument("--compare-regime", action="store_true",
                    help="run rule-based vs HMM regime and report the OOS IC delta")
    args = ap.parse_args()

    if args.compare_regime:
        rule = walk_forward(args.symbol, embargo=args.embargo, regime_mode="rule")
        hmm = walk_forward(args.symbol, embargo=args.embargo, regime_mode="hmm")
        print(f"\n=== {args.symbol} (horizon={rule.horizon}d) — regime: rule vs HMM ===")
        print(f"{'':20s} {'features':>9} {'OOS IC':>8} {'OOS hit':>8} {'IS-OOS gap':>11}")
        for label, r in (("rule regime", rule), ("HMM regime", hmm)):
            print(f"{label:20s} {r.n_features:>9d} {r.oos_ic:>+8.3f} {r.oos_hit:>8.3f} {r.is_oos_gap:>+11.3f}")
        print(f"{'Δ OOS IC':20s} {hmm.n_features - rule.n_features:>+9d} "
              f"{hmm.oos_ic - rule.oos_ic:>+8.3f} {hmm.oos_hit - rule.oos_hit:>+8.3f}")
        return

    if args.compare_fred:
        base = walk_forward(args.symbol, embargo=args.embargo, include_fred=False)
        full = walk_forward(args.symbol, embargo=args.embargo, include_fred=True)
        print(f"\n=== {args.symbol} (horizon={base.horizon}d) — FRED contribution ===")
        print(f"{'':20s} {'features':>9} {'OOS IC':>8} {'OOS hit':>8} {'IS-OOS gap':>11}")
        for label, r in (("V1 (no FRED)", base), ("V1.1 (+FRED)", full)):
            print(f"{label:20s} {r.n_features:>9d} {r.oos_ic:>+8.3f} {r.oos_hit:>8.3f} {r.is_oos_gap:>+11.3f}")
        print(f"{'Δ OOS IC':20s} {full.n_features - base.n_features:>+9d} "
              f"{full.oos_ic - base.oos_ic:>+8.3f} {full.oos_hit - base.oos_hit:>+8.3f}")
        return

    res = walk_forward(args.symbol, embargo=args.embargo)
    print(f"\n=== {res.symbol} (horizon={res.horizon}d) ===")
    print(f"rows={len(res.oos_actual)}  effective_N={res.effective_n:.0f}  features={res.n_features}")
    print(res.fold_metrics.to_string(index=False))
    print(f"\nOOS IC={res.oos_ic:+.3f}  OOS hit={res.oos_hit:.3f}  "
          f"IS-OOS IC gap={res.is_oos_gap:+.3f}  (large gap => overfit)")


if __name__ == "__main__":
    main()
