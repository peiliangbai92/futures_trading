"""Forward (out-of-time) validation for the V1.6 strategies.

The forecasts are purged walk-forward OOS (honest), but the strategy's RULE
PARAMETERS (thresholds, cooldown, trail, breakout) were chosen in-sample on
2005-2026. This module guards against that two ways:

  anchored_walk  — the rigorous generalization test runnable NOW. At each anchor
    date, re-select the rule params on data <= anchor (the params never see the
    forward window) and evaluate the NEXT window flat-start. If forward Sharpe
    tracks in-sample Sharpe, the rules generalize; if it collapses, they were
    overfit. Reported alongside the FIXED V1.6 params for comparison.
  register / evaluate — pre-registration for live use: freeze the spec + a date +
    a block-bootstrap expectation band; as new data arrives, measure realized
    performance ONLY on dates after the freeze and flag a kill-switch.

CLI:
    python -m futures_swing.forward_validation --anchored --symbols ES GC
    python -m futures_swing.forward_validation --register
    python -m futures_swing.forward_validation --evaluate --as-of 2026-06-19
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import INSTRUMENTS, data_loader, model, signal
from .strategy import DESIGN, simulate

REPO = Path(__file__).resolve().parents[2]
REG_DIR = REPO / "configs" / "registrations"
TD = 252

# Per-symbol rule grids for re-selection (a priori, modest).
GRID = {
    "ES": [dict(buy_th=bt, cooldown=cd, sell="mom20", breakout=0)
           for bt in (0.12, 0.15, 0.20, 0.25) for cd in (10, 15, 20)],
    "GC": [dict(buy_th=bt, cooldown=cd, sell="trail", trail_drop=dr, trail_win=60, breakout=bk)
           for bt in (0.20, 0.30, 0.40) for cd in (15, 20) for dr in (0.06, 0.08, 0.10) for bk in (0, 40)],
}


def forecast_sharpe(symbol: str) -> pd.Series:
    """Compute the OOS forecast sharpe ONCE (the expensive walk-forward call)."""
    close = data_loader.load_ohlc(symbol)["close"]
    hzn = INSTRUMENTS[symbol]["horizon"]
    pred = model.walk_forward(symbol).oos_pred
    fc = signal.horizon_forecast_vol(close, hzn).reindex(pred.index)
    return (pred / fc).replace([np.inf, -np.inf], np.nan)


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(TD)) if len(r) > 5 and r.std() > 0 else float("nan")


def returns(symbol, cfg, shp, *, trade_start=None, lo=None, hi=None) -> pd.Series:
    _, eq, _ = simulate(symbol, cfg, sharpe_override=shp, trade_start=trade_start)
    r = eq.pct_change()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index <= pd.Timestamp(hi)]
    return r


def select_params(symbol, shp, train_end) -> dict:
    """Pick the grid cfg with the best in-sample Sharpe on data <= train_end."""
    best, best_s = DESIGN[symbol], -1e9
    for cfg in GRID[symbol]:
        s = _sharpe(returns(symbol, cfg, shp, hi=train_end))
        if np.isfinite(s) and s > best_s:
            best, best_s = cfg, s
    return best


def anchored_walk(symbol, anchors) -> list[dict]:
    shp = forecast_sharpe(symbol)
    end = shp.dropna().index[-1]
    bounds = list(anchors) + [end]
    rows = []
    for a, nxt in zip(bounds[:-1], bounds[1:]):
        sel = select_params(symbol, shp, a)
        is_s = _sharpe(returns(symbol, sel, shp, hi=a))                       # in-sample (selected)
        fwd_sel = _sharpe(returns(symbol, sel, shp, trade_start=a, lo=a, hi=nxt))   # re-selected, forward
        fwd_fix = _sharpe(returns(symbol, DESIGN[symbol], shp, trade_start=a, lo=a, hi=nxt))  # fixed V1.6, forward
        rows.append(dict(anchor=str(pd.Timestamp(a).date()), to=str(pd.Timestamp(nxt).date()),
                         is_sharpe=is_s, fwd_sharpe_reselected=fwd_sel, fwd_sharpe_fixed=fwd_fix,
                         selected=sel))
    return rows


# ----------------------------------------------------- pre-registration (live)


def _block_bootstrap_sharpe(r: pd.Series, n=2000, block=20, seed=0):
    r = r.dropna().to_numpy()
    if len(r) < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(r) / block)); out = []
    for _ in range(n):
        st = rng.integers(0, len(r), size=nb)
        idx = ((st[:, None] + np.arange(block)[None, :]).ravel() % len(r))[:len(r)]
        x = r[idx]; out.append(x.mean() / x.std() * np.sqrt(TD) if x.std() > 0 else 0.0)
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def register(symbol, freeze_date=None) -> dict:
    shp = forecast_sharpe(symbol)
    freeze = pd.Timestamp(freeze_date) if freeze_date else shp.dropna().index[-1]
    r = returns(symbol, DESIGN[symbol], shp, hi=freeze)
    lo, hi = _block_bootstrap_sharpe(r)
    reg = dict(symbol=symbol, frozen_spec=DESIGN[symbol], freeze_date=str(freeze.date()),
               baseline_sharpe=_sharpe(r), expected_sharpe_90pct=[lo, hi],
               note="forecasts OOS; rule params in-sample. Evaluate realized Sharpe on dates "
                    "AFTER freeze_date; kill if it sits below the lower band for 6+ months.")
    REG_DIR.mkdir(parents=True, exist_ok=True)
    (REG_DIR / f"{symbol}.json").write_text(json.dumps(reg, indent=2))
    return reg


def evaluate(symbol, as_of) -> dict:
    reg = json.loads((REG_DIR / f"{symbol}.json").read_text())
    shp = forecast_sharpe(symbol)
    r = returns(symbol, reg["frozen_spec"], shp, trade_start=reg["freeze_date"],
                lo=reg["freeze_date"], hi=as_of)
    live = _sharpe(r); lo = reg["expected_sharpe_90pct"][0]
    n = r.dropna().shape[0]
    verdict = ("too-early (need ~6mo)" if n < 120 else
               "ON TRACK" if live >= lo else "BELOW BAND — investigate / kill")
    return dict(symbol=symbol, freeze_date=reg["freeze_date"], as_of=as_of, fwd_days=n,
                live_sharpe=live, expected_low=lo, baseline_sharpe=reg["baseline_sharpe"], verdict=verdict)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward (out-of-time) validation")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"], choices=list(DESIGN))
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    if args.anchored:
        anchors = ["2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]
        for sym in args.symbols:
            print(f"\n=== {sym}: anchored forward validation (re-select rules on data<=anchor) ===")
            print(f"{'window':24s} {'IS Sharpe':>10s} {'FWD(reselect)':>14s} {'FWD(fixedV1.6)':>15s}")
            rows = anchored_walk(sym, anchors)
            for r in rows:
                print(f"{r['anchor']}..{r['to']:10s} {r['is_sharpe']:>+10.2f} "
                      f"{r['fwd_sharpe_reselected']:>+14.2f} {r['fwd_sharpe_fixed']:>+15.2f}")
            fwd = np.array([r["fwd_sharpe_reselected"] for r in rows if np.isfinite(r["fwd_sharpe_reselected"])])
            fix = np.array([r["fwd_sharpe_fixed"] for r in rows if np.isfinite(r["fwd_sharpe_fixed"])])
            print(f"  mean forward Sharpe: re-selected {fwd.mean():+.2f} | fixed-V1.6 {fix.mean():+.2f}")

    if args.register:
        for sym in args.symbols:
            reg = register(sym)
            print(f"registered {sym}: freeze {reg['freeze_date']}, baseline Sharpe {reg['baseline_sharpe']:+.2f}, "
                  f"90% band [{reg['expected_sharpe_90pct'][0]:+.2f},{reg['expected_sharpe_90pct'][1]:+.2f}] "
                  f"-> {REG_DIR / (sym + '.json')}")

    if args.evaluate:
        for sym in args.symbols:
            print(json.dumps(evaluate(sym, args.as_of or "2026-06-19"), indent=2))


if __name__ == "__main__":
    main()
