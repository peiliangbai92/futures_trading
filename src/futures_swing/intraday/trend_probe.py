"""Feasibility PROBE — negative-gamma intraday trend (H1, trend half).

H1: on NEGATIVE net-gamma days dealer hedging amplifies moves → intraday momentum.
We isolate the directional claim cleanly: entries fire at the SAME extension events
(|dist_vwap| >= k_entry); only the DIRECTION differs —
    momentum : trade WITH the extension (long when above VWAP)   <- H1 prediction
    fade     : trade AGAINST it
    random   : random sign (null)
If H1 holds on negative-gamma days, momentum should beat fade and the random null.

Mechanics (point-in-time, no look-ahead):
  * signal computed from bar i (close_i, vwap_i); EXECUTED at open_{i+1}.
  * exit on VWAP reclaim OR ATR stop OR end-of-day; flat by the last RTH bar.
  * costs charged per round-trip in ES points.
  * only sessions whose D-1 gamma profile is negative are traded.

~16 negative-gamma days of free data → a FEASIBILITY read (tens of trades), not a
validation. Numbers are directional; do not size on them.

CLI:
    python -m futures_swing.intraday.trend_probe
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import features as feat

PT_VALUE = 50.0          # ES = $50/point (full); MES = $5
ENTRY_WIN = (600, 900)   # only open new trades 10:00-15:00 ET (minutes since midnight)
FLAT_MIN = 955           # force flat at/after 15:55 ET


def _session_pnl(day: pd.DataFrame, direction, k_entry, tp_atr, k_stop, cost_pts, rng) -> list[float]:
    o = day["open"].to_numpy(float); c = day["close"].to_numpy(float)
    vw = day["vwap"].to_numpy(float); atr = day["atr"].to_numpy(float)
    tmin = (day.index.hour * 60 + day.index.minute).to_numpy()
    n = len(day); pos = 0; entry = entry_atr = 0.0; trades = []
    for i in range(n):
        if not np.isfinite(vw[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        dv = (c[i] - vw[i]) / atr[i]
        if pos == 0:
            if ENTRY_WIN[0] <= tmin[i] < ENTRY_WIN[1] and i + 1 < n and abs(dv) >= k_entry:
                raw = 1 if dv > 0 else -1                      # extension direction
                side = {"momentum": raw, "fade": -raw,
                        "random": (1 if rng.random() < 0.5 else -1)}[direction]
                pos = side; entry = o[i + 1]; entry_atr = atr[i]
        else:
            # SYMMETRIC exit (direction-agnostic so momentum vs fade is a fair test):
            # ATR take-profit / stop bracket, else flat by end of day.
            move = (c[i] - entry) * pos
            hit = move >= tp_atr * entry_atr or move <= -k_stop * entry_atr
            eod = tmin[i] >= FLAT_MIN
            if hit and i + 1 < n:
                trades.append((o[i + 1] - entry) * pos - cost_pts); pos = 0
            elif eod:
                px = o[i + 1] if i + 1 < n else c[i]
                trades.append((px - entry) * pos - cost_pts); pos = 0
    if pos != 0:                                              # safety: close at last close
        trades.append((c[-1] - entry) * pos - cost_pts)
    return trades


def simulate(features_df, *, direction="momentum", k_entry=0.7, tp_atr=1.0, k_stop=1.0,
             cost_pts=0.5, regime=-1, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    days = features_df[features_df["regime"] == regime]
    all_t, day_pnl = [], []
    for _, day in days.groupby("session_date"):
        t = _session_pnl(day.sort_index(), direction, k_entry, tp_atr, k_stop, cost_pts, rng)
        all_t += t; day_pnl.append(sum(t))
    a = np.array(all_t)
    dp = np.array(day_pnl)
    sharpe = float(dp.mean() / dp.std() * np.sqrt(252)) if len(dp) > 2 and dp.std() > 0 else float("nan")
    return dict(direction=direction, n_days=len(dp), n_trades=len(a),
                total_pts=float(a.sum()), avg_pts=float(a.mean()) if len(a) else float("nan"),
                win=float((a > 0).mean()) if len(a) else float("nan"),
                total_usd=float(a.sum() * PT_VALUE), day_sharpe=sharpe)


def run_probe(k_entry=0.7, tp_atr=1.0, k_stop=1.0, cost_pts=0.5, n_null=200) -> pd.DataFrame:
    f = feat.build_features("ES", "SPY")
    rows = []
    for d in ("momentum", "fade"):
        rows.append(simulate(f, direction=d, k_entry=k_entry, tp_atr=tp_atr, k_stop=k_stop, cost_pts=cost_pts))
    # random-direction null: distribution over seeds, same entry events
    nulls = [simulate(f, direction="random", k_entry=k_entry, tp_atr=tp_atr, k_stop=k_stop,
                      cost_pts=cost_pts, seed=s)["total_pts"] for s in range(n_null)]
    rows.append(dict(direction="random(null)", n_days=rows[0]["n_days"], n_trades=rows[0]["n_trades"],
                     total_pts=float(np.mean(nulls)), avg_pts=float("nan"),
                     win=float("nan"), total_usd=float(np.mean(nulls) * PT_VALUE),
                     day_sharpe=float("nan"),
                     null_p5=float(np.percentile(nulls, 5)), null_p95=float(np.percentile(nulls, 95))))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Negative-gamma intraday trend probe")
    ap.add_argument("--k-entry", type=float, default=0.7)
    ap.add_argument("--tp-atr", type=float, default=1.0)
    ap.add_argument("--k-stop", type=float, default=1.0)
    ap.add_argument("--cost-pts", type=float, default=0.5)
    args = ap.parse_args()
    df = run_probe(args.k_entry, args.tp_atr, args.k_stop, args.cost_pts)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"\n=== negative-gamma intraday TREND probe (k_entry={args.k_entry}, "
          f"tp={args.tp_atr} stop={args.k_stop} cost={args.cost_pts}pt) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFEASIBILITY ONLY — ~16 days / tens of trades. H1 favored if momentum > fade and > null.")


if __name__ == "__main__":
    main()
