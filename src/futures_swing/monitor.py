"""Live forward-tracking monitor — the 'track before you trade' layer.

Before risking money on the PRE-REGISTERED strategies (configs/registrations/),
run this daily to (1) emit today's action from the *validated V1.6 strategy*
(strategy.live_signal — the same design forward_validation tracks, NOT the older
V1.4 pipeline), (2) append it to an append-only live log (the forward record),
and (3) score realized post-freeze performance against the pre-registered
block-bootstrap band with a kill-switch verdict. Until ~6 months of post-freeze
data accrue the verdict is "too-early" — the honest state right after registering.

CLI:
    python -m futures_swing.monitor               # dashboard
    python -m futures_swing.monitor --log         # also append today's action to the live log
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import strategy
from .forward_validation import REG_DIR, evaluate

REPO = Path(__file__).resolve().parents[2]
LIVE_LOG = REPO / "data" / "signals" / "live_log.csv"


def append_live_log(rows: list[dict]) -> None:
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if LIVE_LOG.exists():
        df = pd.concat([pd.read_csv(LIVE_LOG), df], ignore_index=True).drop_duplicates(
            ["asof", "symbol"], keep="last")
    df.to_csv(LIVE_LOG, index=False)


def status(symbols, as_of, do_log):
    sigs = [strategy.live_signal(s) for s in symbols]
    if do_log:
        append_live_log(sigs)

    print("\n================ LIVE ACTION (pre-registered V1.6 strategy) ================")
    print(f"{'symbol':6s} {'asof':12s} {'pos_now':>8s} {'today':>14s} {'pos_after':>10s} {'sharpe':>7s} {'exit':>7s}")
    for s in sigs:
        print(f"{s['symbol']:6s} {s['asof']:12s} {s['position_now']:>8d} {s['pending_action']:>14s} "
              f"{s['position_after']:>10d} {s['sharpe']:>+7.3f} {s['exit_rule']:>7s}")

    print("\n================ FORWARD TRACKING (vs pre-registered band) ================")
    for sym in symbols:
        rp = REG_DIR / f"{sym}.json"
        if not rp.exists():
            print(f"  {sym}: not registered"); continue
        reg = json.loads(rp.read_text()); ev = evaluate(sym, as_of)
        print(f"  {sym}: frozen {reg['freeze_date']} | baseline {reg['baseline_sharpe']:+.2f} "
              f"(band low {reg['expected_sharpe_90pct'][0]:+.2f}) | forward {ev['fwd_days']}d "
              f"→ live {ev['live_sharpe']:+.2f}  ⇒  {ev['verdict']}")
    print("\n(kill rule: live Sharpe below the band low for 6+ months → investigate / pull.)")


def main():
    ap = argparse.ArgumentParser(description="Live forward-tracking monitor")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"], choices=["ES", "GC"])
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    as_of = args.as_of or str(pd.Timestamp.today().date())
    status(args.symbols, as_of, args.log)
    if args.log:
        print(f"\nlogged to {LIVE_LOG}")


if __name__ == "__main__":
    main()
