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
# Tracked path (not data/signals, which is gitignored) so the daily GitHub Actions
# run can commit the forward record back to the repo.
LIVE_LOG = REPO / "tracking" / "live_log.csv"


def append_live_log(rows: list[dict]) -> None:
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if LIVE_LOG.exists():
        df = pd.concat([pd.read_csv(LIVE_LOG), df], ignore_index=True).drop_duplicates(
            ["asof", "symbol"], keep="last")
    df.to_csv(LIVE_LOG, index=False)


def render_issue(sigs: list[dict]) -> tuple[str, str] | None:
    """If any symbol signals a tradeable action today (not HOLD), render a GitHub
    issue (title, body) for the daily notifier. All-HOLD days return None (quiet)."""
    actionable = [s for s in sigs if s["pending_action"] != "HOLD"]
    if not actionable:
        return None
    asof = max(s["asof"] for s in sigs)
    head = ", ".join(f"{s['symbol']} {s['pending_action']}" for s in actionable)
    title = f"Trade signal {asof}: {head}"
    rows = ["| symbol | action | pos now → after | sharpe | exit |",
            "|---|---|---|---|---|"]
    for s in sigs:
        b = "**" if s["pending_action"] != "HOLD" else ""
        rows.append(f"| {b}{s['symbol']}{b} | {b}{s['pending_action']}{b} | "
                    f"{s['position_now']} → {s['position_after']} | {s['sharpe']:+.3f} | {s['exit_rule']} |")
    body = (f"**As of {asof}** — pre-registered V1.6 strategy. Action executes at the "
            f"**next session open** (1 micro lot per buy, max 2 lots).\n\n" + "\n".join(rows) +
            "\n\n_The model's mechanical signal — verify before placing. Sized for ~$220k equity; "
            "forward-tracking is still in its early window, so treat as a heads-up, not a mandate._")
    return title, body


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
    return sigs


def main():
    ap = argparse.ArgumentParser(description="Live forward-tracking monitor")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"], choices=["ES", "GC"])
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--json", default=None, help="write today's signals to this JSON path")
    ap.add_argument("--issue-dir", default=None,
                    help="if a tradeable action fires, write issue_title.txt + issue_body.md here")
    args = ap.parse_args()
    as_of = args.as_of or str(pd.Timestamp.today().date())
    sigs = status(args.symbols, as_of, args.log)

    if args.json:
        p = Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sigs, indent=2))
    if args.issue_dir:
        d = Path(args.issue_dir); d.mkdir(parents=True, exist_ok=True)
        rendered = render_issue(sigs)
        if rendered:
            (d / "issue_title.txt").write_text(rendered[0])
            (d / "issue_body.md").write_text(rendered[1])
            print(f"\nALERT — tradeable action; issue files written to {d}")
        else:
            print("\nquiet — all HOLD, no issue rendered")
    if args.log:
        print(f"\nlogged to {LIVE_LOG}")


if __name__ == "__main__":
    main()
