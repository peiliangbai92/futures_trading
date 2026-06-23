"""Layer-2 risk circuit breaker — halt NEW entries (only-close) when the live
book deteriorates, FASTER than the slow forward-Sharpe kill-switch (which needs
~6 months of overlapping returns to confirm edge decay).

It runs on YOUR real book (each symbol simulated from its tracking/account.json
go-live with the live fit_full signal — the same trades you'd actually take),
and trips when EITHER:
  - book drawdown from its high-water mark <= -MAX_DD (8%), or
  - any symbol shows CONSEC_LOSS (4) consecutive losing round-trips.

Tripping is STICKY: once halted it stays halted until a human re-enables it
(set "halted": false in tracking/circuit_breaker.json) — losses pause trading;
a person decides whether to resume. Until then live_signal suppresses BUY/ADD.

CLI:
    python -m futures_swing.circuit_breaker            # show status
    python -m futures_swing.circuit_breaker --reset    # clear a halt (re-enable)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import data_loader, strategy

STATE_FILE = data_loader.REPO_ROOT / "tracking" / "circuit_breaker.json"
ACCOUNT_FILE = data_loader.REPO_ROOT / "tracking" / "account.json"

MAX_DD = 0.08        # halt if book draws down >8% from its high-water mark
CONSEC_LOSS = 4      # halt if any symbol strings together 4 losing round-trips

_FRESH = {"halted": False, "tripped_on": None, "reason": None}


def _load_state() -> dict:
    try:
        return {**_FRESH, **json.loads(STATE_FILE.read_text())}
    except FileNotFoundError:
        return dict(_FRESH)


def _save_state(st: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=2) + "\n")


def _account() -> dict:
    try:
        return json.loads(ACCOUNT_FILE.read_text())
    except FileNotFoundError:
        return {}


def book_metrics(symbols=("ES", "GC")) -> dict:
    """Combined-book drawdown + per-symbol consecutive losses, on the real
    (since-go-live, live-signal) book."""
    acct = _account()
    init = strategy.INIT_EQUITY
    pnl_sum = None
    per = {}
    for s in symbols:
        cfg = strategy.DESIGN[s]
        go = (acct.get(s) or {}).get("go_live")
        df, equity, trades = strategy.simulate(
            s, cfg, sharpe_override=strategy._live_sharpe(s), trade_start=go)
        cl = 0
        for t in reversed(trades):           # trailing consecutive losers
            if t["pnl"] < 0:
                cl += 1
            else:
                break
        per[s] = dict(consec_loss=cl, n_trades=len(trades),
                      pnl=round(float(equity.iloc[-1] - init), 2))
        e = equity - init
        pnl_sum = e if pnl_sum is None else pnl_sum.add(e, fill_value=0)
    book_eq = init + pnl_sum
    hwm = book_eq.cummax()
    dd = float(book_eq.iloc[-1] / hwm.iloc[-1] - 1.0)
    max_cl = max((per[s]["consec_loss"] for s in symbols), default=0)
    return dict(dd=dd, hwm=float(hwm.iloc[-1]), equity=float(book_eq.iloc[-1]),
                max_consec_loss=max_cl, per=per)


def evaluate(symbols=("ES", "GC"), *, persist=False, as_of=None) -> dict:
    """Check the breaker. ``persist`` writes a newly-tripped halt to the state
    file (so CI commits the sticky halt). Returns status + metrics."""
    m = book_metrics(symbols)
    st = _load_state()
    breaches = []
    if m["dd"] <= -MAX_DD:
        breaches.append(f"book drawdown {m['dd']*100:.1f}% (<= -{MAX_DD*100:.0f}%)")
    if m["max_consec_loss"] >= CONSEC_LOSS:
        sym = max(m["per"], key=lambda s: m["per"][s]["consec_loss"])
        breaches.append(f"{sym} {m['max_consec_loss']} consecutive losing trades (>= {CONSEC_LOSS})")
    newly = bool(breaches) and not st["halted"]
    if newly:
        st = {"halted": True,
              "tripped_on": str(as_of or pd.Timestamp.today().date()),
              "reason": "; ".join(breaches)}
        if persist:
            _save_state(st)
    return dict(halted=st["halted"], newly_tripped=newly, breaches=breaches,
                reason=st.get("reason"), tripped_on=st.get("tripped_on"), **m)


def reset() -> None:
    _save_state(dict(_FRESH))
    print("circuit breaker reset — entries re-enabled.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-2 risk circuit breaker")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"])
    ap.add_argument("--reset", action="store_true", help="clear a halt and re-enable entries")
    args = ap.parse_args()
    if args.reset:
        reset()
        return
    r = evaluate(tuple(args.symbols))
    flag = "🛑 HALTED (only-close)" if r["halted"] else "✅ OK"
    print(f"circuit breaker: {flag}")
    print(f"  book equity {r['equity']:.0f} | HWM {r['hwm']:.0f} | "
          f"drawdown {r['dd']*100:+.1f}% / -{MAX_DD*100:.0f}%")
    print(f"  max consecutive losses {r['max_consec_loss']} / {CONSEC_LOSS}")
    for s, p in r["per"].items():
        print(f"    {s}: trades {p['n_trades']}, consec_loss {p['consec_loss']}, pnl {p['pnl']:+.0f}")
    if r["reason"]:
        print(f"  reason: {r['reason']} (tripped {r['tripped_on']})")
    if r["halted"]:
        print("  -> entries suppressed. Re-enable with: python -m futures_swing.circuit_breaker --reset")


if __name__ == "__main__":
    main()
