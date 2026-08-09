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

from . import circuit_breaker, strategy
from .forward_validation import REG_DIR, evaluate

REPO = Path(__file__).resolve().parents[2]
# Tracked path (not data/signals, which is gitignored) so the daily GitHub Actions
# run can commit the forward record back to the repo.
LIVE_LOG = REPO / "tracking" / "live_log.csv"
# Your real book: per-symbol go-live date (you start flat there and take only
# fresh signals — see strategy.live_signal). Missing -> model-continuous view.
ACCOUNT_FILE = REPO / "tracking" / "account.json"
# Pre-registered candidate alphas (shadow-tracked, NOT live) + their forward log.
SHADOW_DIR = REPO / "configs" / "registrations" / "candidates"
SHADOW_LOG = REPO / "tracking" / "shadow_log.csv"


def load_account() -> dict:
    try:
        return json.loads(ACCOUNT_FILE.read_text())
    except FileNotFoundError:
        return {}


def append_live_log(rows: list[dict]) -> None:
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if LIVE_LOG.exists():
        df = pd.concat([pd.read_csv(LIVE_LOG), df], ignore_index=True).drop_duplicates(
            ["asof", "symbol"], keep="last")
    df.to_csv(LIVE_LOG, index=False)


def shadow_signals() -> list[dict]:
    """Forward-track pre-registered CANDIDATE alphas next to the live baseline.

    For each configs/registrations/candidates/*.json spec, fit BOTH the frozen
    candidate and the live baseline on all realized data and record today's
    forecast + vol-scaled sharpe. Forecasts are logged before their outcomes
    realize, so the later promotion decision (the JSON's promotion_rule,
    scored by ``forward_validation --shadow``) rests on true forward evidence,
    immune to later code or data drift."""
    if not SHADOW_DIR.exists():
        return []
    from . import INSTRUMENTS, data_loader, features, model
    from .signal import horizon_forecast_vol

    rows: list[dict] = []
    for path in sorted(SHADOW_DIR.glob("*.json")):
        cand = json.loads(path.read_text())
        sym = cand["symbol"]
        # horizon and BOTH alpha specs come from the frozen JSON — never from the
        # live INSTRUMENTS, which may drift during the multi-month shadow window.
        horizon = int(cand.get("horizon", INSTRUMENTS[sym]["horizon"]))
        X = features.build_feature_matrix(sym, dropna=True)
        close = data_loader.load_ohlc_model(sym)["close"]
        fc = float(horizon_forecast_vol(close, horizon).iloc[-1])
        asof = str(X.index[-1].date())
        for name, spec in (("baseline", cand["baseline_alpha"]), (cand["name"], cand["alpha"])):
            mdl, cols, _ = model.fit_full(sym, alpha_spec=spec)
            want = spec.get("features")
            if isinstance(want, list) and set(cols) != set(want):
                # _select_features silently intersects with available columns; a
                # partial fit (e.g. basis features missing on a cold data cache)
                # is a DIFFERENT model — refuse to log it under the frozen name.
                # The caller's try/except turns this into a skipped day, which is
                # recoverable; a contaminated forward row is not.
                raise RuntimeError(f"{path.name}/{name}: fitted {sorted(cols)} != "
                                   f"frozen features {sorted(want)}")
            pred = float(mdl.predict(X[cols].iloc[[-1]])[0])
            sharpe = pred / fc if fc > 0 else float("nan")
            rows.append(dict(asof=asof, symbol=sym, model=name,
                             pred=round(pred, 6), sharpe=round(sharpe, 4)))
    return rows


def append_shadow_log(rows: list[dict]) -> None:
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if SHADOW_LOG.exists():
        df = pd.concat([pd.read_csv(SHADOW_LOG), df], ignore_index=True).drop_duplicates(
            ["asof", "symbol", "model"], keep="last")
    df.to_csv(SHADOW_LOG, index=False)


def render_issue(sigs: list[dict], cb: dict | None = None) -> tuple[str, str] | None:
    """Render the GitHub issue (title, body) when there's something to push: a
    tradeable action for YOUR book, OR a freshly-tripped circuit breaker. Quiet
    (None) otherwise. A halted breaker prepends a red banner."""
    actionable = [s for s in sigs if strategy.is_actionable(s)]
    halted = bool(cb and cb.get("halted"))
    newly = bool(cb and cb.get("newly_tripped"))
    if not actionable and not newly:
        return None
    asof = max(s["asof"] for s in sigs)
    head = []
    if newly:
        head.append("🛑 CIRCUIT BREAKER TRIPPED")
    if actionable:
        head.append(", ".join(f"{s['symbol']} {s['your_action']}" for s in actionable))
    title = f"Trade signal {asof}: " + " · ".join(head)
    banner = ""
    if halted:
        banner = (f"## 🛑 CIRCUIT BREAKER {'TRIPPED' if newly else 'ACTIVE'} — entries HALTED (only-close)\n"
                  f"**Reason:** {cb.get('reason')}\n"
                  f"book drawdown {cb['dd']*100:+.1f}% (HWM {cb['hwm']:.0f}) · "
                  f"max consecutive losses {cb['max_consec_loss']}\n\n"
                  f"Close-only until you investigate and re-enable: "
                  f"`python -m futures_swing.circuit_breaker --reset`\n\n")
    rows = ["| symbol | YOUR action | your pos | model pos | sharpe | exit |",
            "|---|---|---|---|---|---|"]
    for s in sigs:
        b = "**" if strategy.is_actionable(s) else ""
        rows.append(f"| {b}{s['symbol']}{b} | {b}{s['your_action']}{b} | {s['your_position']} | "
                    f"{s['model_position']} | {s['sharpe']:+.3f} | {s['exit_rule']} |")
    body = (banner + f"**As of {asof}** — pre-registered V1.6 strategy. Trade the **YOUR action** column "
            f"at the **next session open** (1 micro lot per buy, max 2). _model pos_ is the model's "
            f"continuous position, shown for context only — you do **not** chase a position the model "
            f"opened before you went live.\n\n" + "\n".join(rows) +
            "\n\n_Mechanical signal — verify before placing. Sized for ~$220k equity; forward-tracking "
            "is still in its early window, so treat as a heads-up, not a mandate._")
    return title, body


def status(symbols, as_of, do_log):
    acct = load_account()
    sigs = [strategy.live_signal(s, since=(acct.get(s) or {}).get("go_live")) for s in symbols]
    cb = circuit_breaker.evaluate(tuple(symbols), persist=do_log, as_of=as_of)
    for s in sigs:      # preserve the pre-suppression signal in the forward record
        s["raw_action"] = s["your_action"]
    if cb["halted"]:                       # halted => suppress new entries (only-close)
        for s in sigs:
            if s["your_action"].startswith(("BUY", "ADD")):
                s["your_action"] = "HALT-no entry"
    if do_log:
        append_live_log(sigs)
        try:                       # shadow tracking must never break live ops
            srows = shadow_signals()
            if srows:
                append_shadow_log(srows)
                print(f"shadow: logged {len(srows)} candidate/baseline forecasts")
        except Exception as e:  # noqa: BLE001 - isolate candidate failures
            print(f"shadow tracking skipped: {type(e).__name__}: {e}")

    if cb["halted"]:
        print(f"\n🛑 CIRCUIT BREAKER: HALTED (only-close) — {cb['reason']}")
        print(f"   book DD {cb['dd']*100:+.1f}% (HWM {cb['hwm']:.0f}) · max consec losses {cb['max_consec_loss']}"
              f"   ·  re-enable: python -m futures_swing.circuit_breaker --reset")
    else:
        print(f"\n✅ circuit breaker OK — book DD {cb['dd']*100:+.1f}% / -{int(circuit_breaker.MAX_DD*100)}% · "
              f"max consec losses {cb['max_consec_loss']}/{circuit_breaker.CONSEC_LOSS}")

    print("\n================ YOUR ACTION (pre-registered V1.6 strategy) ================")
    print(f"{'symbol':6s} {'asof':12s} {'your_pos':>8s} {'your_action':>16s} {'model_pos':>9s} {'sharpe':>7s} {'exit':>7s}")
    for s in sigs:
        print(f"{s['symbol']:6s} {s['asof']:12s} {s['your_position']:>8d} {s['your_action']:>16s} "
              f"{s['model_position']:>9d} {s['sharpe']:>+7.3f} {s['exit_rule']:>7s}")

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
    return sigs, cb


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
    sigs, cb = status(args.symbols, as_of, args.log)

    if args.json:
        p = Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sigs, indent=2))
    if args.issue_dir:
        d = Path(args.issue_dir); d.mkdir(parents=True, exist_ok=True)
        rendered = render_issue(sigs, cb)
        if rendered:
            (d / "issue_title.txt").write_text(rendered[0], encoding="utf-8")
            (d / "issue_body.md").write_text(rendered[1], encoding="utf-8")
            print(f"\nALERT — tradeable action; issue files written to {d}")
        else:
            print("\nquiet — all HOLD, no issue rendered")
    if args.log:
        print(f"\nlogged to {LIVE_LOG}")


if __name__ == "__main__":
    main()
