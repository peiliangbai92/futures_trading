"""Daily ES/GC briefing — a futures-open heads-up with signal + price levels.

Unlike the event-driven alert (monitor.render_issue, which fires ONLY on a
tradeable BUY/EXIT), this ALWAYS produces a short daily digest: latest price,
1d/5d move, 20d range, ATR(14), today's signal (your book + the model), and an
ATR-based if-you-bought stop/target. The daily-briefing workflow posts it at
~3pm PT (futures open).

CLI:
    python -m futures_swing.briefing --out out/briefing.md
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import INSTRUMENTS, data_loader, fair_value, strategy
from . import vol as volmod
from .execution import levels

ACCOUNT_FILE = data_loader.REPO_ROOT / "tracking" / "account.json"

# Two daily reports, same content, different open: the US cash open and the
# futures reopen. The label is cosmetic; the morning read is on the in-progress
# bar (signal not final until the 5pm ET settle), so it's flagged as preliminary.
SESSIONS = {"morning": "US cash open · 9:30am ET", "evening": "futures open · 6pm ET"}


def _account() -> dict:
    try:
        return json.loads(ACCOUNT_FILE.read_text())
    except FileNotFoundError:
        return {}


def _stats(symbol: str, sig: dict) -> dict:
    spec = INSTRUMENTS[symbol]
    ohlc = data_loader.load_ohlc(symbol)
    close = ohlc["close"]
    last = float(close.iloc[-1])
    chg1 = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else float("nan")
    chg5 = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else float("nan")
    atr = float(volmod.atr(ohlc, window=14).iloc[-1])
    hi20, lo20 = float(close.iloc[-20:].max()), float(close.iloc[-20:].min())
    lv = levels(1, last, atr)   # reference stop/target if you entered long today
    return dict(symbol=symbol, micro=spec["micro_symbol"], pv=spec["point_value"],
                last=last, chg1=chg1, chg5=chg5, atr=atr, hi20=hi20, lo20=lo20,
                stop=lv["stop"], target=lv["target"], sig=sig)


def render(symbols: list[str], brief_date: str, session: str = "evening") -> str:
    acct = _account()
    rows = []
    for s in symbols:
        sig = strategy.live_signal(s, since=(acct.get(s) or {}).get("go_live"))
        rows.append(_stats(s, sig))

    label = SESSIONS.get(session, SESSIONS["evening"])
    lines = [f"## 📋 ES / GC daily briefing — {brief_date} ({label})", ""]
    for r in rows:
        sig, sym = r["sig"], r["symbol"]
        th = strategy.DESIGN[sym]["buy_th"]
        you = "flat" if sig["your_position"] == 0 else f"{sig['your_position']} lot"
        lines += [
            f"**{sym}** ({r['micro']}, ${r['pv']:.0f}/pt) — last **{r['last']:.2f}**  "
            f"(1d {r['chg1'] * 100:+.1f}%, 5d {r['chg5'] * 100:+.1f}%)",
            f"- signal: **{sig['your_action']}** · sharpe {sig['sharpe']:+.3f} (BUY at ≥ +{th:.2f})",
            f"- levels: 20d range {r['lo20']:.0f}–{r['hi20']:.0f} · ATR(14) ~{r['atr']:.0f} pts · "
            f"if-long stop {r['stop']:.0f} / target {r['target']:.0f}",
        ]
        try:                       # cost-of-carry fair value — best-effort, never block the post
            fv_ln = fair_value.line(sym, asof=brief_date, future_price=r["last"])
            if fv_ln:
                lines.append(fv_ln)
        except Exception as e:
            lines.append(f"_(fair value skipped: {type(e).__name__})_")
        lines += [
            f"- you: {you} · model: {sig['model_action']}",
            "",
        ]
    actionable = [r["sig"] for r in rows if strategy.is_actionable(r["sig"])]
    if actionable:
        lines.append("→ **Action at this open:** "
                     + ", ".join(f"{s['symbol']} {s['your_action']}" for s in actionable))
    else:
        lines.append("→ No action at this open — stay flat.")
    lines.append("_Morning read on the in-progress bar — the daily signal isn't final until the "
                 "5pm ET settle; the 6pm ET briefing is authoritative._" if session == "morning"
                 else "_A separate alert fires only on a BUY/EXIT._")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily ES/GC briefing")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"])
    ap.add_argument("--out", default="out/briefing.md")
    ap.add_argument("--date", default=os.environ.get("BRIEF_DATE", ""))
    ap.add_argument("--session", default=os.environ.get("BRIEF_SESSION", "evening"),
                    choices=["morning", "evening"])
    args = ap.parse_args()
    brief_date = args.date or str(data_loader.load_ohlc(args.symbols[0]).index[-1].date())
    md = render(args.symbols, brief_date, session=args.session)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")   # emoji-safe on Windows (locale default is not UTF-8)
    try:
        print(md)
    except UnicodeEncodeError:           # non-UTF-8 local console; the file is authoritative
        print(md.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
