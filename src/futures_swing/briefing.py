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

from . import INSTRUMENTS, data_loader, strategy
from . import vol as volmod
from .execution import levels
from .intraday import gold_gamma

ACCOUNT_FILE = data_loader.REPO_ROOT / "tracking" / "account.json"


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


GAMMA_REQUIRED = ("regime", "regime_label", "conv", "asof", "below", "above")
GAMMA_STALE_DAYS = 4   # > a long weekend → OI/levels too old to show as live


def _stale_days(asof: str, brief_date: str) -> int | None:
    from datetime import date
    try:
        return (date.fromisoformat(brief_date[:10]) - date.fromisoformat(asof[:10])).days
    except ValueError:
        return None


def _level(x: dict) -> str:
    """One mapped level with its tags + gamma sign (+g cushion / -g accelerant)."""
    tag = " (line-in-sand)" if x.get("line_in_sand") else (" (call wall)" if x.get("call_wall") else "")
    g = "+g" if x.get("sign", 0) > 0 else "-g"
    return f"{x['gc']:.0f}{tag} {g}"


def _gamma_lines(brief_date: str) -> list[str]:
    """GC options-structure levels from the committed GLD-gamma snapshot. Empty when
    the snapshot is absent/partial (e.g. CI without QR) so the briefing still runs.
    Strikes carry a +g/-g sign: +g = dealer-long (cushion/pin), -g = short (accelerant);
    side names are geometric (below/above spot), NOT a promise of support/resistance."""
    snap = gold_gamma.load_snapshot()
    if not snap or not all(k in snap for k in GAMMA_REQUIRED):
        return []
    days = _stale_days(snap["asof"], brief_date)
    if days is not None and days > GAMMA_STALE_DAYS:   # too old → suppress specific levels
        return [f"**GC gamma map** — unavailable (snapshot {days}d stale, as of {snap['asof']}; "
                f"regenerate via `python -m futures_swing.intraday.gold_gamma`).", ""]
    stale = f" ⚠ as of {snap['asof']}" if days and days > 0 else ""
    icon = {-1: "🔴", 1: "🟢", 0: "⚪"}.get(snap["regime"], "⚪")
    below = " · ".join(_level(x) for x in snap["below"]) or "n/a"
    above = " · ".join(_level(x) for x in snap["above"]) or "n/a"
    centroid = snap.get("centroid_gc"); dpivot = snap.get("downside_pivot_gc")
    lines = [
        f"**GC gamma map** (GLD options → GC ×{snap['conv']:.1f}{stale}) — {icon} {snap['regime_label']}",
        f"- below spot: {below}",
        f"- above spot: {above} · centroid {centroid if centroid is not None else 'n/a'} "
        f"· down-pivot {dpivot if dpivot is not None else 'n/a'}",
    ]
    if snap["regime"] < 0:
        lines.append("- read: short-gamma — **neither** dips nor rallies dealer-cushioned. "
                     "-g levels are accelerants (price gets drawn to them, then runs on a break); "
                     "+g levels can still pin/cap locally.")
    elif snap["regime"] > 0:
        lines.append("- read: long-gamma — dealers dampen moves; these levels tend to hold (mean-revert).")
    lines.append("")
    return lines


def render(symbols: list[str], brief_date: str) -> str:
    acct = _account()
    rows = []
    for s in symbols:
        sig = strategy.live_signal(s, since=(acct.get(s) or {}).get("go_live"))
        rows.append(_stats(s, sig))

    lines = [f"## 📋 ES / GC daily briefing — {brief_date} (futures open, ~3pm PT)", ""]
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
            f"- you: {you} · model: {sig['model_action']}",
            "",
        ]
        if sym == "GC":
            try:                       # the gamma map is a nice-to-have — never let it kill the post
                lines += _gamma_lines(brief_date)
            except Exception as e:
                lines += [f"_(GC gamma map skipped: {type(e).__name__})_", ""]
    actionable = [r["sig"] for r in rows if strategy.is_actionable(r["sig"])]
    if actionable:
        lines.append("→ **Action at this open:** "
                     + ", ".join(f"{s['symbol']} {s['your_action']}" for s in actionable))
    else:
        lines.append("→ No action at this open — stay flat. "
                     "(A separate alert fires only on a BUY/EXIT.)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily ES/GC briefing")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"])
    ap.add_argument("--out", default="out/briefing.md")
    ap.add_argument("--date", default=os.environ.get("BRIEF_DATE", ""))
    args = ap.parse_args()
    brief_date = args.date or str(data_loader.load_ohlc(args.symbols[0]).index[-1].date())
    md = render(args.symbols, brief_date)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
