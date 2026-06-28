"""Futures fair value — cost-of-carry pre-open context for ES / GC.

The no-arbitrage fair value of a future is its spot/cash carried to expiry (the
standard cost-of-carry model; Alma, *Futures Trading Pt 1*).

ES (equity index) — we have an independent cash level (^GSPC), so we price the
future directly:
    FV = S_cash * exp((r - q) * T),     basis = future - FV
A future ABOVE fair value is rich (premium), BELOW is cheap (discount); basis
-> 0 as the contract converges to cash at expiry.

GC (gold) — there is no clean free daily *spot* feed (the LBMA FRED series was
discontinued; XAUUSD is delisted on Yahoo). So we read carry straight off the
futures TERM STRUCTURE: the slope between two listed contracts is the market's
implied carry, and the fair-value test is whether it equals cost-of-carry
(r + storage - lease). Implied carry < r, or an inverted (backwardated) curve,
flags physical tightness — the bullish gold signal in docs/gold_data_sources.md.

Inputs are pulled/cached SEPARATELY from the model feeds so the alpha pipeline is
untouched, and every read degrades to ``None`` when an input is missing — the
briefing shows "unavailable" rather than breaking.

CLI:
    python -m futures_swing.fair_value --refresh   # pull the input feeds
    python -m futures_swing.fair_value             # print ES/GC fair value
"""
from __future__ import annotations

import argparse
import calendar
import math
from datetime import date

import pandas as pd

from . import INSTRUMENTS, data_loader

# Index fair-value feeds, kept OUT of the model's symbol dicts. key -> source id.
FV_YF = {"SPX_CASH": "^GSPC"}            # S&P 500 cash index (yfinance)
FV_FRED = {"TBILL_3M": "DGS3MO"}         # 3M T-bill, secondary market (%) — the carry rate
# GC term-structure contracts are pulled dynamically (see _gc_curve_contracts).
GC_MONTH_CODE = {2: "G", 4: "J", 6: "M", 8: "Q", 10: "V", 12: "Z"}   # active even months
STALE_DAYS = 6   # inputs older than this (vs as-of) are flagged in the output


# ----------------------------------------------------------------- expiry calendar


def _third_friday(y: int, m: int) -> date:
    """Quarterly equity-index expiry: the 3rd Friday of the month."""
    first = date(y, m, 1)
    first_friday = 1 + (4 - first.weekday()) % 7   # weekday(): Mon=0 .. Fri=4
    return date(y, m, first_friday + 14)


def _nth_last_business_day(y: int, m: int, n: int) -> date:
    """The n-th-from-last weekday of a month (GC last-trade ~ 3rd-to-last bd)."""
    days = [date(y, m, d) for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    bdays = [d for d in days if d.weekday() < 5]
    return bdays[-n]


def next_quarterly_expiry(asof: date) -> date:
    """Front equity-index expiry strictly after ``asof`` (3rd Fri Mar/Jun/Sep/Dec)."""
    cands = [_third_friday(y, m) for y in (asof.year, asof.year + 1)
             for m in (3, 6, 9, 12) if _third_friday(y, m) > asof]
    return min(cands)


def _gc_curve_contracts(asof: date, n: int = 3) -> list[tuple[date, str, str]]:
    """Next ``n`` active even-month GC contracts after ``asof``.

    Returns (expiry, yahoo_symbol, cache_key); expiry ~ 3rd-to-last business day of
    the contract month. The few-days approximation is immaterial to the implied
    carry over months.
    """
    out: list[tuple[date, str, str]] = []
    for y in (asof.year, asof.year + 1, asof.year + 2):
        for m in (2, 4, 6, 8, 10, 12):
            exp = _nth_last_business_day(y, m, 3)
            if exp > asof:
                sym = f"GC{GC_MONTH_CODE[m]}{y % 100:02d}.CMX"
                out.append((exp, sym, f"GCCURVE_{y}{m:02d}"))
            if len(out) >= n:
                return out
    return out


# --------------------------------------------------------------------- inputs


def refresh_inputs(asof: str | None = None) -> dict[str, int]:
    """Pull the fair-value feeds into ``data/raw/<KEY>.parquet`` (reuses the
    model's loaders + cache dir). Returns {key: n_rows} for what succeeded."""
    data_loader.RAW_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.Timestamp(asof).date() if asof else date.today()
    out: dict[str, int] = {}

    def _save(key: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(data_loader.RAW_DIR / f"{key}.parquet")
        out[key] = len(frame)
        print(f"  {key:14s} {len(frame)} rows .. {frame.index[-1].date()}")

    for key, sym in FV_YF.items():
        try:
            _save(key, data_loader.download(sym))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {key:14s} ({sym}) FAILED: {exc}")
    for key, sid in FV_FRED.items():
        try:
            _save(key, data_loader.download_fred(sid))
        except Exception as exc:  # noqa: BLE001
            print(f"  {key:14s} (FRED:{sid}) FAILED: {exc}")
    for _exp, sym, key in _gc_curve_contracts(ref):
        try:
            _save(key, data_loader.download(sym))
        except Exception as exc:  # noqa: BLE001 - a not-yet-listed back month may 404
            print(f"  {key:14s} ({sym}) FAILED: {exc}")
    return out


def _asof_value(key: str, asof: date | None) -> tuple[float | None, date | None]:
    """Last cached value (and its date) of a single-column feed at/<= asof."""
    try:
        s = data_loader.load(key)["close"].dropna()
    except FileNotFoundError:
        return None, None
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    if s.empty:
        return None, None
    return float(s.iloc[-1]), s.index[-1].date()


# --------------------------------------------------------------------- compute


def _ref_date(symbol: str, asof: str | None, future_price: float | None) -> tuple[float | None, date | None]:
    """Resolve the headline future price + reference date for ``symbol``."""
    asof_ts = pd.Timestamp(asof) if asof else None
    fut = data_loader.load_ohlc(symbol)["close"].dropna()
    if asof_ts is not None:
        fut = fut[fut.index <= asof_ts]
    if fut.empty:
        return future_price, (asof_ts.date() if asof_ts is not None else None)
    return (future_price if future_price is not None else float(fut.iloc[-1])), fut.index[-1].date()


def summary(symbol: str, asof: str | None = None, future_price: float | None = None) -> dict | None:
    """Cost-of-carry fair value for ``symbol`` as of ``asof`` (date str, or latest),
    or ``None`` if the future or a required carry input is absent."""
    spec = INSTRUMENTS[symbol].get("fair_value")
    if spec is None:
        return None
    future, ref = _ref_date(symbol, asof, future_price)
    if future is None or ref is None:
        return None
    r, r_dt = _asof_value(spec["rate_key"], ref)
    if r is None:
        return None
    r = r / 100.0                                  # FRED rates are in percent
    stale_src = [r_dt]

    if spec["kind"] == "index":
        cash, s_dt = _asof_value(spec["cash_key"], ref)
        if cash is None:
            return None
        stale_src.append(s_dt)
        q = float(spec.get("div_yield", 0.0))
        expiry = next_quarterly_expiry(ref)
        T = (expiry - ref).days / 365.0
        if T <= 0:
            return None
        fv = cash * math.exp((r - q) * T)
        out = dict(kind="index", cash=cash, q=q, expiry=str(expiry), dte=(expiry - ref).days,
                   T=T, fair_value=fv, basis=future - fv)

    elif spec["kind"] == "commodity":
        pts = []
        for exp, _sym, key in _gc_curve_contracts(ref, spec.get("curve_points", 3)):
            px, _dt = _asof_value(key, ref)
            if px is not None:
                pts.append((exp, px))
        if len(pts) < 2:
            return None
        (e1, p1), (e2, p2) = pts[0], pts[-1]        # nearest vs farthest available
        T1, T2 = (e1 - ref).days / 365.0, (e2 - ref).days / 365.0
        if T2 <= T1:
            return None
        impl_carry = math.log(p2 / p1) / (T2 - T1)  # market-implied annualized carry
        theo_carry = r + float(spec.get("storage", 0.0)) - float(spec.get("lease", 0.0))
        out = dict(kind="commodity", near_exp=str(e1), far_exp=str(e2), near_px=p1, far_px=p2,
                   near_label=f"{calendar.month_abbr[e1.month]}{e1.year % 100:02d}",
                   far_label=f"{calendar.month_abbr[e2.month]}{e2.year % 100:02d}",
                   impl_carry=impl_carry, theo_carry=theo_carry,
                   basis=(impl_carry - theo_carry))   # +ve = steeper than cost-of-carry
    else:
        return None

    stale = max((ref - d).days for d in stale_src if d is not None) if any(stale_src) else None
    out.update(symbol=symbol, asof=str(ref), r=r, future=float(future), stale_days=stale)
    return out


# --------------------------------------------------------------------- render


def line(symbol: str, asof: str | None = None, future_price: float | None = None) -> str | None:
    """One-line markdown fair-value read for the briefing, or ``None`` if absent."""
    s = summary(symbol, asof=asof, future_price=future_price)
    if s is None:
        return None
    flag = f" ⚠ inputs {s['stale_days']}d stale" if s["stale_days"] and s["stale_days"] > STALE_DAYS else ""
    if s["kind"] == "index":
        basis = s["basis"]
        word = "rich (premium to fair)" if basis > 0 else "cheap (discount to fair)" if basis < 0 else "at fair"
        return (f"- fair value **{s['fair_value']:.1f}** "
                f"(cash {s['cash']:.1f} · r {s['r']*100:.2f}% − q {s['q']*100:.1f}% · {s['dte']}d) — "
                f"future {s['future']:.1f} → **{basis:+.1f}** {word}{flag}")
    # commodity: implied carry from the curve vs cost-of-carry
    impl, theo = s["impl_carry"] * 100, s["theo_carry"] * 100
    if impl < 0:
        read = "**backwardation** — physical tightness (bullish)"
    elif impl < theo - 1.0:
        read = "soft contango — tightening vs carry"
    elif impl > theo + 1.0:
        read = "steep contango — above cost-of-carry"
    else:
        read = "contango ≈ cost-of-carry (fairly priced)"
    return (f"- fair value: front **{s['future']:.1f}** · curve {s['near_label']} {s['near_px']:.1f} → "
            f"{s['far_label']} {s['far_px']:.1f} · implied carry {impl:+.1f}%/yr vs cost-of-carry "
            f"{theo:.1f}%/yr → {read}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Futures cost-of-carry fair value")
    ap.add_argument("--refresh", action="store_true", help="pull the input feeds first")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"])
    ap.add_argument("--date", default=None, help="as-of date (YYYY-MM-DD); default latest")
    args = ap.parse_args()
    if args.refresh:
        print("Refreshing fair-value inputs ->", data_loader.RAW_DIR)
        refresh_inputs(asof=args.date)
    for sym in args.symbols:
        ln = line(sym, asof=args.date)
        print(f"{sym}: {ln if ln else 'unavailable (missing future or carry input)'}")


if __name__ == "__main__":
    main()
