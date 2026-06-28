"""Index-futures fair value — cost-of-carry pre-open context for ES.

The no-arbitrage fair value of an index future is its cash level carried to
expiry (Alma, *Futures Trading Pt 1*; continuous form, r = SOFR financing):

    FV = S_cash * exp((r - q) * T)            [T = days / 360, ACT/360 money-market]
    fair basis = FV - cash                     (carry - dividend; rigid future-cash gap)
    implied cash open = future - fair basis    (Alma's pre-open read)

A future ABOVE fair value is rich (premium), BELOW is cheap (discount), but the
premium alone is noise — directional only if persistent and risk-reversal-confirmed.
(The institutional ACT/360 form sums a discrete dated dividend stream; here the
continuous yield q is that stream's simplification.)

Inputs (^GSPC cash, SOFR) are pulled/cached SEPARATELY from the model feeds so
the alpha pipeline is untouched, and every read degrades to ``None`` when an
input is missing — the briefing shows nothing rather than breaking.

CLI:
    python -m futures_swing.fair_value --refresh   # pull the input feeds
    python -m futures_swing.fair_value             # print the fair value
"""
from __future__ import annotations

import argparse
import math
from datetime import date

import pandas as pd

from . import INSTRUMENTS, data_loader

# Fair-value input feeds, kept OUT of the model's symbol dicts. key -> source id.
FV_YF = {"SPX_CASH": "^GSPC"}            # S&P 500 cash index (yfinance)
FV_FRED = {"SOFR": "SOFR"}               # SOFR — the financing rate in cost-of-carry (Alma)
STALE_DAYS = 6     # inputs older than this (vs as-of) are flagged in the output
DAY_COUNT = 360    # ACT/360 money-market day-count (Alma: institutional carry convention)


# ----------------------------------------------------------------- expiry calendar


def _third_friday(y: int, m: int) -> date:
    """Quarterly equity-index expiry: the 3rd Friday of the month."""
    first = date(y, m, 1)
    first_friday = 1 + (4 - first.weekday()) % 7   # weekday(): Mon=0 .. Fri=4
    return date(y, m, first_friday + 14)


def next_quarterly_expiry(asof: date) -> date:
    """Front equity-index expiry strictly after ``asof`` (3rd Fri Mar/Jun/Sep/Dec)."""
    cands = [_third_friday(y, m) for y in (asof.year, asof.year + 1)
             for m in (3, 6, 9, 12) if _third_friday(y, m) > asof]
    return min(cands)


# --------------------------------------------------------------------- inputs


def refresh_inputs() -> dict[str, int]:
    """Pull the fair-value feeds into ``data/raw/<KEY>.parquet`` (reuses the
    model's loaders + cache dir). Returns {key: n_rows} for what succeeded."""
    data_loader.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, int] = {}

    def _save(key: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(data_loader.RAW_DIR / f"{key}.parquet")
        out[key] = len(frame)
        print(f"  {key:10s} {len(frame)} rows .. {frame.index[-1].date()}")

    for key, sym in FV_YF.items():
        try:
            _save(key, data_loader.download(sym))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {key:10s} ({sym}) FAILED: {exc}")
    for key, sid in FV_FRED.items():
        try:
            _save(key, data_loader.download_fred(sid))
        except Exception as exc:  # noqa: BLE001
            print(f"  {key:10s} (FRED:{sid}) FAILED: {exc}")
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
    """Cost-of-carry fair value for an index ``symbol`` as of ``asof`` (date str,
    or latest), or ``None`` if the future or a carry input is absent."""
    spec = INSTRUMENTS[symbol].get("fair_value")
    if spec is None or spec.get("kind") != "index":
        return None
    future, ref = _ref_date(symbol, asof, future_price)
    if future is None or ref is None:
        return None
    r, r_dt = _asof_value(spec["rate_key"], ref)
    if r is None:
        return None
    r = r / 100.0                                  # FRED rates are in percent
    cash, s_dt = _asof_value(spec["cash_key"], ref)
    if cash is None:
        return None
    q = float(spec.get("div_yield", 0.0))
    expiry = next_quarterly_expiry(ref)
    dte = (expiry - ref).days
    T = dte / DAY_COUNT                            # ACT/360 money-market (Alma)
    if T <= 0:
        return None
    fv = cash * math.exp((r - q) * T)
    fair_basis = fv - cash                         # carry - dividend = rigid future-cash gap
    stale = max((ref - d).days for d in (r_dt, s_dt) if d is not None) if (r_dt or s_dt) else None
    return dict(symbol=symbol, kind="index", asof=str(ref), expiry=str(expiry), dte=dte, T=T,
                r=r, q=q, future=float(future), cash=cash, fair_value=fv, fair_basis=fair_basis,
                basis=future - fv, implied_cash_open=future - fair_basis, stale_days=stale)


# --------------------------------------------------------------------- render


def line(symbol: str, asof: str | None = None, future_price: float | None = None) -> str | None:
    """One-line markdown fair-value read for the briefing, or ``None`` if absent."""
    s = summary(symbol, asof=asof, future_price=future_price)
    if s is None:
        return None
    flag = f" ⚠ inputs {s['stale_days']}d stale" if s["stale_days"] and s["stale_days"] > STALE_DAYS else ""
    basis = s["basis"]
    word = "rich" if basis > 0 else "cheap" if basis < 0 else "at fair"
    return (f"- fair value **{s['fair_value']:.1f}** "
            f"(cash {s['cash']:.1f} · SOFR {s['r']*100:.2f}% − q {s['q']*100:.1f}% · {s['dte']}d ACT/360) · "
            f"fair basis **{s['fair_basis']:+.1f}** — future {s['future']:.1f} → **{basis:+.1f}** {word}, "
            f"implied cash open ~{s['implied_cash_open']:.0f} (premium alone = noise){flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Index futures cost-of-carry fair value")
    ap.add_argument("--refresh", action="store_true", help="pull the input feeds first")
    ap.add_argument("--symbols", nargs="+", default=["ES"])
    ap.add_argument("--date", default=None, help="as-of date (YYYY-MM-DD); default latest")
    args = ap.parse_args()
    if args.refresh:
        print("Refreshing fair-value inputs ->", data_loader.RAW_DIR)
        refresh_inputs()
    for sym in args.symbols:
        ln = line(sym, asof=args.date)
        print(f"{sym}: {ln if ln else 'unavailable (missing future or carry input)'}")


if __name__ == "__main__":
    main()
