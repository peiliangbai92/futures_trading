"""Data layer — yfinance daily pull + parquet cache (V1).

Pulls the V1 universe (ES, GC + cross-asset macro) from Yahoo Finance and
caches one tidy parquet per key under ``data/raw/<KEY>.parquet``. Keys:

    ES, GC                      OHLCV futures (continuous, Yahoo-rolled)
    VIX, VVIX, DXY, UST10Y,     macro / cross-asset (close used as level)
    TIP, OIL

Caveat (mirrors QuantitativeResearch/scripts/futures_mr.py): Yahoo continuous
futures carry quarterly roll jumps with no back-adjustment — a small artifact
in multi-day returns at roll dates. V2 replaces this with Databento continuous
contracts.

CLI:
    python -m futures_swing.data_loader            # refresh all (period=max)
    python -m futures_swing.data_loader --period 5y
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import FRED_SERIES, INSTRUMENTS, MACRO_SYMBOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# key -> yahoo symbol for everything we pull
YF_SYMBOLS: dict[str, str] = {k: v["yf_symbol"] for k, v in INSTRUMENTS.items()}
YF_SYMBOLS.update(MACRO_SYMBOLS)

_OHLC_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance frame to a tidy OHLCV frame, DatetimeIndex 'date'."""
    if isinstance(df.columns, pd.MultiIndex):
        # single-ticker download still returns (field, ticker) columns
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    for col in _OHLC_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[_OHLC_COLS].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in _OHLC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Drop rows with missing or non-positive prices. This also removes the
    # 2020-04-20 negative WTI settle (CL=F), which would break log returns.
    price_cols = ["open", "high", "low", "close"]
    out = out[(out[price_cols] > 0).all(axis=1)]
    # Repair range glitches so low <= {O,H,L,C} <= high (Yahoo occasionally
    # ships GC bars where high < low); keeps the bar instead of dropping it.
    out["high"] = out[price_cols].max(axis=1)
    out["low"] = out[price_cols].min(axis=1)
    return out


def download(yf_symbol: str, *, period: str = "max", start: str | None = None) -> pd.DataFrame:
    """Download one symbol's daily bars from Yahoo. Returns a tidy OHLCV frame."""
    import yfinance as yf

    kwargs = dict(interval="1d", progress=False, auto_adjust=False)
    if start:
        raw = yf.download(yf_symbol, start=start, **kwargs)
    else:
        raw = yf.download(yf_symbol, period=period, **kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_symbol}")
    return _flatten(raw)


def download_fred(series_id: str, *, start: str = "1990-01-01") -> pd.DataFrame:
    """Download a FRED series via the keyless fredgraph CSV. Returns a frame with
    a DatetimeIndex 'date' and a single 'close' column (so ``load_close`` works)."""
    import io

    import requests

    resp = requests.get(FRED_CSV_URL, params={"id": series_id, "cosd": start}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "close"]  # observation_date, <SERIES_ID>
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")  # FRED marks gaps "."
    return df.dropna(subset=["close"]).set_index("date").sort_index()


def refresh(keys: list[str] | None = None, *, period: str = "max") -> dict[str, pd.DataFrame]:
    """Pull each key (yfinance + FRED) and write ``data/raw/<KEY>.parquet``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    yf_keys = list(YF_SYMBOLS) if keys is None else [k for k in keys if k in YF_SYMBOLS]
    fred_keys = list(FRED_SERIES) if keys is None else [k for k in keys if k in FRED_SERIES]
    out: dict[str, pd.DataFrame] = {}

    for key in yf_keys:
        sym = YF_SYMBOLS[key]
        try:
            frame = download(sym, period=period)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {key:12s} ({sym}) FAILED: {exc}")
            continue
        frame.to_parquet(RAW_DIR / f"{key}.parquet")
        out[key] = frame
        print(f"  {key:12s} ({sym}) {len(frame)} rows {frame.index[0].date()}..{frame.index[-1].date()}")

    for key in fred_keys:
        sid = FRED_SERIES[key]
        try:
            frame = download_fred(sid)
        except Exception as exc:  # noqa: BLE001
            print(f"  {key:12s} (FRED:{sid}) FAILED: {exc}")
            continue
        frame.to_parquet(RAW_DIR / f"{key}.parquet")
        out[key] = frame
        print(f"  {key:12s} (FRED:{sid}) {len(frame)} rows {frame.index[0].date()}..{frame.index[-1].date()}")
    return out


def load(key: str) -> pd.DataFrame:
    """Load a cached raw frame. Raises if not yet pulled."""
    path = RAW_DIR / f"{key}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m futures_swing.data_loader` first")
    return pd.read_parquet(path)


def load_ohlc(symbol: str) -> pd.DataFrame:
    """OHLC(V) frame for an instrument key (ES / GC)."""
    if symbol not in INSTRUMENTS:
        raise KeyError(f"{symbol} is not an instrument; use load() for macro keys")
    return load(symbol)


def load_close(key: str) -> pd.Series:
    """Close series for any key (used as the level for macro feeds)."""
    return load(key)["close"].rename(key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the V1 yfinance data cache")
    ap.add_argument("--period", default="max", help="yfinance period (default: max)")
    ap.add_argument("--keys", nargs="*", default=None, help="subset of keys to refresh")
    args = ap.parse_args()
    print(f"Refreshing raw cache -> {RAW_DIR}")
    got = refresh(args.keys, period=args.period)
    total = len(args.keys) if args.keys else len(YF_SYMBOLS) + len(FRED_SERIES)
    print(f"Done: {len(got)}/{total} keys cached")


if __name__ == "__main__":
    main()
