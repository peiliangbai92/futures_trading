"""Daily pipeline — produce today's signal + entry/stop/target per instrument.

Trains the alpha on all realized history, forecasts the latest date, risk-adjusts
to a Sharpe signal, and emits direction + ATR levels + micro-contract size
(doc Section 18). Writes ``data/signals/latest_signal.csv``.

CLI:
    python -m futures_swing.pipeline               # all instruments, $50k equity
    python -m futures_swing.pipeline --equity 25000 --symbols ES
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import INSTRUMENTS, data_loader, model, regime, signal
from . import vol as volmod
from .execution import levels
from .risk import conviction, load_event_dates, near_event, vol_target_size

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_CSV = REPO_ROOT / "data" / "signals" / "latest_signal.csv"

_SIGNAL_TEXT = {1: "LONG", 0: "FLAT", -1: "SHORT"}


def signal_for(symbol: str, *, equity: float) -> dict:
    spec = INSTRUMENTS[symbol]
    pv, horizon = spec["point_value"], spec["horizon"]

    mdl, cols, _ = model.fit_full(symbol)
    last_date, pred = model.predict_latest(symbol, mdl, cols)

    ohlc = data_loader.load_ohlc(symbol)
    close = ohlc["close"]
    fc_vol = signal.horizon_forecast_vol(close, horizon).get(last_date, np.nan)
    sharpe = pred / fc_vol if fc_vol and np.isfinite(fc_vol) and fc_vol > 0 else np.nan
    side = signal.discretize(sharpe)

    atr_val = float(volmod.atr(ohlc, window=14).get(last_date, np.nan))
    entry = float(close.get(last_date, np.nan))
    reg = regime.classify(data_loader.load_close("ES"), data_loader.load_close("VIX"))
    reg_label = reg.reindex([last_date], method="ffill").iloc[0]

    row = dict(
        date=str(pd.Timestamp(last_date).date()), symbol=symbol, regime=reg_label,
        signal=_SIGNAL_TEXT[side], expected_return=round(float(pred), 5),
        sharpe=round(float(sharpe), 3) if np.isfinite(sharpe) else np.nan,
        entry=np.nan, stop=np.nan, target=np.nan, contracts=0, note="",
    )

    if side == 0 or not (atr_val > 0):
        row["note"] = "flat"
        return row

    ann_vol = float(volmod.close_to_close_volatility(close, window=21).get(last_date, np.nan))
    on_event = near_event(last_date, load_event_dates())
    qty = 0 if on_event else vol_target_size(equity, ann_vol, entry, pv, conviction_mult=conviction(sharpe))
    lv = levels(side, entry, atr_val)
    if on_event:
        note = "event-blocked"
    elif qty == 0:
        note = "sub-min-size"
    else:
        note = ""
    row.update(
        entry=round(entry, 4), stop=round(lv["stop"], 4), target=round(lv["target"], 4),
        contracts=int(qty), note=note,
    )
    return row


def run(symbols: list[str], *, equity: float) -> pd.DataFrame:
    rows = [signal_for(s, equity=equity) for s in symbols]
    out = pd.DataFrame(rows)
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SIGNALS_CSV, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily swing signal generator")
    ap.add_argument("--symbols", nargs="+", default=list(INSTRUMENTS), choices=list(INSTRUMENTS))
    ap.add_argument("--equity", type=float, default=220_000.0)
    args = ap.parse_args()

    out = run(args.symbols, equity=args.equity)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(out.to_string(index=False))
    print(f"\nwrote {SIGNALS_CSV}")


if __name__ == "__main__":
    main()
