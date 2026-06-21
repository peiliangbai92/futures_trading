"""V1.6 full signal-design backtest — the per-symbol strategies from the
signal-map study, with position management (1 lot per buy, max 2) and costs.

ES (mean-reversion):
  BUY  ridge sharpe >= +0.20, de-clustered by a 15d cooldown (keep strongest).
  EXIT (all) when 20d momentum rolls negative (sell the rip).
GC (trend):
  BUY  lgbm sharpe >= +0.30 OR a fresh 40d-high breakout, 20d cooldown.
  EXIT (all) on a trailing stop: price 8% below the peak since first entry.

Forecasts are the purged walk-forward OOS predictions (honest), but the rule
parameters (thresholds, cooldown, momentum/trail/breakout windows) were chosen
IN-SAMPLE across 2005-2026. So this is OOS-forecast / in-sample-rules — treat the
numbers as an upper bound and PRE-REGISTER + forward-test before trusting them.

CLI:
    python -m futures_swing.strategy --symbols ES GC
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import INSTRUMENTS, data_loader, features, model, signal
from . import vol as volmod
from .backtest import COMMISSION_PER_SIDE, SLIPPAGE_TICKS, _ann_stats

INIT_EQUITY = 220_000.0
MAX_LOTS = 2

DESIGN = {
    "ES": dict(buy_th=0.20, cooldown=15, sell="mom20", breakout=0),
    "GC": dict(buy_th=0.30, cooldown=20, sell="trail", trail_drop=0.08, trail_win=60, breakout=40),
}


def _cluster_first(cands, cooldown):
    """Keep the strongest signal per cooldown window (cands: [(i, strength)])."""
    cands = sorted(cands); kept, last, i = [], -10**9, 0
    while i < len(cands):
        j = i
        while j + 1 < len(cands) and cands[j + 1][0] - cands[i][0] < cooldown:
            j += 1
        best = max(cands[i:j + 1], key=lambda t: t[1])
        if best[0] - last >= cooldown:
            kept.append(best[0]); last = best[0]
        i = j + 1
    return set(kept)


def build_signals(symbol: str, cfg: dict, sharpe_override: pd.Series | None = None):
    ohlc = data_loader.load_ohlc(symbol); close = ohlc["close"]
    horizon = INSTRUMENTS[symbol]["horizon"]
    if sharpe_override is not None:        # for the shuffled-forecast null / baselines
        shp = sharpe_override
    else:
        pred = model.walk_forward(symbol).oos_pred
        fc = signal.horizon_forecast_vol(close, horizon).reindex(pred.index)
        shp = (pred / fc).replace([np.inf, -np.inf], np.nan)
    idx = shp.dropna().index

    df = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close"):
        df[c] = ohlc[c].reindex(idx)
    df["sharpe"] = shp.reindex(idx)
    df = df.dropna(subset=["open", "high", "low", "close", "sharpe"])
    idx = df.index

    # buy candidates: forecast + (optional) fresh breakout
    cands = [(k, df["sharpe"].iloc[k]) for k in range(len(df)) if df["sharpe"].iloc[k] >= cfg["buy_th"]]
    if cfg.get("breakout"):
        prior_hi = close.shift(1).rolling(cfg["breakout"]).max().reindex(idx)
        have = {k for k, _ in cands}
        for k in range(len(df)):
            if k not in have and np.isfinite(prior_hi.iloc[k]) and df["close"].iloc[k] > prior_hi.iloc[k]:
                cands.append((k, cfg["buy_th"]))
    buy_days = _cluster_first(cands, cfg["cooldown"])

    # momentum-rollover sell (ES); trailing handled in the sim (GC)
    sell_days: set[int] = set()
    if cfg["sell"] == "mom20":
        ret20 = np.log(df["close"] / df["close"].shift(20))
        sell_days = {k for k in range(len(df)) if ret20.iloc[k] < 0 and ret20.iloc[k - 1] >= 0}
    return df, buy_days, sell_days


def simulate(symbol: str, cfg: dict, *, init_equity=INIT_EQUITY, max_lots=MAX_LOTS,
             sharpe_override=None, trade_start=None):
    """``trade_start`` (a Timestamp): no NEW entries before it — gives a flat start
    for forward-window evaluation; equity stays at ``init_equity`` until then."""
    df, buy_days, sell_days = build_signals(symbol, cfg, sharpe_override=sharpe_override)
    trade_start = pd.Timestamp(trade_start) if trade_start is not None else None
    pv = INSTRUMENTS[symbol]["point_value"]; tick = INSTRUMENTS[symbol]["tick"]
    roundtrip = 2 * COMMISSION_PER_SIDE + 2 * SLIPPAGE_TICKS * tick * pv
    o, h, c = (df[k].to_numpy(float) for k in ("open", "high", "close"))
    n = len(df); idx = df.index
    cash = init_equity; equity = pd.Series(init_equity, index=idx, dtype=float)
    pos = 0; cost_basis = 0.0; peak = 0.0; last_buy = -10**9
    trades = []; entry_first = None
    trail = cfg["sell"] == "trail"

    for i in range(n):
        equity.iloc[i] = cash + (c[i] * pos - cost_basis) * pv
        do_exit = False
        if pos > 0:
            peak = max(peak, h[i])
            if trail and c[i] < peak * (1 - cfg["trail_drop"]):
                do_exit = True
            if (not trail) and i in sell_days:
                do_exit = True
        if do_exit and i + 1 < n:                                  # full exit at next open
            px = o[i + 1]
            pnl = (px * pos - cost_basis) * pv - roundtrip * pos
            cash += pnl
            trades.append(dict(exit=str(idx[i + 1].date()), lots=pos, pnl=round(pnl, 2),
                               entry=str(entry_first.date())))
            pos = 0; cost_basis = 0.0; peak = 0.0; entry_first = None
        if ((i in buy_days) and pos < max_lots and (i - last_buy) >= cfg["cooldown"] and i + 1 < n
                and (trade_start is None or idx[i] >= trade_start)):
            pos += 1; cost_basis += o[i + 1]; peak = max(peak, o[i + 1])
            last_buy = i
            if entry_first is None:
                entry_first = idx[i + 1]
    if pos > 0:
        equity.iloc[-1] = cash + (c[-1] * pos - cost_basis) * pv
    return df, equity, trades


def _live_sharpe(symbol: str) -> pd.Series:
    """Sharpe signal extended to the LATEST bar: OOS walk-forward for history,
    then fit_full (train-on-all, predict) for the post-last-fold tail the backtest
    folds don't cover. The OOS series alone lags by the purge (~last fold), so a
    live read of 'today' needs the fit_full tail."""
    close = data_loader.load_ohlc(symbol)["close"]
    hz = INSTRUMENTS[symbol]["horizon"]
    oos = model.walk_forward(symbol).oos_pred.dropna()
    mdl, cols, _ = model.fit_full(symbol)
    X = features.build_feature_matrix(symbol, dropna=True)[cols]
    tail = X[X.index > oos.index[-1]]
    pred = pd.concat([oos, pd.Series(mdl.predict(tail), index=tail.index)]) if len(tail) else oos
    fc = signal.horizon_forecast_vol(close, hz).reindex(pred.index)
    return (pred / fc).replace([np.inf, -np.inf], np.nan)


def live_signal(symbol: str, *, max_lots=MAX_LOTS) -> dict:
    """Today's action for the PRE-REGISTERED strategy (what to actually trade):
    walk the position to the latest bar and report the position held now + the
    action signalled today (executes at the next open). Uses fit_full-extended
    signals so 'today' is current, not the lagging backtest OOS prediction."""
    cfg = DESIGN[symbol]
    df, buy_days, sell_days = build_signals(symbol, cfg, sharpe_override=_live_sharpe(symbol))
    n = len(df); idx = df.index
    c, h = df["close"].to_numpy(float), df["high"].to_numpy(float)
    pos = last_buy = 0; last_buy = -10**9; peak = 0.0
    trail = cfg["sell"] == "trail"; drop = cfg.get("trail_drop", 0.08)
    pos_now = 0; pending = "HOLD"
    for i in range(n):
        if i == n - 1:
            pos_now = pos
        exited = entered = False
        if pos > 0:
            peak = max(peak, h[i])
            if (trail and c[i] < peak * (1 - drop)) or (not trail and i in sell_days):
                exited = True; pos = 0; peak = 0.0
        if (i in buy_days) and pos < max_lots and (i - last_buy) >= cfg["cooldown"]:
            pos += 1; last_buy = i; peak = max(peak, c[i]); entered = True
        if i == n - 1:
            pending = "EXIT ALL" if exited else ("BUY 1 lot" if entered else "HOLD")
    return dict(symbol=symbol, asof=str(idx[-1].date()), position_now=pos_now,
                pending_action=pending, position_after=pos,
                sharpe=round(float(df["sharpe"].iloc[-1]), 3), exit_rule=cfg["sell"])


def run(symbol: str, *, init_equity=INIT_EQUITY) -> dict:
    cfg = DESIGN[symbol]
    df, equity, trades = simulate(symbol, cfg, init_equity=init_equity)
    s = _ann_stats(equity)
    pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    win = float((pnls > 0).mean()) if len(pnls) else float("nan")
    # buy & hold baseline on the same window
    bh = (1 + df["close"].pct_change().fillna(0)).cumprod() * init_equity
    bhs = _ann_stats(bh)
    return dict(symbol=symbol, sharpe=s["sharpe"], maxdd=s["max_dd"], cagr=s["cagr"],
                vol=s["ann_vol"], n_trades=len(trades), win=win,
                bh_sharpe=bhs["sharpe"], bh_maxdd=bhs["max_dd"],
                period=f"{df.index[0].date()}..{df.index[-1].date()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="V1.6 signal-design backtest")
    ap.add_argument("--symbols", nargs="+", default=["ES", "GC"], choices=list(DESIGN))
    ap.add_argument("--equity", type=float, default=INIT_EQUITY)
    args = ap.parse_args()
    for sym in args.symbols:
        r = run(sym, init_equity=args.equity)
        print(f"\n=== {sym}  ({r['period']}) — {DESIGN[sym]['sell']} design ===")
        print(f"strategy: Sharpe {r['sharpe']:+.2f} | vol {r['vol']*100:.1f}% | maxDD {r['maxdd']*100:.1f}% | "
              f"CAGR {r['cagr']*100:+.1f}% | trades {r['n_trades']} | win {r['win']*100:.0f}%")
        print(f"buy&hold: Sharpe {r['bh_sharpe']:+.2f} | maxDD {r['bh_maxdd']*100:.1f}%")


if __name__ == "__main__":
    main()
