"""Walk-forward backtest — turn OOS forecasts into trades, PnL, and metrics.

Pipeline: model.walk_forward (purged OOS forecasts) -> signal.compute_signals
-> day-by-day trade simulation (next-open entry, ATR stop/target, time stop,
signal-reversal exit) with **micro-contract costs** and risk-based sizing ->
metrics + baselines (buy & hold, 12-1 momentum). Acting only on the OOS period
keeps the evaluation honest.

CLI:
    python -m futures_swing.backtest --symbols ES GC
Outputs:
    reports/<SYM>/backtest.md, trades_<SYM>.csv, equity_<SYM>.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import INSTRUMENTS, data_loader, model, regime, signal
from . import vol as volmod
from .execution import hit_exit, levels
from .risk import (
    TARGET_VOL,
    RiskManager,
    conviction,
    load_event_dates,
    near_event,
    position_size,
    vol_target_size,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"

INIT_EQUITY = 220_000.0
COMMISSION_PER_SIDE = 0.62   # ~micro futures commission per contract per side
SLIPPAGE_TICKS = 1.0         # assumed slippage per side, in ticks
TRADING_DAYS = 252


# ------------------------------------------------------------------ assembly


def _assemble(symbol: str, pred: pd.Series) -> pd.DataFrame:
    """OHLC + ATR + signal + sharpe + regime, aligned to the OOS dates."""
    ohlc = data_loader.load_ohlc_model(symbol)
    atr = volmod.atr(ohlc, window=14)
    sig = signal.compute_signals(symbol, pred.dropna())
    reg = regime.classify(data_loader.load_close("ES"), data_loader.load_close("VIX"))

    df = pd.DataFrame(index=sig.index)
    for col in ("open", "high", "low", "close"):
        df[col] = ohlc[col].reindex(df.index)
    df["atr"] = atr.reindex(df.index)
    df["ann_vol"] = volmod.close_to_close_volatility(ohlc["close"], window=21).reindex(df.index)
    df["signal"] = sig["signal"]
    df["sharpe"] = sig["sharpe"]
    df["regime"] = reg.reindex(df.index, method="ffill")
    return df.dropna(subset=["open", "high", "low", "close", "atr", "ann_vol"])


# ------------------------------------------------------------------ simulate


def simulate(
    symbol: str, df: pd.DataFrame, *, init_equity: float = INIT_EQUITY,
    sizing_mode: str = "vol_target", target_vol: float = TARGET_VOL,
    exit_override: dict | None = None,
) -> tuple[list[dict], pd.Series, int]:
    spec = INSTRUMENTS[symbol]
    pv, tick, horizon = spec["point_value"], spec["tick"], spec["horizon"]
    target_vol = spec.get("target_vol", target_vol)   # per-symbol de-rate (GC runs at 0.05)
    roundtrip_cost = 2 * COMMISSION_PER_SIDE + 2 * SLIPPAGE_TICKS * tick * pv  # $ / contract
    events = load_event_dates()
    # Exit policy (V1.5). "atr": fixed ATR stop/target + horizon time-stop (default,
    # GC). "trail": initial ATR stop, then a peak-following trailing stop (rides the
    # rally, sells on the pullback). "signal": reversion sell-high — hold the long
    # until the (smoothed) forecast flips to overbought (sharpe <= -signal_exit_th),
    # keeping the protective ATR stop and a max-hold backstop.
    exit_cfg = exit_override or spec.get("exit", {"mode": "atr"})
    exit_mode = exit_cfg.get("mode", "atr")
    trail_mult = float(exit_cfg.get("trail_mult", 3.0))
    max_hold = int(exit_cfg.get("max_hold", horizon))
    signal_exit_th = float(exit_cfg.get("signal_exit_th", spec.get("signal_th", 0.35)))
    use_stop = bool(exit_cfg.get("use_stop", True))   # signal mode: protective ATR stop on/off
    trend_ma_win = int(exit_cfg.get("trend_ma_win", 50))  # trend_stop mode: the stop MA
    entry_filter = exit_cfg.get("entry_filter")       # "trend_up" -> only buy above the trend MA

    idx = df.index
    n = len(idx)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sig, shp = df["atr"].to_numpy(float), df["signal"].to_numpy(int), df["sharpe"].to_numpy(float)
    annv = df["ann_vol"].to_numpy(float)
    reg = df["regime"].to_numpy(object)
    trend_ma = df["close"].rolling(trend_ma_win, min_periods=trend_ma_win).mean().to_numpy()

    cash = init_equity
    equity = pd.Series(init_equity, index=idx, dtype=float)
    rm = RiskManager(init_equity)
    trades: list[dict] = []
    days_in_pos = 0

    in_pos = False
    side = contracts = entry_i = 0
    entry_price = stop = target = entry_sharpe = 0.0
    peak = trough = entry_atr = 0.0
    entry_date = entry_regime = None

    def close_trade(i: int, price: float, reason: str) -> None:
        nonlocal cash, in_pos
        gross = (price - entry_price) * side * pv * contracts
        pnl = gross - roundtrip_cost * contracts
        cash += pnl
        rm.update(cash)
        risk_dollars = (stop - entry_price) * -side * pv * contracts  # positive stop loss size
        trades.append(
            dict(
                symbol=symbol, side=side, contracts=contracts,
                entry_date=str(entry_date.date()), entry_price=round(entry_price, 4),
                exit_date=str(idx[i].date()), exit_price=round(price, 4),
                bars_held=i - entry_i, reason=reason,
                pnl_dollar=round(pnl, 2),
                r_multiple=round(pnl / risk_dollars, 3) if risk_dollars > 0 else float("nan"),
                entry_sharpe=round(entry_sharpe, 3), regime=entry_regime,
            )
        )
        in_pos = False

    for i in range(n):
        if in_pos:
            if exit_mode == "trail":
                peak = max(peak, h[i]); trough = min(trough, l[i])
                if side > 0:
                    eff_stop = max(stop, peak - trail_mult * entry_atr)  # ratchets up
                    reason, price = hit_exit(side, h[i], l[i], eff_stop, float("inf"))
                else:
                    eff_stop = min(stop, trough + trail_mult * entry_atr)
                    reason, price = hit_exit(side, h[i], l[i], eff_stop, float("-inf"))
                if reason is None and (i - entry_i) >= max_hold:
                    reason, price = "time", c[i]
            elif exit_mode == "signal":
                # reversion sell-high: hold until the forecast flips to overbought
                # (long) / oversold (short); the overbought signal sits at ~64th
                # local pct, so this sells high IF we don't get stopped out first.
                eff_stop = stop if use_stop else (float("-inf") if side > 0 else float("inf"))
                reason, price = hit_exit(side, h[i], l[i], eff_stop,
                                         float("inf") if side > 0 else float("-inf"))
                if reason is None and ((side > 0 and shp[i] <= -signal_exit_th) or
                                       (side < 0 and shp[i] >= signal_exit_th)):
                    reason, price = "signal", c[i]
                if reason is None and (i - entry_i) >= max_hold:
                    reason, price = "time", c[i]
            elif exit_mode == "trend_stop":
                # the pairing: reversion TRIGGERS the sell (overbought = sell high),
                # the TREND decides the stop — hold through dips while price holds
                # the trend MA, cut fast when the uptrend breaks (kills falling knives).
                reason = price = None
                if side > 0 and np.isfinite(trend_ma[i]) and c[i] < trend_ma[i]:
                    reason, price = "trend_stop", c[i]
                elif side < 0 and np.isfinite(trend_ma[i]) and c[i] > trend_ma[i]:
                    reason, price = "trend_stop", c[i]
                if reason is None and ((side > 0 and shp[i] <= -signal_exit_th) or
                                       (side < 0 and shp[i] >= signal_exit_th)):
                    reason, price = "signal", c[i]
                if reason is None and (i - entry_i) >= max_hold:
                    reason, price = "time", c[i]
            else:
                reason, price = hit_exit(side, h[i], l[i], stop, target)
                if reason is None and (i - entry_i) >= horizon:
                    reason, price = "time", c[i]
            if reason is None and sig[i] != 0 and np.sign(sig[i]) != side:
                reason, price = "reversal", c[i]
            if reason is not None:
                close_trade(i, price, reason)

        if in_pos:
            days_in_pos += 1
            equity.iloc[i] = cash + (c[i] - entry_price) * side * pv * contracts
        else:
            equity.iloc[i] = cash

        entry_ok = (entry_filter != "trend_up") or (np.isfinite(trend_ma[i]) and c[i] > trend_ma[i])
        if (not in_pos) and entry_ok and i + 1 < n and sig[i] != 0 and atr[i] > 0:
            mult = rm.size_multiplier(cash)
            if mult > 0 and not near_event(idx[i + 1], events):
                if sizing_mode == "vol_target":
                    qty = int(vol_target_size(cash, annv[i], o[i + 1], pv, target_vol=target_vol,
                                              conviction_mult=conviction(shp[i])) * mult)
                else:
                    qty = int(position_size(cash, shp[i], atr[i], pv) * mult)
                if qty > 0:
                    side = int(sig[i]); contracts = qty; entry_price = o[i + 1]
                    lv = levels(side, entry_price, atr[i])
                    stop, target = lv["stop"], lv["target"]
                    peak = trough = entry_price; entry_atr = atr[i]
                    entry_i = i + 1; entry_date = idx[i + 1]
                    entry_sharpe = shp[i]; entry_regime = reg[i]
                    in_pos = True

    if in_pos:
        close_trade(n - 1, c[n - 1], "eod")
        equity.iloc[n - 1] = cash

    return trades, equity, days_in_pos


# ------------------------------------------------------------------ metrics


def _ann_stats(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * math.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std() * math.sqrt(TRADING_DAYS) if ret.std() > 0 else float("nan")
    downside = ret[ret < 0].std()
    sortino = ret.mean() / downside * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    return dict(total_return=equity.iloc[-1] / equity.iloc[0] - 1, cagr=cagr, ann_vol=vol,
                sharpe=sharpe, sortino=sortino, max_dd=max_dd, calmar=calmar, years=years)


def _trade_stats(trades: list[dict], n_days: int, days_in_pos: int) -> dict:
    if not trades:
        return dict(n_trades=0, win_rate=float("nan"), avg_win=float("nan"),
                    avg_loss=float("nan"), exposure=0.0, turnover=0.0)
    pnls = np.array([t["pnl_dollar"] for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    years = max(n_days / TRADING_DAYS, 1e-9)
    return dict(
        n_trades=len(trades),
        win_rate=len(wins) / len(trades),
        avg_win=float(wins.mean()) if len(wins) else float("nan"),
        avg_loss=float(losses.mean()) if len(losses) else float("nan"),
        exposure=days_in_pos / n_days if n_days else 0.0,
        turnover=len(trades) / years,
    )


def _hit_by_regime(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades)
    g = t.assign(win=t["pnl_dollar"] > 0).groupby("regime")
    return pd.DataFrame({"n": g.size(), "win_rate": g["win"].mean(), "pnl": g["pnl_dollar"].sum()}).round(3)


def _baselines(close: pd.Series, oos_index: pd.Index) -> dict:
    """Buy & hold and 12-1 (120d) time-series momentum on the OOS window."""
    c = close.reindex(close.index)
    ret = c.pct_change()
    bh = (1 + ret.reindex(oos_index).fillna(0)).cumprod()
    mom = np.sign(np.log(c / c.shift(120)))
    mom_ret = (mom.shift(1) * ret).reindex(oos_index).fillna(0)
    mom_eq = (1 + mom_ret).cumprod()
    return {"buy_hold": _ann_stats(bh), "momentum_12_1": _ann_stats(mom_eq)}


# ------------------------------------------------------------------ run


def run(symbol: str, *, init_equity: float = INIT_EQUITY, write: bool = True,
        sizing_mode: str = "vol_target", target_vol: float = TARGET_VOL) -> dict:
    wf = model.walk_forward(symbol)
    df = _assemble(symbol, wf.oos_pred)
    trades, equity, days_in_pos = simulate(symbol, df, init_equity=init_equity,
                                           sizing_mode=sizing_mode, target_vol=target_vol)

    stats = _ann_stats(equity)
    tstats = _trade_stats(trades, len(df), days_in_pos)
    base = _baselines(data_loader.load_ohlc_model(symbol)["close"], df.index)
    regime_hits = _hit_by_regime(trades)
    # Fair comparison: Sharpe is leverage-invariant, so compare the strategy
    # *levered to buy & hold's volatility* — CAGR ~= Sharpe x vol at that vol.
    bh = base["buy_hold"]
    matched_cagr = stats["sharpe"] * bh["ann_vol"]
    summary = dict(symbol=symbol, horizon=wf.horizon, oos_ic=wf.oos_ic, oos_hit=wf.oos_hit,
                   is_oos_gap=wf.is_oos_gap, period=f"{df.index[0].date()}..{df.index[-1].date()}",
                   sizing_mode=sizing_mode, matched_cagr=matched_cagr,
                   bh_cagr=bh["cagr"], bh_vol=bh["ann_vol"], **stats, **tstats)

    if write:
        out = REPORTS / symbol
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trades).to_csv(out / f"trades_{symbol}.csv", index=False)
        equity.rename("equity").to_csv(out / f"equity_{symbol}.csv")
        _write_report(out / "backtest.md", symbol, summary, base, regime_hits, wf)
    summary["_baselines"] = base
    summary["_regime_hits"] = regime_hits
    return summary


def _fmt(stats: dict) -> str:
    return (f"CAGR {stats['cagr']*100:+.1f}%  vol {stats['ann_vol']*100:.1f}%  "
            f"Sharpe {stats['sharpe']:+.2f}  Sortino {stats['sortino']:+.2f}  "
            f"maxDD {stats['max_dd']*100:.1f}%  Calmar {stats['calmar']:.2f}")


def _write_report(path: Path, symbol: str, s: dict, base: dict, regime_hits: pd.DataFrame, wf) -> None:
    lines = [
        f"# {symbol} swing backtest (walk-forward, OOS)", "",
        f"- Period: {s['period']}  (horizon {s['horizon']}d)",
        f"- Forecast quality: OOS IC {wf.oos_ic:+.3f}, OOS hit {wf.oos_hit:.3f}, "
        f"IS-OOS IC gap {wf.is_oos_gap:+.3f} "
        f"({'large — capacity-overfit (benign, see GAP_DIAGNOSIS.md)' if wf.is_oos_gap > 0.15 else 'small — no capacity overfit'})",
        f"- Effective N {wf.effective_n:.0f} on {wf.n_features} features", "",
        f"## Strategy (net of micro-contract costs; sizing={s['sizing_mode']})",
        f"- {_fmt(s)}",
        f"- Total return {s['total_return']*100:+.1f}%  | trades {s['n_trades']}  "
        f"win {s['win_rate']*100:.0f}%  | exposure {s['exposure']*100:.0f}%  turnover {s['turnover']:.1f}/yr",
        f"- avg win ${s['avg_win']:.0f} | avg loss ${s['avg_loss']:.0f}", "",
        "## Baselines (same OOS window)",
        f"- Buy & hold:    {_fmt(base['buy_hold'])}",
        f"- 12-1 momentum: {_fmt(base['momentum_12_1'])}", "",
        "## Fair comparison (leverage-invariant)",
        f"- Sharpe doesn't change with leverage, so at buy & hold's {s['bh_vol']*100:.0f}% vol the "
        f"strategy would earn ~{s['matched_cagr']*100:+.1f}% CAGR vs buy & hold's {s['bh_cagr']*100:+.1f}%.",
        f"- Verdict: {'strategy beats' if s['matched_cagr'] > s['bh_cagr'] else 'buy & hold beats strategy'} "
        f"on risk-adjusted return.", "",
        "## Hit rate by regime",
        "```", regime_hits.to_string() if not regime_hits.empty else "_no trades_", "```", "",
        "_Caveat: Yahoo continuous-contract roll jumps are not back-adjusted; "
        "costs modeled as commission + 1-tick slippage per side._",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--symbols", nargs="+", default=list(INSTRUMENTS), choices=list(INSTRUMENTS))
    ap.add_argument("--equity", type=float, default=INIT_EQUITY)
    args = ap.parse_args()

    for sym in args.symbols:
        s = run(sym, init_equity=args.equity)
        print(f"\n=== {sym} ({s['period']}, horizon {s['horizon']}d) ===")
        print(f"forecast: OOS IC {s['oos_ic']:+.3f} | hit {s['oos_hit']:.3f} | IS-OOS gap {s['is_oos_gap']:+.3f}")
        print(f"strategy: {_fmt(s)}  [sizing={s['sizing_mode']}]")
        print(f"          total {s['total_return']*100:+.1f}% | trades {s['n_trades']} | "
              f"win {s['win_rate']*100:.0f}% | exposure {s['exposure']*100:.0f}%")
        print(f"buy&hold: {_fmt(s['_baselines']['buy_hold'])}")
        print(f"momentum: {_fmt(s['_baselines']['momentum_12_1'])}")
        print(f"fair    : at B&H {s['bh_vol']*100:.0f}% vol, strategy ~{s['matched_cagr']*100:+.1f}% CAGR "
              f"vs B&H {s['bh_cagr']*100:+.1f}% (Sharpe is leverage-invariant)")


if __name__ == "__main__":
    main()
