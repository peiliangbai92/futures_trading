"""Multi-sleeve portfolio layer (V1.4) — combine complementary alpha sleeves.

The gap diagnosis left ES with a single long-only mean-reversion sleeve. Reversion
and trend pay off in opposite regimes (reversion in chop, trend in directional
markets), and on ES they are only ~+0.3 correlated, so an equal-risk combination
improves *both* Sharpe and matched-vol drawdown over either leg alone.

Sleeves produce daily RETURN streams; we combine at **risk parity** — scale each
to a common target vol (using the same close-to-close realized-vol formula the
rest of the system uses, ``vol.close_to_close_volatility``: √252 × stdev of 21d
daily log returns) then equal-weight. Sharpe is leverage-invariant so the scaling
only matters for the combined stream and for matched-vol drawdown comparison.

Legs:
  reversion : the production ridge sleeve (long-only ret_5/ret_20) run through the
              trade-level backtest (ATR stops, costs, vol-target sizing).
  trend     : vol-targeted time-series momentum, sign(ret_lookback), daily
              rebalanced, net of a per-turnover futures cost.

CLI:
    python -m futures_swing.portfolio --symbol ES
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import data_loader, model
from .backtest import INIT_EQUITY, _assemble, _ann_stats, simulate
from .vol import close_to_close_volatility

TRADING_DAYS = 252


# ------------------------------------------------------------------ sleeves


def reversion_returns(symbol: str, *, init_equity: float = INIT_EQUITY) -> pd.Series:
    """Daily returns of the production reversion sleeve (trade-level, net costs)."""
    wf = model.walk_forward(symbol)
    df = _assemble(symbol, wf.oos_pred)
    _, equity, _ = simulate(symbol, df, init_equity=init_equity)
    return equity.pct_change().dropna().rename("reversion")


def trend_returns(symbol: str, *, lookback: int = 252, target_vol: float = 0.10,
                  cost_bps: float = 1.0, max_leverage: float = 3.0) -> pd.Series:
    """Vol-targeted time-series momentum daily returns, net of turnover cost.

    Position (as a fraction of equity notional) = sign(ret_lookback) × target_vol
    / trailing close-to-close vol, leverage-capped. Entered with a 1-day lag;
    turnover (daily change in leverage) is charged ``cost_bps`` per unit."""
    close = data_loader.load_ohlc_model(symbol)["close"]
    es_ret = close.pct_change()
    mom = np.sign(np.log(close / close.shift(lookback)))
    rv = close_to_close_volatility(close, window=21)          # same c2c formula as sizing
    lev = (mom * (target_vol / rv)).clip(-max_leverage, max_leverage)
    gross = lev.shift(1) * es_ret
    cost = lev.diff().abs() * (cost_bps / 1e4)
    return (gross - cost).dropna().rename(f"trend{lookback}")


# ------------------------------------------------------------------ combine


def _scale_to_vol(r: pd.Series, target_vol: float) -> pd.Series:
    s = r.std() * np.sqrt(TRADING_DAYS)
    return r * (target_vol / s) if s > 0 else r


def risk_parity(streams: list[pd.Series], *, target_vol: float = 0.10,
                weights: list[float] | None = None) -> pd.Series:
    """Scale each stream to ``target_vol`` then weight (equal by default).
    The combined stream's realized vol is < target_vol when streams diversify."""
    df = pd.concat(streams, axis=1).dropna()
    w = np.array(weights) if weights is not None else np.ones(df.shape[1]) / df.shape[1]
    scaled = pd.concat([_scale_to_vol(df[c], target_vol) for c in df.columns], axis=1)
    return (scaled * w).sum(axis=1).rename("combo")


def _stats(r: pd.Series, target_vol: float = 0.10) -> dict:
    """Sharpe (scale-free) + drawdown re-levered to a common vol for fair compare."""
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else float("nan")
    natvol = r.std() * np.sqrt(TRADING_DAYS)
    eqm = (1 + _scale_to_vol(r, target_vol).fillna(0)).cumprod()
    dd_matched = (eqm / eqm.cummax() - 1).min()
    return dict(sharpe=float(sharpe), nat_vol=float(natvol), maxdd_matched=float(dd_matched))


# ------------------------------------------------------------------ run


def run_combined(symbol: str, *, lookback: int = 252, target_vol: float = 0.10) -> dict:
    rev = reversion_returns(symbol)
    trd = trend_returns(symbol, lookback=lookback, target_vol=target_vol)
    idx = rev.index.intersection(trd.index)
    rev, trd = rev.reindex(idx), trd.reindex(idx)
    combo = risk_parity([rev, trd], target_vol=target_vol)
    corr = float(rev.corr(trd))
    legs = {"reversion": rev, f"trend{lookback}": trd, "combo (risk-parity)": combo}
    return dict(symbol=symbol, lookback=lookback, corr=corr,
                stats={k: _stats(v, target_vol) for k, v in legs.items()},
                streams=legs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-sleeve portfolio combination")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--lookback", type=int, default=252, help="trend TSMOM lookback (days)")
    ap.add_argument("--target-vol", type=float, default=0.10)
    args = ap.parse_args()

    out = run_combined(args.symbol, lookback=args.lookback, target_vol=args.target_vol)
    print(f"\n=== {args.symbol}: reversion + trend{args.lookback} (risk-parity, "
          f"corr={out['corr']:+.2f}) ===")
    print(f"{'leg':22s} {'Sharpe':>7s} {'natVol':>7s} {'maxDD@%.0f%%' % (args.target_vol*100):>10s}")
    for name, s in out["stats"].items():
        print(f"{name:22s} {s['sharpe']:>+7.2f} {s['nat_vol']*100:>6.1f}% {s['maxdd_matched']*100:>+9.1f}%")


if __name__ == "__main__":
    main()
