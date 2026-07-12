# GC Multi-Timeframe RSI Strategy (TradingView / Pine v6)

`gc_mtf_rsi.pine` — a long-only, multi-timeframe **RSI mean-reversion** strategy for
gold (GC), implementing the entry design in `../design_plan/desing.md` plus an
ATR-stop / RSI-overbought exit. Written as a `strategy()` so it backtests in the
TradingView **Strategy Tester** and can also fire entry/exit alerts.

## Why Pine instead of an IBKR data pipeline

The original design assumed pulling IBKR 5-minute bars. That path has two hard
blockers: IBKR's native API needs a running TWS/Gateway session (can't run in
GitHub Actions), and pulling/storing 5-minute history is heavy. TradingView already
serves every timeframe natively (`request.security`), computes the indicators, and
pushes alerts — so the whole data-collection problem disappears. You just attach the
script to a GC chart.

## Install

1. TradingView → **Pine Editor** → paste the contents of `gc_mtf_rsi.pine`.
2. **Add to chart.** Apply it to a gold futures symbol — `COMEX:GC1!` (full-size) or
   `COMEX:MGC1!` (micro; fits a small account).
3. Use a **5-minute** base chart. The 5 higher frames (10m/30m/1h/4h) and the 1m
   frame are pulled internally, so the chart TF only needs to be 5m.
   - Note on 1m: requesting 1m from a 5m chart returns the latest 1m bar's value per
     chart bar — fine for the "1m RSI ≤ 30" gate. For maximum 1m fidelity, run the
     chart at 1m; the higher frames still resolve correctly.

## Signal logic (matches the design)

**Entry (long)** = `(Gate A OR Gate B) AND Momentum gate`

- **Gate A — oversold ladder:** `1m RSI ≤ 30 AND 5m ≤ 30 AND 10m ≤ 30 AND 30m ≤ 35
  AND 1h ≤ 37 AND 4h ≤ 40`. (1m added per your follow-up.)
- **Gate B — bullish divergence:** a regular bullish RSI divergence on **5m / 10m /
  30m** (price lower-low while RSI higher-low, between two confirmed RSI pivot lows),
  while `1h RSI < 40 AND 4h RSI < 40`.
- **Momentum gate:** `5m, 10m, 30m` ROC "significantly positive" (`> momThr`) and
  `1h, 4h` "not negative" (`≥ momFloor`).

**Exit** = ATR stop **or** RSI overbought, whichever comes first:

- **ATR stop:** `entry − atrMult × ATR(atrLen)` on the chart TF.
- **RSI overbought take-profit:** `5m RSI ≥ 70 AND 10m RSI ≥ 70` → close.

Long-only, one position at a time (`pyramiding = 0`).

## RSI is SMA-smoothed on purpose

The design says "RSI using SMA, length 14". Standard RSI (and TradingView's built-in
`ta.rsi`) uses **Wilder's RMA** smoothing. This script computes RSI with **SMA** of
up/down moves instead, to match your spec — so its values differ slightly from the
built-in RSI indicator. That difference is expected, not a bug.

## Non-repainting

- All higher-timeframe reads use `lookahead = barmerge.lookahead_off` (no future
  leak) and, with **"Non-repainting HTF"** on (default), reference the **last closed**
  HTF bar (`[1]` offset) — so historical and live signals agree. Turn it off for a
  more responsive (but intrabar-repainting) live read.
- Divergence uses `ta.pivotlow`, which confirms a pivot only `Pivot right bars` later;
  the script references it only after confirmation, so it is causal.
- Strategy orders and alerts fire on **confirmed chart bars** only.

## Backtest

1. Open the **Strategy Tester** panel → check Overview / List of Trades.
2. Set realistic costs in **Strategy Tester → Settings → Properties**: commission
   (e.g. ~\$0.62/side for micros or your broker's rate) and slippage (1–2 ticks).
   Contract point value comes from the symbol itself (GC vs MGC), so pick the symbol
   that matches what you'd trade.
3. Sanity-check a few entries land on multi-TF oversold / divergence + momentum-up,
   and that exits are either the ATR stop or the RSI-overbought close.

> Small-account reality check: at ~\$1.7k equity even one MGC (~\$40k notional,
> ~\$2–4k day-trade margin) is a stretch and full GC is out of reach. Treat this as a
> signal/alert tool first; sizing to real contracts needs a larger account.

## Alerts

Two `alertcondition`s are exposed — **ENTRY** and **RSI overbought exit**. Create an
alert (right-click chart → Add alert, or the alarm icon), pick this script's
condition, and set **"Once Per Bar Close"** so alerts match the non-repainting
signals. The TradingView mobile app then pushes them to your phone. The strategy's
own `strategy.entry/exit` orders also generate order-fill alerts if you alert on
"Any alert() function call".

## Tuning

Every threshold, length, and gate is an input, grouped in the settings dialog:
RSI length; per-TF oversold thresholds (Gate A); divergence context + pivot/range
params (Gate B); momentum length/thresholds; ATR stop + overbought exit; and
execution (contracts, non-repaint, table). Start from the defaults, then adjust on
the Strategy Tester.

The status table (top-right) shows each TF's RSI, whether it passes its threshold,
its divergence/momentum state, and which gate is currently firing.

## Scope

GC only for now, per plan. To extend to ES/NQ later, the same script works on those
symbols; give each its own RSI thresholds (equity-index RSI behaves differently from
gold, so don't assume 30/30/35/37/40 transfers).
