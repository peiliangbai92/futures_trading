# Multi-Timeframe RSI Signal Indicator (TradingView / Pine v6)

`gc_mtf_rsi.pine` — a **symbol-agnostic**, multi-timeframe **RSI mean-reversion
signal indicator** implementing (and extending) the entry design in
`../docs/intraday_design.md`. It is an `indicator()`, not a `strategy()`: it
places no broker orders, but tracks a **virtual position** (flat / long / short)
on confirmed bars so entry/exit signals and alerts behave exactly like a
strategy's fills without repainting.

Everything is pulled for the *current chart symbol* via `syminfo.tickerid`, so
the same script runs on MGC, GC, ES, CL, or anything else — the default
thresholds were tuned for gold, so retune per instrument.

## Why Pine instead of an IBKR data pipeline

The original design assumed pulling IBKR 5-minute bars. That path has two hard
blockers: IBKR's native API needs a running TWS/Gateway session (can't run in
GitHub Actions), and pulling/storing 5-minute history is heavy. TradingView
already serves every timeframe natively (`request.security`), computes the
indicators, and pushes alerts — so the whole data-collection problem
disappears. You just attach the script to a chart.

## Install

1. TradingView → **Pine Editor** → paste the contents of `gc_mtf_rsi.pine`.
2. **Add to chart.** For gold: `COMEX:GC1!` (full-size) or `COMEX:MGC1!` (micro).
3. Use a **5-minute** base chart. The other frames (1m/10m/30m/1h/4h) are pulled
   internally via `request.security`; whichever slot equals the chart TF is
   computed natively (zero security lag).
   - Note on 1m: requesting 1m from a 5m chart returns the latest 1m bar's value
     per chart bar — fine for the "1m RSI ≤ 30" gate. For maximum 1m fidelity,
     run the chart at 1m.

## Signal logic

**Long entry** = `((Gate A OR Gate B) AND Momentum gate) OR 4h-BB lower touch`,
all subject to the **trend filter**.

- **Gate A — oversold ladder:** every **ticked** timeframe must be oversold
  (unticked TFs are ignored). Defaults: `1m ≤ 30, 5m ≤ 30, 10m ≤ 30, 30m ≤ 35,
  1h ≤ 37, 4h ≤ 40`, all ticked.
- **Gate B — bullish divergence:** a regular bullish RSI divergence on
  **5m / 10m / 30m** (price lower-low while RSI higher-low, between two
  confirmed RSI pivot lows), while `1h RSI < 40 AND 4h RSI < 40`. A divergence
  stays "active" for `divHold` bars (default 3) after confirmation.
- **Momentum gate:** `5m, 10m, 30m` ROC `> momThr` and `1h, 4h ≥ momFloor`.
- **Bollinger trigger:** the bar's low touching the **4h lower band** is an
  extra long trigger that bypasses the RSI gates (configurable TF/length/mult).
- **Trend filter:** entries are blocked *against* a confirmed higher-TF trend
  (default 4h EMA-50 + slope): no longs in a confirmed downtrend, no shorts in
  a confirmed uptrend.

**Short entry** (on by default, `useShort`) is the mirror: every ticked TF
**overbought** (defaults `1m ≥ 70, 5m ≥ 70, 10m ≥ 70, 30m ≥ 65, 1h ≥ 63,
4h ≥ 60`) **or** a 4h-BB **upper** touch, subject to the trend filter. There is
no short-side divergence/momentum gate.

**Exits** (whichever fires first, evaluated on confirmed bars):

- **ATR stop:** initial stop at `entry ∓ 2.0 × ATR(14)` (chart TF).
- **Breakeven:** once `1.0 × ATR` in profit, the stop locks to entry.
- **ATR trail:** the stop then trails price by `2.0 × ATR` (only ever moves in
  your favour).
- **Take-profit:** fixed at `entry ± 3.0 × ATR`.
- **Opposite RSI signal** (`useObExit`): the ticked-ladder *overbought* set
  closes a long; the *oversold* Gate A set closes a short.
- **Opposite BB touch:** upper-band touch closes a long, lower-band touch
  closes a short.

## RSI is SMA-smoothed on purpose

The design says "RSI using SMA, length 14". Standard RSI (and TradingView's
built-in `ta.rsi`) uses **Wilder's RMA** smoothing. This script computes RSI
with **SMA** of up/down moves instead, to match the spec — so its values differ
slightly from the built-in RSI indicator. That difference is expected, not a bug.

## Non-repainting

- All higher-timeframe reads use `lookahead = barmerge.lookahead_off` and, with
  **"Non-repainting HTF"** on (default), reference the **last closed** HTF bar
  (`[1]` offset). The chart's own TF is computed natively with no offset, so
  the base signal fires on the very bar that triggers it.
- Divergence uses `ta.pivotlow`, which confirms a pivot only `Pivot right bars`
  later; the script references it only after confirmation, so it is causal.
- The virtual-position state machine and all alerts run on **confirmed chart
  bars** only (`barstate.isconfirmed`).

## Alerts

Six `alertcondition`s: **LONG ENTRY**, **SHORT ENTRY**, **CLOSE LONG**,
**CLOSE SHORT**, plus combined **ANY ENTRY** / **ANY EXIT**. Create an alert
(right-click chart → Add alert), pick this script's condition, and set **"Once
Per Bar Close"** so alerts match the non-repainting signals. The TradingView
mobile app pushes them to your phone. Alert messages use `{{ticker}}` /
`{{interval}}` / `{{close}}` placeholders, so one alert per chart works across
symbols.

## Backtesting note

An `indicator()` does **not** run in the Strategy Tester. To backtest, either
eyeball the plotted entry/exit markers + stop/target lines on history, or
temporarily convert: change `indicator(...)` to `strategy(...)` and replace the
state-machine assignments with `strategy.entry/close` calls. The virtual
position logic (enter only when flat, ATR stop/target, breakeven, trail) was
written to mirror strategy semantics one-to-one.

> Small-account reality check: at ~$1.7k equity even one MGC (~$40k notional,
> ~$2–4k day-trade margin) is a stretch and full GC is out of reach. Treat this
> as a signal/alert tool first; sizing to real contracts needs a larger account.

## Tuning

Every threshold, length, and gate is an input, grouped in the settings dialog:
RSI length; per-TF tick-boxes + thresholds for the Gate A entry ladder and the
overbought exit ladder; divergence context + pivot/range params (Gate B);
momentum length/thresholds; ATR stop/target/breakeven/trail multiples; trend
filter TF/length; Bollinger TF/length/mult; and execution toggles (shorts,
non-repaint, status table).

The status table (top-right) shows each TF's RSI, whether it passes its Gate A
threshold (• marks ticked TFs), its divergence/momentum state, and which
signal (Gate A / overbought / momentum gate) is currently firing.

## Scope

Defaults are tuned for gold. The script itself is symbol-agnostic — to use on
ES/NQ/CL, retune the per-TF thresholds (equity-index RSI behaves differently
from gold, so don't assume 30/30/30/35/37/40 transfers).
