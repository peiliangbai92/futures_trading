# futures_swing — ES & GC systematic swing-trading model (V1)

A systematic swing-trading pipeline for **ES** (E-mini S&P 500) and **GC** (Gold)
futures, holding ~2–15 days. It forecasts a forward return, risk-adjusts it to a
signal, attaches ATR-based entry/stop/target and micro-contract sizing, and
validates everything with a purged walk-forward backtest.

```
data_loader → features → regime → model (LightGBM alpha) → signal (Sharpe)
            → execution (entry/stop/target) → risk (sizing + gates)
            → backtest (walk-forward)  +  pipeline (daily signal)
```

ES and GC are modeled **independently** (different macro drivers), unified only at
the risk layer.

## Scope (V1)

- **Data: yfinance daily only** — `ES=F, GC=F, ^VIX, ^VVIX, DX-Y.NYB, ^TNX, TIP, CL=F`.
- **Contracts: micros** — MES ($5/pt), MGC ($10/pt) — so 0.5–1% per-trade risk
  sizing is realistic on a mid-size account.
- **Excluded (→ V2):** option-flow features (dealer gamma / GEX — no usable
  history), FRED macro (2Y, real yield, breakeven, HY/IG OAS), intraday/hourly,
  Databento data, live execution. See the V2 roadmap below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**macOS only:** LightGBM needs the OpenMP runtime:

```bash
brew install libomp
```

## Usage

```bash
# 1. pull / refresh the data cache (writes data/raw/*.parquet)
python -m futures_swing.data_loader            # period=max

# 2. inspect forecast quality (purged walk-forward CV)
python -m futures_swing.model --symbol ES
python -m futures_swing.model --symbol GC

# 3. full backtest -> reports/<SYM>/backtest.md + trades/equity CSVs
python -m futures_swing.backtest --symbols ES GC

# 4. today's signal -> data/signals/latest_signal.csv
python -m futures_swing.pipeline --equity 50000

# tests (no-lookahead + alignment guarantees)
pytest -q
```

## How it works

- **features.py** — point-in-time matrix: multi-horizon returns, MA distances,
  Yang-Zhang vol, ATR, cross-asset macro (levels/returns/changes), regime code.
  The forward-return *target* is deliberately separate (`forward_log_return`).
- **regime.py** — rule-based market regime (risk_on / range / risk_off / stress)
  from ES-vs-MA100 and VIX. Pluggable for GMM/HMM/Markov-switching in V2.
- **model.py** — LightGBM forecast with **purged + embargoed walk-forward CV**.
  Overlapping forward returns shrink the *effective* sample to ~ n_days/horizon,
  so the CV purges training rows whose label window overlaps the test block and
  reports the in-sample vs out-of-sample IC gap as an overfit check.
- **signal.py / execution.py / risk.py** — Sharpe signal → ATR stop/target →
  risk-based micro-contract sizing with drawdown gates and an event filter.
- **backtest.py** — day-by-day trade sim (next-open entry, costs) with metrics
  and **baseline comparison** (buy & hold, 12-1 momentum).

## Honest results (OOS, net of micro costs; per-symbol regime, 10% vol-target sizing, $220k acct)

| | ES (5d) | GC (10d) |
|---|---|---|
| OOS rank IC | ~0.015 | ~0.07 |
| Strategy Sharpe | ~0.43 | ~0.57 |
| Strategy maxDD | ~−5% | ~−9% |
| Buy & hold Sharpe | ~0.55 | ~0.69 |
| Buy & hold maxDD | ~−57% | ~−44% |

The features carry only a **weak forecast** edge (OOS IC: GC > ES), yet the
*strategy* lands close to buy-and-hold on Sharpe (0.43/0.57 vs 0.55/0.69) at
roughly **one-sixth the drawdown** (≈−5/−9% vs −44/−57%) — the payoff of
vol-target sizing + regime/conviction gating, not forecast skill. It still
doesn't *beat* buy-and-hold risk-adjusted (levered to B&H vol it earns slightly
less CAGR), but it's a credible low-drawdown profile. Treat it as validated
plumbing with a small real edge; the larger forecast edge is still ahead.

> Account-size matters with micros: at $50k, vol-target sizing rounds to 0–1
> contracts (coarse, GC sub-min-size) and Sharpe drops to ~0.3/0.4; at $220k the
> integer rounding barely distorts and sizing expresses fully. Default equity is
> $220k; set `--equity` to your account.

### V1.1 — FRED macro factors (tested, off by default)

Added keyless FRED factors (2Y, 2s10s curve, 10Y real yield, breakeven inflation)
plus a HYG-minus-LQD credit-appetite proxy, and measured the OOS IC delta:

```
python -m futures_swing.model --symbol ES --compare-fred
python -m futures_swing.model --symbol GC --compare-fred
```

| Δ vs V1 | ES (5d) | GC (10d) |
|---|---|---|
| Δ OOS IC | +0.002 | −0.003 |
| Δ IS-OOS gap | +0.043 (worse) | +0.052 (worse) |

**FRED factors did not improve out-of-sample IC** for either instrument and
*widened* the overfit gap. The seductive part: in a full-sample univariate scan,
GC's `real_yield_level` (+0.13) and ES's `ust2y_chg20` (−0.09) rank as the single
strongest features — but the strong ones are non-stationary **levels** whose
in-sample IC is inflated by common trending, which is exactly why they fail to
generalize. The purged walk-forward caught the trap. FRED is therefore kept
behind `include_fred=` (default **off**); extracting real macro edge needs
stationary transforms and regime-conditioning, not raw features — that's V2.

### V1.2 — HMM regime (adopted, default)

Replaced the rule-based regime feature with a **3-state Gaussian HMM** on the
market (ES return, short realized vol, log-VIX). Lookahead is handled rigorously:
HMM params are fit once on an initial window strictly before the OOS region, and
the per-day state is a **causal forward-filtered posterior** (uses only obs ≤ t,
not Viterbi/smoothing). A unit test proves the posterior at a date is identical
with or without future data.

```
python -m futures_swing.model --symbol ES --compare-regime
```

| rule → HMM | ES (5d) | GC (10d) |
|---|---|---|
| OOS IC | +0.004 → **+0.013** | +0.070 → +0.069 |
| IS-OOS gap | ~flat | ~flat |

The ES gain is **robust** — every (n_states ∈ {2,3,4}, seed ∈ {0,42,7}) config
beats the rule baseline (+0.008…+0.027), so it's signal not seed-luck. ES is
still only weakly predictive in absolute terms — a real but small step.

**Per-asset HMM (tested, not adopted for GC).** A gold-specific HMM (GC return +
short vol + real-yield change + DXY change) was tested for GC and **did not
help** — gold-HMM +0.063 is *worse* than rule (+0.070) and market-HMM (+0.072),
across all seeds/states. GC's edge is in its price/trend features, not regime.
So the regime default is **per-symbol** (`INSTRUMENTS[sym]["regime"]`): ES → HMM,
GC → rule. Gold obs remain behind `hmm_kwargs={"obs_source":"gold"}`.

### V1.3 — Vol-target sizing (adopted, default)

Replaced pure stop-risk sizing with **inverse-vol notional targeting** (size to
a 10% annualized per-trade vol, scaled by conviction). This both deploys capital
and *improves* risk-adjusted return — sizing up in calm regimes and down in
turbulent ones. Default `sizing_mode="vol_target"`, `target_vol=0.10` (`"risk"`
mode kept). At a $220k account this lifts Sharpe to ~0.43 (ES) / ~0.57 (GC) and
holds drawdown to ~−5/−9%.

**Account-size note:** with micros, integer-contract rounding interacts with
sizing. One MGC at gold ~$4,000 is ~$40k notional, so a $50k account is
*sub-min-size* for GC at a 10% vol target and coarse for ES; a $220k account
sizes both cleanly and is the default. Use `--equity` to reflect your account.

> Caveat: Yahoo continuous futures carry un-adjusted quarterly roll jumps (small
> artifact in multi-day returns); Databento continuous contracts fix this in V2.

## V2 roadmap

Databento + IBKR live data/execution; FRED macro factors; option-flow features
once history accrues; GMM→HMM→Markov-switching/FAVAR regime; pullback/breakout
entries; portfolio optimizer; pullback execution; better roll handling.

## Reuse

Vol estimators in `vol.py` are vendored from the sibling `QuantitativeResearch`
repo (`option_research/realized_vol.py`); the no-lookahead walk-forward idiom and
the macro event calendar are adapted from the same project.
