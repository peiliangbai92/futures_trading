# V1.8 — expansion studies: ES features, short sleeve, NQ (2026-08-08)

Three pre-specified studies, all on the **identical purged walk-forward
protocol** (expanding folds, 252d test blocks, purge gap = horizon + embargo;
OOS 2005-06..2026-07, data through 2026-08-07), with the decision rule fixed
in advance: **adopt only if pooled OOS IC beats the baseline with a
zero-excluding circular-block-bootstrap 95% CI AND the cost-aware trade sim
(V1.6 conventions: next-open fills, 1 micro lot, max 2, commission + 1-tick
slippage) does not degrade.** All three came back negative. Nothing was
adopted; the evidence is recorded here so the questions don't get re-litigated
from scratch.

## Study 1 — ES expanded features

Question: is ridge on `ret_5`/`ret_20` leaving alpha on the table? Candidates
added futures–cash **basis** features (`basis_pct`, `basis_chg5`, new in the
matrix), macro, regime (HMM posterior), and the full 25-feature set.

| config | OOS IC [95% CI] | gap | sim Sharpe | trades |
|---|---|---|---|---|
| **ridge-2 (baseline)** | **+0.056 [+0.010, +0.102]** | −0.04 | **0.42** | 44 |
| ridge-2 + basis | +0.064 [+0.020, +0.106] | +0.00 | 0.39 | 83 |
| ridge-2 + macro-mini | +0.054 [+0.009, +0.099] | −0.00 | 0.39 | 77 |
| ridge-all (25f) | +0.038 [−0.009, +0.085] | +0.15 | 0.31 | 119 |
| lgbm-all (25f) | +0.037 [−0.009, +0.086] | +0.34 | 0.42 | 115 |
| ridge-2 + hmm_p2 | +0.043 [−0.004, +0.089] | −0.02 | 0.47 | 47 |

- The only IC "improvement" (basis, +0.0088) is noise under a **paired**
  block bootstrap: ΔIC CI [−0.036, +0.054], P(better) = 0.65 — and it *costs*
  Sharpe (0.42 → 0.39) by nearly doubling turnover. Its IC advantage lives in
  the early half and evaporates recently.
- The V1.4 dilution finding **replicates on fresh data**: full-feature models
  halve the IC and blow up the IS-OOS gap (lgbm +0.34).
- The regime interaction *hurts* the linear sleeve (paired ΔIC −0.0125,
  P(better) = 0.07); its higher sim Sharpe on 47 trades is selection noise.
- **Watchlist**: `ridge-2 + macro-mini` has the strongest recent-half IC of
  any config (+0.087 vs baseline +0.054). Not adoptable on pooled evidence —
  flag for a **pre-registered re-test in 6–12 months**.

## Study 2 — short-side entries (ES and GC)

Question: is there a tradeable short rule mirroring the V1.6 long
architecture? A-priori 72-cell grid per symbol: cover ∈ {mom20 up-cross,
8% trail off trough, time-stop} × th ∈ {0.15, 0.25, 0.35} × cooldown ∈
{15, 20} × regime filter (risk_off/stress) on/off × trend filter (< MA100)
on/off.

- **ES: structurally untradeable.** The vol-scaled forecast spent 26 of 5,332
  OOS days below −0.15 (once below −0.20) and the left tail is
  **anti-predictive**: the most-bearish days precede **+0.89%** mean 5d
  forward returns vs +0.18% unconditional (35% hit-negative). All 6 traded
  cells negative; the honest symmetric short made 1 trade in 21 years and
  lost.
- **GC: fails every gate.** Honest symmetric short −$6.5k (Sharpe −0.14,
  negative in both halves). Grid median traded Sharpe **−0.20**, only 2/66
  cells positive; the best cell (+0.03) has CI [−0.46, +0.30] with a negative
  early half → the pre-registered **noise-mining guard triggered**. Every
  family median (all covers, thresholds, cooldowns, filters) is negative;
  the trend filter actively hurts (−0.36 vs −0.10).
- Diagnosis: **both models' OOS IC lives entirely on the long side** (GC's
  most-bearish forecasts precede *average* returns), consistent with V1.4's
  short legs −$4.5k vs long legs +$38k. **Stay long-only.**

## Study 3 — NQ onboarding

Question: does NQ (MNQ) deserve a live sleeve? Scaffolding (data, ^NDX cash
anchor, roll cleaner, basis features) landed first and is **kept**.

- **Data sanity: clean.** 90 rolls detected (median 106bp), adjusted tail
  equals raw, residual out-of-window divergences are self-cancelling ±pairs
  from high-vol close-timestamp mismatch (2000, 2008), not missed rolls.
- **No model class is significant**: best is ridge-2+basis at
  +0.038 [−0.008, +0.083] (p≤0 = 0.05) — about half of ES's edge; ridge-all /
  lgbm-all add IS-OOS gap (0.14 / 0.36) for less IC.
- **The trade plateau is drift, not alpha.** th=0.12 cells show net Sharpe
  0.56–0.57 — but a **shuffled-forecast null** (permute the forecast, same
  architecture, 40 draws) produces the *same* Sharpe (null mean 0.567 vs real
  0.564, p = 0.55). The long-only mom20 architecture harvests NQ's up-drift;
  the forecast adds zero timing. NQ buy-and-hold on the window: Sharpe 0.71.
- **Portfolio angle:** the NQ sleeve's daily returns correlate 0.59–0.69 with
  the production ES sleeve (underlying corr 0.87), ~0.00 with GC — enrolling
  it would double-count the existing equity exposure with a weaker forecast.
- Verdict: **not enrolled.** `INSTRUMENTS["NQ"]` stays as a research/data
  scaffold (no `DESIGN` entry, not in monitor/briefing defaults). Re-evaluate
  only with a genuinely NQ-specific hypothesis (e.g. NDX-specific flows) or
  after materially more OOS accrues.

## Study 4 (addendum, same day) — GC shorts from REVERSED long triggers

Study 2 rejected *forecast-threshold* shorts. This follow-up tested the other
channel: inverting the **price-action triggers** of the long design (rip-fade,
fresh-40d-LOW breakdown, trend-break), event-study first, sims only for
survivors. Independently verified by from-scratch recomputation (event stats,
sim engine, trade PnLs penny-exact).

Drift context first: GC 2005–2026 CAGR **+11.4%/yr** (B&H Sharpe 0.59,
unconditional 20d drift +0.86%) — a calendar-time short fights all of it.

- **Rip-fade: dead at the event stage.** Every overextension variant
  (ret5 ≥ 4%/6%, close > MA20 + 2/3×ATR) has conditionally **positive**
  forward returns — gold overextension *continues* (ret5 ≥ 6%: +1.05% fwd5).
- **Breakdown (mirror of the 40d-high breakout long): anti-predictive.**
  Fresh 40d lows precede **+1.63%** 20d returns (t +2.89 vs unconditional,
  hit-neg 30%) — GC breakdowns get *bought*. The mirror-symmetry hypothesis is
  falsified outright.
- **Trend-break: the only conditionally-negative variant (8% off the 100d
  peak) is insignificant** (t −0.99, CI straddles 0) and its negativity lives
  entirely **pre-2016** (post-2016 the same event precedes +1.5–2.1% rallies).
  Its 6 sim cells are ALL negative (median −0.155, best −0.12 with CI
  [−0.43, +0.23]); anchored selection collapses (pre-2016 best +0.25 →
  2016+ forward **−0.35**); and the combined book gets *worse* (Sharpe 0.54 →
  0.48) despite a −0.11 sleeve correlation — diversification cannot rescue
  negative expectancy.
- **The live intuition, measured**: after a +7%/5d rally (n=14 in 21 years),
  the next 5–10 days are a high-dispersion coin flip and the 20d horizon
  resolves **above** the unconditional drift (+1.46% vs +0.86%). The two most
  recent trend-break shorts of exactly this setup lost $1,800 (Oct-2025) and
  $3,514 (Feb-2026) per MGC.

**Verdict: both the model channel (Study 2) and the price-action channel
(this study) for GC daily shorts are measured dead. GC stays long-only by
measurement.** The user's short instinct — hold minutes, catch waterfalls —
is an *intraday* phenomenon and lives in `pine/gc_velocity_short.pine`
(velocity-breakdown tactic; validate on TradingView's multi-year 5m history),
not in this daily system.

## What this bought

The three "no"s are load-bearing: ES's 2-feature ridge is confirmed
**not under-specified** on fresh data; long-only is confirmed **by
measurement, not assumption**; and NQ is confirmed **redundant equity beta**
rather than a second edge. The next genuine expansion lead on file is the
macro-mini recent-half signal (Study 1 watchlist) and V2 data (Databento roll
handling, options flow).
