# Gold Fundamental / Positioning Data Sources — acquisition map (V2 shopping list)

Where to get gold-specific predictors beyond the financial-market macro block
(real yield / DXY / VIX / breakeven). Verified live (2026-06). **Difficulty:**
EASY = free + scriptable (FRED CSV, yfinance, CFTC Socrata) · MEDIUM = scrape /
registration · HARD = paid (Databento/Bloomberg).

> **Empirical note (this repo, tested):** of the EASY items below, COT, precious
> ratios (gold/silver, gold/copper), real-yield-curve slope, MOVE, survey
> inflation were all evaluated by pooled OOS IC + incremental model gates on GC's
> 10–40d forward return. **None robustly beat the existing 18-feature model's
> ~+0.070.** A few (gold/copper, real-yield-curve, gold/silver) show univariate
> |IC|~0.1–0.15 but only at long (40d) horizon, short history (2011+), mutually
> correlated, and add nothing incrementally. GC's edge is at the data ceiling for
> price+macro+these. The one genuinely untested tactical avenue is the **futures
> term-structure / backwardation backtest**, which needs HARD (paid) curve data.

## ⭐ Prioritized shortlist (predictive value × ease)

| Rank | Signal | Source / identifier | Difficulty |
|---|---|---|---|
| 1 | CFTC COT **Managed Money** net long (contrarian extreme) | Socrata `72hh-3qpy`, gold code `088691` | EASY |
| 2 | Bond vol **MOVE** | yfinance `^MOVE` | EASY |
| 3 | **Gold/copper** ratio | `GC=F` / `HG=F` (or `CPER`) | EASY |
| 4 | **Real-yield curve slope** (DFII30−DFII5) | FRED `DFII5,DFII7,DFII10,DFII20,DFII30` | EASY |
| 5 | Survey / forward inflation | FRED `MICH`, `T5YIFR` | EASY |
| 6 | **Futures term structure / basis** | yfinance `GC=F` + `GC{M}{YY}.CMX`; spot FRED `GOLDPMGBD228NLBM` | EASY (current) / HARD (backtest) |
| 7 | Gold ETF flows (aggregate) | WGC Goldhub ETF Excel | MEDIUM |
| 8 | Crisis: GPR / EPU | matteoiacoviello.com/gpr.htm; FRED `USEPUINDXD` | EASY/MED |

## Key sources (concrete identifiers)

**CFTC COT** — use **Disaggregated** report (gold is physical): resource `72hh-3qpy`,
`cftc_contract_market_code=088691`. Signal = `m_money_positions_long_all − m_money_positions_short_all`
(exclude `_spread`). Contrarian at **extremes** (tail/nonlinear, NOT a linear factor) — normalize
to net/OI or rolling percentile (0–100). Weekly, Tue snapshot, **Fri 15:30 ET release → lag ~3d**
(key off release date). Full history via `cftc.gov/files/dea/history/fut_disagg_txt_<YEAR>.zip`.

**Futures term structure** — back months `GC{F..Z}{YY}.CMX` (active: Feb/Apr/Jun/Aug/Dec).
Gold is almost always in mild contango (~cost-of-carry); the signal is the **deviation from
carry / flattening toward backwardation** (= physical stress = bullish), not the raw sign.
Yahoo back-month *history is gappy* → clean backtest needs **Databento `GLBX.MDP3` root GC** (HARD).
Avoid Nasdaq `CHRIS/CME_GC*` (frozen since 2018).

**Lease rates** — GOFO discontinued 2015 (no public replacement). DIY proxy: `SOFR (FRED) −
GC-implied-forward (from calendar spread)`. Stress/tail flag only.

**ETF holdings** — SPDR GLD CSV (`spdrgoldshares.com/.../GLD_US_archive_EN.csv`) now serves a
**decoy PDF to scripts** → needs headless browser. Better: WGC Goldhub aggregate ETF tonnes/flows
(registration). Endogenous with price (coincident, not leading).

**Inventory** — CME `Gold_Stocks.xls` **403s scrapers** (manual/licensed only). LBMA vault data
monthly .xlsx (EASY-MED). Weak/noisy — regime flag, not alpha.

**Central-bank reserves** — IMF **IRFCL** via `api.imf.org/external/sdmx/2.1/` (keyless; old
`dataservices.imf.org` is DEAD). WGC reserves-by-country (registration). Monthly ~2mo lag.
Structural regime overlay, not tactical.

**Other EASY adds:** `^MOVE` (rates vol → gold, leads stress); gold-in-non-USD (`GC=F × USDxxx=X`,
new-highs-in-EUR-first is a leading tell); FRED `DTWEXBGS` (broad USD; old `DTWEXB` discontinued);
gold seasonality (winter Dec–Feb strong, Sep weak — monthly dummies from GC itself).

**Bottom line:** the cheap wins are COT (Managed Money, extreme/nonlinear), `^MOVE`, gold/copper,
real-yield-curve slope, `MICH`/`T5YIFR`, GC basis — but per this repo's tests none beat +0.070
on GC's 10–40d horizon. Physical/official data is regime context, not weeks-to-months alpha. The
real remaining tactical lead (term-structure backtest) requires paid curve data (Databento).
