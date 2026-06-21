# GC (Gold Futures) Alpha Model — Complete Design Inventory

Precise, line-cited inventory of the **GC** model in `src/futures_swing/`. Feature
count is **18** (matches `reports/diagnostics/GC.json` `n_features: 18`). Use this
as the reference when redesigning GC. (Generated from a code audit, 2026-06.)

## 1. GC instrument config — `INSTRUMENTS["GC"]` (`__init__.py:39`)

| Field | Value |
|---|---|
| `yf_symbol` | `GC=F` (Yahoo continuous gold front-month) |
| `micro_symbol` / `point_value` / `tick` | `MGC` / **$10/pt** / `0.10` |
| `horizon` | **10** trading days (forward-return + holding horizon) |
| `regime` | **`"rule"`** (HMM / gold-HMM did NOT help GC) |
| `alpha` | **`{"kind":"lgbm","features":"all"}`** (ridge kills GC's nonlinear edge) |
| `signal_th` | **0.35** (symmetric long/short threshold) |
| `long_only` | *unset* → **GC trades both sides** (unlike ES) |
| `signal_smooth` | *unset* → no EMA smoothing |

GC is modeled independently from ES; they share only the market-wide regime + the VIX feature.

## 2. Complete feature list — 18 columns

Order: trend (8) → vol (3) → macro (6) → regime (1). `include_fred=False` by default
(FRED block excluded — V1.1 found it did not help and the strong yield *levels* are
non-stationary).

### Block A — TREND (8) · `_trend_block` (`features.py:35`), source = GC close
| # | Feature | Formula | Meaning |
|---|---|---|---|
| 1–4 | `ret_5,20,60,120` | `log(close_t / close_{t-n})` | 1w / 1m / 1q / 6m momentum |
| 5–8 | `ma_dist_20,50,100,200` | `close_t / SMA_n(close)_t − 1` | % stretch above/below the n-day mean (trend position). `ma_dist_200` also gates the matrix start |

### Block B — VOLATILITY (3) · `_vol_block` (`features.py:45`), source = GC OHLC
| # | Feature | Formula | Meaning |
|---|---|---|---|
| 9 | `yz_vol` | Yang-Zhang (2000) annualized vol, 21d (overnight + open-close + Rogers-Satchell) | range-based realized vol incl. overnight gaps |
| 10 | `vol_chg20` | `yz_vol_t / yz_vol_{t-20} − 1` | vol expansion/contraction |
| 11 | `atr_pct` | `ATR14 / close_t` (TR = max(H−L,|H−prevC|,|L−prevC|)) | normalized daily swing |

### Block C — MACRO / CROSS-ASSET (6) · `_macro_block` GC branch (`features.py:84`)
All reindexed onto GC's calendar with **forward-fill**. `lvl`=level, `ret`=log n-ret, `chg`=n-day diff.
| # | Feature | Transform | Symbol | Meaning |
|---|---|---|---|---|
| 12 | `ust10y_level` | `lvl` | `^TNX` | 10Y nominal yield level — gold cost-of-carry |
| 13 | `ust10y_chg20` | `chg(20)` | `^TNX` | 20d move in 10Y yield (headwind when rising) |
| 14 | `tip_ret20` | `ret(20)` | `TIP` ETF | **real-yield proxy** (the dominant gold driver; used since FRED is off) |
| 15 | `dxy_ret20` | `ret(20)` | `DX-Y.NYB` | USD index 20d return (stronger USD = headwind) |
| 16 | `oil_ret20` | `ret(20)` | `CL=F` | crude 20d return (commodity/inflation co-move) |
| 17 | `vix_level` | `lvl` | `^VIX` | equity fear gauge — safe-haven demand |

> GC macro set is gold-centric (rates, real-yield proxy, USD, oil, VIX). It does NOT use vvix/vix_chg5 (those are ES-only).

### Block D — REGIME (1) · rule regime (`features.py:141`, `regime.py:25`)
| # | Feature | Definition |
|---|---|---|
| 18 | `regime_code` | Ordinal 0–3 of the **market-wide** regime from **ES vs MA100** + **VIX**: `risk_on`(VIX<18 & ES>MA100)=0, `range`=1, `risk_off`(VIX>25 & ES<MA100)=2, `stress`(VIX +20% over 5d, overrides)=3. Forward-filled onto GC's calendar |

> Caveat: the 0–3 ordering is not monotone in risk (stress=3 is an override). LightGBM treats it as a threshold-splittable numeric.

### Excluded by default (behind `include_fred=True`) — 6 FRED cols
`real_yield_level`(DFII10), `real_yield_chg20`, `breakeven_level`(T10YIE), `breakeven_chg20`,
`ust2y_level`(DGS2), `ust2y_chg20`. OFF because they did not improve OOS IC and the yield
*levels* are non-stationary (inflated IS IC).

## 3. The model

- **Estimator**: LightGBM `LGBMRegressor` on all 18 features. `DEFAULT_PARAMS` (`model.py:34`): `objective=huber, n_estimators=300, lr=0.02, num_leaves=15, max_depth=3, min_child_samples=100, subsample=0.8, colsample=0.8, reg_alpha=1, reg_lambda=5`. NaNs handled natively.
- **Target**: `log(close[t+10] / close[t])` (10d forward log return), kept out of the feature matrix (no leak).
- **CV**: purged + embargoed expanding walk-forward — `MIN_TRAIN=1000`, `TEST_SIZE=252`, train-test gap = `horizon+embargo = 20d`. Effective N ≈ n_days/10 ≈ **627** (21 folds).
- **Forecast**: per fold, fresh LGBM → OOS `pred`. `fit_full`/`predict_latest` for live.

## 4. Forecast → signal → trade (`signal.py`)
1. `sharpe = pred_ret / fc_vol`, where `fc_vol` = trailing 21d annualized close-to-close vol × √(10/252).
2. No smoothing for GC.
3. Threshold at **±0.35**: `+1` long if sharpe ≥ +0.35, `−1` short if ≤ −0.35, else flat. **GC trades both sides.**
4. Sizing: V1.3 inverse-vol notional target (10% vol, conviction-scaled) on MGC; strategy Sharpe ~0.57, maxDD ~−9% at $220k.

## 5. Known GC findings (validated, from GC.json / README)
- **Real OOS edge**: pooled OOS IC **+0.070**, block-boot 95% CI **[+0.009, +0.131]** (excludes 0), p=0.012; per-fold mean +0.104, t=3.23, 17/21 folds positive; shuffled-null clean (p≈0).
- **Edge is price/trend, not regime**: HMM (market +0.072 / gold +0.063) did not beat rule (+0.070).
- **FRED did not help** (ΔOOS −0.003, gap widened); the strong `real_yield_level` is a non-stationary level.
- **Ridge kills it** (OOS +0.007 vs lgbm +0.070) → genuine nonlinearity, keep LightGBM.
- **Gap is benign capacity-overfit** (IS 0.529 vs OOS 0.104; ~52% of IS IC is noise-fitting), not leakage.
- **Decaying**: full-sample +0.07 significant, recent half ~+0.056 not. Forward-validation (V1.6) confirms: anchored forward Sharpe mean only ~+0.25–0.32 (vs IS ~0.6), with a negative window — GC is fragile / regime-dependent, and the lgbm forecast adds little (~0.12 of the strategy Sharpe; the rest is trend-structure in the gold bull).
