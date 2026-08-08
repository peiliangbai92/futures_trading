# IS–OOS gap diagnosis (V1.4)

*What the walk-forward's in-sample vs out-of-sample IC gap actually means, and
what changed because of it. Referenced by `README.md` and by every generated
`reports/<SYM>/backtest.md`. Reproduce with:*

```bash
python -m futures_swing.diagnostics --symbol ES --experiment all
python -m futures_swing.diagnostics --symbol GC --experiment all --json reports/diagnostics/GC.json
```

## Question

The purged walk-forward CV reported a large IS-OOS Spearman-IC gap for GC
(~0.43) and, pre-V1.4, for ES. Is that **leakage** (a bug that would also
inflate OOS results) or **capacity overfit** (the model memorizing training
noise, with OOS still honest)?

## Method (`diagnostics.py`)

1. **Per-fold OOS IC t-test + sign test** — is the OOS edge consistent across
   folds or driven by a few lucky blocks?
2. **Circular block-bootstrap CI** of pooled OOS IC (block length = forecast
   horizon, respecting overlapping-label autocorrelation).
3. **Capacity sweep** — stump / shallow / deep LightGBM plus a Ridge baseline,
   watching IS IC, OOS IC, and the gap as capacity grows.
4. **Shuffled-label null** — refit on permuted targets to measure how much IS
   IC a capacity-rich model extracts from pure noise.
5. **Drop-top-k folds** — does the pooled OOS edge survive removing its best
   folds (edge concentration check)?

## Findings

- **The gap is benign capacity-overfit, not leakage.** As model capacity rises,
  IS IC → 1.0 and the gap → 0.96 while OOS IC stays flat or goes negative. The
  shuffled-label null shows **~55% of IS IC is pure noise-fitting**. Leakage
  was ruled out five ways (purge/embargo audit, causal-feature tests, the null,
  the capacity sweep shape, and fold consistency).
- **Consequence: stop reading IS IC or the gap as a quality signal.** Judge a
  model by pooled OOS IC and its block-bootstrap CI.
- **ES is a linear problem.** The 23-feature LightGBM diluted a real signal to
  noise (OOS IC +0.015, CI includes 0); a **ridge sleeve on `ret_5`/`ret_20`**
  surfaces a significant short-horizon mean-reversion edge — OOS IC **+0.073**,
  block-bootstrap 95% CI **[+0.03, +0.12]**, survives drop-top-5 and
  Bonferroni×40. Long-only (shorting ES's secular drift lost money). ES gap
  went +0.335 → −0.029.
- **GC keeps LightGBM** (a ridge kills its genuinely nonlinear edge). Its
  full-sample +0.07 OOS IC is significant, but the recent half (~+0.056) is
  not — the edge is **real but decaying**; size it with a haircut and watch a
  rolling OOS IC.

These conclusions are encoded in `INSTRUMENTS[sym]["alpha"]`
(`src/futures_swing/__init__.py`) and discussed further in the README's V1.4
section.

> Caveat: the ES reversion edge emerged from a config scan (it survives
> Bonferroni) and the long-only/threshold choices were picked on the full OOS
> backtest, so pre-register and collect forward OOS before treating it as a
> live edge — that is what `forward_validation.py` + `monitor.py` do.
