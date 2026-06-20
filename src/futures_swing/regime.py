"""Rule-based market regime (V1).

A single market-wide risk regime derived from the S&P (ES) trend vs its MA100
and the VIX level / spike, per the plan (doc Section 7.2). Used both as a model
feature and as a gate (e.g. block shorts in strong risk-on, cut size in stress).

Point-in-time: every value at date t uses only data <= t (MA, VIX, VIX change).
The interface is intentionally small so a GMM / HMM / Markov-switching model can
replace ``classify`` in V2 without touching callers.

    risk_on   VIX < vix_low  and ES > MA100
    risk_off  VIX > vix_high and ES < MA100
    range     otherwise (VIX mid / no clear trend)
    stress    VIX spikes >= stress_jump over stress_lookback days (overrides)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS = ("risk_on", "range", "risk_off", "stress")
REGIME_CODES = {label: i for i, label in enumerate(REGIME_LABELS)}


def classify(
    es_close: pd.Series,
    vix_close: pd.Series,
    *,
    ma_window: int = 100,
    vix_low: float = 18.0,
    vix_high: float = 25.0,
    stress_jump: float = 0.20,
    stress_lookback: int = 5,
) -> pd.Series:
    """Return a regime label per date on ``es_close``'s index (object dtype).

    Warmup dates (before MA100 is defined) are ``pd.NA``.
    """
    es_close = es_close.astype(float).sort_index()
    vix = vix_close.astype(float).reindex(es_close.index, method="ffill")

    ma = es_close.rolling(ma_window, min_periods=ma_window).mean()
    above = es_close > ma
    vix_jump = vix / vix.shift(stress_lookback) - 1.0

    labels = pd.Series("range", index=es_close.index, dtype=object)
    labels[(vix < vix_low) & above] = "risk_on"
    labels[(vix > vix_high) & (~above)] = "risk_off"
    labels[vix_jump >= stress_jump] = "stress"   # spike overrides
    labels[ma.isna()] = pd.NA
    return labels.rename("regime")


def classify_default(**kwargs) -> pd.Series:
    """Convenience: load ES + VIX from the cache and classify."""
    from . import data_loader

    es = data_loader.load_close("ES")
    vix = data_loader.load_close("VIX")
    return classify(es, vix, **kwargs)


def code_series(labels: pd.Series) -> pd.Series:
    """Map regime labels to ordinal codes (NA-safe) for use as a model feature."""
    return labels.map(REGIME_CODES).astype("float").rename("regime_code")


# ---------------------------------------------------------------------------
# HMM regime (V1.2) — causal forward-filtered latent states
# ---------------------------------------------------------------------------
#
# Lookahead is the whole game here. A standard HMM's Viterbi / smoothed posterior
# (hmmlearn predict / predict_proba) uses the FULL observation sequence, i.e.
# future data — unusable as a feature. We instead:
#   1. fit HMM params ONCE on an initial window strictly before the OOS region
#      (params at any OOS test date depend only on earlier data);
#   2. standardize observations with that window's mean/std (no full-sample leak);
#   3. run a manual **forward filter** so the posterior at t uses only obs <= t.
# States are relabeled by ascending vol so column semantics are stable
# (state 0 = calmest ... K-1 = most volatile).

# Fit params strictly before the OOS region. The first walk-forward OOS fold
# begins ~2005-06 (feature row MIN_TRAIN=1000), so a 2005-01-01 cutoff keeps HMM
# parameter estimation causal for every OOS test date — for both the market HMM
# (ES obs from 2000) and the gold HMM (obs from 2003, real-yield constrained).
HMM_FIT_CUTOFF = "2005-01-01"
HMM_VOL_WINDOW = 10


def _gaussian_loglik(Z: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-sample, per-state diagonal-Gaussian log-likelihood -> (n, K)."""
    diff = Z[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(np.log(2 * np.pi * var)[None, :, :] + diff**2 / var[None, :, :], axis=2)


def _forward_filter(logB: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Causal filtered posteriors P(state_t | obs<=t). Stable, normalized per step."""
    n, k = logB.shape
    filt = np.zeros((n, k))
    l0 = logB[0] + np.log(startprob + 1e-12)
    l0 -= l0.max()
    p0 = np.exp(l0); filt[0] = p0 / p0.sum()
    for t in range(1, n):
        prior = filt[t - 1] @ transmat              # predict
        lb = logB[t] - logB[t].max()
        a = prior * np.exp(lb)                       # update with emission
        s = a.sum()
        filt[t] = a / s if s > 0 else prior
    return filt


def _fit_filter(obs: pd.DataFrame, *, vol_col: str, n_states: int, fit_cutoff: str, seed: int) -> pd.DataFrame:
    """Fit a Gaussian HMM on observations before ``fit_cutoff`` and return causal
    forward-filtered posteriors over the whole series. States are relabeled by
    ascending mean of ``vol_col`` (0 = calmest) for stable column semantics."""
    import warnings

    from hmmlearn.hmm import GaussianHMM

    obs = obs.replace([np.inf, -np.inf], np.nan).dropna()
    train = obs[obs.index < pd.Timestamp(fit_cutoff)]
    if len(train) < 100:                          # fallback if too little pre-cutoff data
        train = obs.iloc[: max(100, len(obs) // 2)]
    mu, sd = train.mean(), train.std(ddof=0).replace(0, 1.0)
    Z = (obs - mu) / sd
    Ztr = Z.loc[train.index].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
        hmm.fit(Ztr)

    vci = obs.columns.get_loc(vol_col)
    order = np.argsort(hmm.means_[:, vci])
    means = hmm.means_[order]
    var = np.stack([np.diag(hmm.covars_[k]) for k in range(n_states)])[order]
    startprob = hmm.startprob_[order]
    transmat = hmm.transmat_[order][:, order]

    logB = _gaussian_loglik(Z.to_numpy(), means, var)
    filt = _forward_filter(logB, startprob, transmat)
    out = pd.DataFrame({f"hmm_p{i}": filt[:, i] for i in range(n_states)}, index=obs.index)
    out["hmm_state"] = filt.argmax(axis=1).astype(float)
    return out


def _market_observations(es_close: pd.Series, vix_close: pd.Series, *, vol_window: int) -> pd.DataFrame:
    es = es_close.astype(float).sort_index()
    vix = vix_close.astype(float).reindex(es.index, method="ffill")
    ret = np.log(es).diff()
    return pd.DataFrame({"ret": ret, "rvol": ret.rolling(vol_window).std(), "lvix": np.log(vix)})


def _gold_observations(gc_close: pd.Series, real_yield: pd.Series, dxy: pd.Series, *, vol_window: int) -> pd.DataFrame:
    gc = gc_close.astype(float).sort_index()
    ret = np.log(gc).diff()
    ry = real_yield.astype(float).reindex(gc.index, method="ffill").diff(5)       # real-yield 5d change
    dx = dxy.astype(float).reindex(gc.index, method="ffill").where(lambda x: x > 0)
    return pd.DataFrame({"ret": ret, "rvol": ret.rolling(vol_window).std(),
                         "ry_chg5": ry, "dxy_ret5": np.log(dx).diff(5)})


def hmm_features(es_close, vix_close, *, n_states: int = 3, fit_cutoff: str = HMM_FIT_CUTOFF,
                 vol_window: int = HMM_VOL_WINDOW, seed: int = 42) -> pd.DataFrame:
    """Causal market HMM regime (ES return + short vol + log-VIX)."""
    obs = _market_observations(es_close, vix_close, vol_window=vol_window)
    return _fit_filter(obs, vol_col="rvol", n_states=n_states, fit_cutoff=fit_cutoff, seed=seed)


def hmm_features_for(symbol: str, *, obs_source: str = "market", n_states: int = 3,
                     fit_cutoff: str = HMM_FIT_CUTOFF, vol_window: int = HMM_VOL_WINDOW,
                     seed: int = 42) -> pd.DataFrame:
    """HMM regime for a symbol. Default uses the **market** observations (ES+VIX);
    ``obs_source='gold'`` uses gold-specific obs (GC ret/vol + real-yield change +
    DXY change) — tested in V1.2 and found *not* to help GC, kept for research."""
    from . import data_loader

    use_gold = obs_source == "gold"
    if use_gold:
        obs = _gold_observations(data_loader.load_close("GC"), data_loader.load_close("REAL_YIELD"),
                                 data_loader.load_close("DXY"), vol_window=vol_window)
    else:
        obs = _market_observations(data_loader.load_close("ES"), data_loader.load_close("VIX"),
                                   vol_window=vol_window)
    return _fit_filter(obs, vol_col="rvol", n_states=n_states, fit_cutoff=fit_cutoff, seed=seed)


def hmm_features_default(**kwargs) -> pd.DataFrame:
    """Convenience: load ES + VIX from the cache and compute the market HMM."""
    from . import data_loader

    return hmm_features(data_loader.load_close("ES"), data_loader.load_close("VIX"), **kwargs)
