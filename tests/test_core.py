"""Core correctness tests — the no-lookahead / alignment guarantees matter most
for a forward-return model. Tests that need the data cache skip cleanly if it is
absent (run `python -m futures_swing.data_loader` first)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from futures_swing import INSTRUMENTS, features, model, signal
from futures_swing import vol as volmod
from futures_swing.execution import hit_exit, levels
from futures_swing.risk import position_size

try:
    from futures_swing import data_loader

    _HAVE_DATA = (data_loader.RAW_DIR / "ES.parquet").exists()
except Exception:  # pragma: no cover
    _HAVE_DATA = False

needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="raw data cache not built")


def _synthetic_ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    openp = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close}, index=idx)


# --------------------------------------------------------------- vol / atr


def test_yang_zhang_and_atr_finite_positive():
    df = _synthetic_ohlc()
    yz = volmod.yang_zhang_volatility(df, window=21).dropna()
    atr = volmod.atr(df, window=14).dropna()
    assert (yz > 0).all() and np.isfinite(yz).all()
    assert (atr > 0).all() and np.isfinite(atr).all()


def test_atr_matches_manual_true_range():
    df = _synthetic_ohlc(n=40)
    manual = pd.concat(
        [df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()],
        axis=1,
    ).max(axis=1).rolling(14).mean()
    pd.testing.assert_series_equal(volmod.atr(df, window=14), manual.rename("atr"))


# --------------------------------------------------------------- target / causality


def test_forward_log_return_alignment():
    close = pd.Series([10, 11, 12, 13, 14.0], index=pd.date_range("2020-01-01", periods=5))
    fwd = features.forward_log_return(close, 2)
    assert fwd.iloc[0] == pytest.approx(math.log(12 / 10))   # t -> t+2
    assert math.isnan(fwd.iloc[-1]) and math.isnan(fwd.iloc[-2])  # no future


def test_trend_block_is_causal():
    df = _synthetic_ohlc(n=60)
    block = features._trend_block(df["close"])
    i = 40
    assert block["ret_5"].iloc[i] == pytest.approx(math.log(df["close"].iloc[i] / df["close"].iloc[i - 5]))
    ma20 = df["close"].iloc[i - 19 : i + 1].mean()
    assert block["ma_dist_20"].iloc[i] == pytest.approx(df["close"].iloc[i] / ma20 - 1)


# --------------------------------------------------------------- purged CV


def test_purged_folds_enforce_gap():
    horizon, embargo = 10, 10
    folds = model.purged_walk_forward_folds(4000, horizon=horizon, embargo=embargo, min_train=1000, test_size=252)
    assert folds
    for tr, te in folds:
        assert te.start - tr.stop == horizon + embargo   # purge+embargo gap
        assert tr.start == 0 and tr.stop > 0
        assert te.stop <= 4000


# --------------------------------------------------------------- execution / risk


def test_levels_and_exit_logic():
    lv = levels(1, 100.0, 5.0)
    assert lv["stop"] == 90.0 and lv["target"] == 115.0
    assert hit_exit(1, 116, 95, 90, 115) == ("target", 115)
    assert hit_exit(1, 112, 89, 90, 115) == ("stop", 90)       # stop assumed first
    assert hit_exit(1, 112, 95, 90, 115) == (None, None)


def test_position_size_respects_risk_budget():
    # risk per contract = 2*ATR*pv = 2*10*5 = $100; budget = 50k*0.0075 = $375
    qty = position_size(50_000, 1.0, atr=10.0, point_value=5.0)
    assert qty == 3  # floor(375/100 * conviction(=1))
    assert position_size(50_000, float("nan"), 10.0, 5.0) == 0
    assert position_size(50_000, 1.0, 0.0, 5.0) == 0


# --------------------------------------------------------------- data-backed


@needs_data
def test_make_dataset_has_no_target_nan():
    for sym in INSTRUMENTS:
        X, y, horizon = model.make_dataset(sym)
        assert y.isna().sum() == 0
        assert len(X) == len(y)
        assert horizon == INSTRUMENTS[sym]["horizon"]


@needs_data
def test_hmm_regime_is_causal():
    """Filtered HMM posterior at date t must not depend on data after t."""
    from futures_swing import data_loader, regime

    es, vix = data_loader.load_close("ES"), data_loader.load_close("VIX")
    full = regime.hmm_features(es, vix)
    t0 = full.index[4000]
    trunc = regime.hmm_features(es[es.index <= t0], vix[vix.index <= t0])
    cols = [c for c in full.columns if c.startswith("hmm_p")]
    assert np.allclose(full.loc[t0, cols].to_numpy(), trunc.loc[t0, cols].to_numpy(), atol=1e-9)


@needs_data
def test_compute_signals_columns_and_thresholds():
    X, y, _ = model.make_dataset("ES")
    pred = y.copy()  # use realized as a stand-in series just to exercise the path
    sig = signal.compute_signals("ES", pred)
    assert set(sig.columns) == {"pred_ret", "fc_vol", "sharpe", "signal"}
    assert set(sig["signal"].dropna().unique()) <= {-1, 0, 1}
