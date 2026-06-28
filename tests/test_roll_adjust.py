"""ES roll-jump cleaner — gap removal, tail invariance, and the no-look-ahead
guarantee (back-adjusting with future rolls only rescales past LEVELS by a
constant, so all RETURNS up to t are unchanged by later data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_swing import roll_adjust


def _synthetic(seed=0):
    """Cash that wanders, ES = cash with two injected roll gaps at 3rd Fridays."""
    idx = pd.date_range("2026-01-01", periods=400, freq="B")
    rng = np.random.default_rng(seed)
    cash = pd.Series(5000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    es = cash.copy()
    rolls = [pd.Timestamp(roll_adjust._third_friday(2026, 3)),
             pd.Timestamp(roll_adjust._third_friday(2026, 6))]
    for r in rolls:
        es.loc[es.index >= r] *= 1.01          # +1% spurious overnight gap at the roll
    ohlc = pd.DataFrame({"open": es, "high": es, "low": es, "close": es})
    return ohlc, cash, rolls


def test_detects_and_removes_gaps():
    ohlc, cash, rolls = _synthetic()
    cleaned, found = roll_adjust.backadjust(ohlc, cash, return_rolls=True)
    assert set(found) == set(rolls)                       # both rolls, nothing else
    cr = np.log(cleaned["close"] / cleaned["close"].shift(1))
    cashr = np.log(cash / cash.shift(1))
    for r in rolls:                                       # the +1% jump is gone -> cash return
        assert cr[r] == pytest.approx(cashr[r], abs=1e-9)


def test_tail_price_unchanged():
    ohlc, cash, _ = _synthetic()
    cleaned = roll_adjust.backadjust(ohlc, cash)
    # bars after the last roll are untouched (latest price preserved for display/signal)
    assert cleaned["close"].iloc[-1] == pytest.approx(ohlc["close"].iloc[-1], abs=1e-9)


def test_no_lookahead_returns():
    """Returns up to t must NOT change when later bars (and a later roll) are added."""
    ohlc, cash, rolls = _synthetic()
    full = roll_adjust.backadjust(ohlc, cash)
    full_ret = np.log(full["close"] / full["close"].shift(1))
    t = ohlc.index.get_loc(rolls[1]) - 5                  # cut before the second roll
    trunc = roll_adjust.backadjust(ohlc.iloc[:t + 1], cash.iloc[:t + 1])
    trunc_ret = np.log(trunc["close"] / trunc["close"].shift(1))
    pd.testing.assert_series_equal(full_ret.iloc[1:t + 1], trunc_ret.iloc[1:],
                                   check_names=False, rtol=1e-9, atol=1e-12)


def test_no_gaps_when_flat():
    ohlc, cash, _ = _synthetic()
    flat_cash = ohlc["close"]                             # cash == ES -> no divergence
    cleaned, found = roll_adjust.backadjust(ohlc, flat_cash, return_rolls=True)
    assert found == {}
    assert np.allclose(cleaned["close"].to_numpy(), ohlc["close"].to_numpy())
