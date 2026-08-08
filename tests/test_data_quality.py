"""Data-quality guarantees: total-return series for dividend-paying ETF feeds,
and the per-symbol roll-adjustment policy."""
from __future__ import annotations

import numpy as np
import pytest

from futures_swing import INSTRUMENTS, data_loader, features

_HAVE_TIP = (data_loader.RAW_DIR / "TIP.parquet").exists()


def test_total_return_keys_are_the_etf_feeds():
    assert features.TOTAL_RETURN_KEYS == {"TIP", "HYG", "LQD"}


@pytest.mark.skipif(not _HAVE_TIP, reason="data/raw not built")
def test_load_total_return_close_uses_adj_close():
    raw = data_loader.load_close("TIP")
    tr = data_loader.load_total_return_close("TIP")
    # adjusted series must differ from raw (TIP pays monthly distributions) and
    # match on the most recent bar (adjustment is anchored at the present)
    assert not np.allclose(raw.dropna().tail(500), tr.dropna().tail(500))
    common = raw.index.intersection(tr.index)[-1]
    assert abs(tr.loc[common] / raw.loc[common] - 1) < 0.005


@pytest.mark.skipif(not (data_loader.RAW_DIR / "UST2Y.parquet").exists(),
                    reason="data/raw not built")
def test_load_total_return_close_falls_back_for_fred():
    # FRED frames have no adj_close column -> identical to load_close
    raw = data_loader.load_close("UST2Y")
    tr = data_loader.load_total_return_close("UST2Y")
    assert raw.equals(tr)


def test_roll_adjust_policy_is_per_symbol():
    # ES is cash-anchored back-adjusted; GC deliberately is NOT (no cash anchor
    # for gold in V1 — revisit with Databento in V2). A test documents the choice.
    assert INSTRUMENTS["ES"].get("roll_adjust") is True
    assert INSTRUMENTS["ES"].get("roll_adjust_cash") == "SPX_CASH"
    assert not INSTRUMENTS["GC"].get("roll_adjust")
