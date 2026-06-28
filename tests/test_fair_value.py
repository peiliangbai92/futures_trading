"""Fair-value module — calendar, index cost-of-carry formula, graceful degrade.

Formula tests are cache-independent (they monkeypatch the input readers); one
integration check skips cleanly if the input feeds aren't built.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from futures_swing import data_loader, fair_value as fv


def test_third_friday_and_quarterly():
    assert fv._third_friday(2026, 6) == date(2026, 6, 19)     # 3rd Fri Jun-2026
    assert fv._third_friday(2026, 9) == date(2026, 9, 18)
    nq = fv.next_quarterly_expiry(date(2026, 6, 25))          # Jun already passed
    assert nq == date(2026, 9, 18) and nq.month in (3, 6, 9, 12)


def test_index_fair_value_formula(monkeypatch):
    ref = date(2026, 1, 2)
    monkeypatch.setattr(fv, "_ref_date", lambda s, a, fp: (5050.0, ref))
    monkeypatch.setattr(fv, "_asof_value",
                        lambda key, asof: {"SOFR": (4.0, ref), "SPX_CASH": (5000.0, ref)}.get(key, (None, None)))
    s = fv.summary("ES", asof="2026-01-02")
    T = (fv.next_quarterly_expiry(ref) - ref).days / fv.DAY_COUNT
    expect_fv = 5000.0 * math.exp((0.04 - 0.013) * T)         # cash * exp((r-q)T), ACT/360
    assert s["kind"] == "index"
    assert s["fair_value"] == pytest.approx(expect_fv, rel=1e-9)
    assert s["fair_basis"] == pytest.approx(expect_fv - 5000.0, rel=1e-9)
    assert s["basis"] == pytest.approx(5050.0 - expect_fv, rel=1e-9)
    assert s["implied_cash_open"] == pytest.approx(5050.0 - (expect_fv - 5000.0), rel=1e-9)
    assert "rich" in fv.line("ES", asof="2026-01-02")         # future > FV


def test_degrades_when_inputs_missing(monkeypatch):
    monkeypatch.setattr(fv, "_ref_date", lambda s, a, fp: (5000.0, date(2026, 1, 2)))
    monkeypatch.setattr(fv, "_asof_value", lambda key, asof: (None, None))
    assert fv.summary("ES") is None
    assert fv.line("ES") is None


@pytest.mark.skipif(not (data_loader.RAW_DIR / "SPX_CASH.parquet").exists(),
                    reason="fair-value inputs not built (run fair_value --refresh)")
def test_es_summary_real_inputs():
    s = fv.summary("ES")
    assert s is not None and s["fair_value"] > 0 and s["dte"] > 0
