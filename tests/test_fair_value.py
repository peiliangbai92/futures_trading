"""Fair-value module — calendar, cost-of-carry formula, and graceful degradation.

Formula tests are cache-independent (they monkeypatch the input readers), so they
run anywhere; one integration check skips cleanly if the input feeds aren't built.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from futures_swing import data_loader, fair_value as fv


# --------------------------------------------------------------- expiry calendar


def test_third_friday_and_quarterly():
    assert fv._third_friday(2026, 6) == date(2026, 6, 19)     # 3rd Fri Jun-2026
    assert fv._third_friday(2026, 9) == date(2026, 9, 18)
    nq = fv.next_quarterly_expiry(date(2026, 6, 25))          # Jun already passed
    assert nq == date(2026, 9, 18) and nq.month in (3, 6, 9, 12)


def test_gc_curve_contracts_future_even_ascending():
    cs = fv._gc_curve_contracts(date(2026, 1, 15), 3)
    assert len(cs) == 3
    exps = [e for e, _sym, _key in cs]
    assert exps == sorted(exps)                               # ascending
    assert all(e > date(2026, 1, 15) for e in exps)           # strictly future
    assert all(e.month in (2, 4, 6, 8, 10, 12) for e in exps) # active even months
    assert cs[0][1].startswith("GC") and cs[0][1].endswith(".CMX")


# --------------------------------------------------------------- cost-of-carry


def test_index_fair_value_formula(monkeypatch):
    ref = date(2026, 1, 2)
    monkeypatch.setattr(fv, "_ref_date", lambda s, a, fp: (5050.0, ref))
    monkeypatch.setattr(fv, "_asof_value",
                        lambda key, asof: {"TBILL_3M": (4.0, ref), "SPX_CASH": (5000.0, ref)}.get(key, (None, None)))
    s = fv.summary("ES", asof="2026-01-02")
    T = (fv.next_quarterly_expiry(ref) - ref).days / 365.0
    expect_fv = 5000.0 * math.exp((0.04 - 0.013) * T)         # cash * exp((r-q)T)
    assert s["kind"] == "index"
    assert s["fair_value"] == pytest.approx(expect_fv, rel=1e-9)
    assert s["basis"] == pytest.approx(5050.0 - expect_fv, rel=1e-9)
    assert "rich" in fv.line("ES", asof="2026-01-02")         # future > FV


def test_commodity_implied_carry(monkeypatch):
    ref = date(2026, 1, 2)
    monkeypatch.setattr(fv, "_ref_date", lambda s, a, fp: (4050.0, ref))
    cs = fv._gc_curve_contracts(ref, 3)
    (e1, _s1, k1), (e3, _s3, k3) = cs[0], cs[-1]
    T1, T3 = (e1 - ref).days / 365.0, (e3 - ref).days / 365.0
    p1 = 4050.0
    p3 = p1 * math.exp(0.05 * (T3 - T1))                      # construct a 5%/yr carry
    monkeypatch.setattr(fv, "_asof_value",
                        lambda key, asof: {"TBILL_3M": (4.0, ref), k1: (p1, ref), k3: (p3, ref)}.get(key, (None, None)))
    s = fv.summary("GC", asof="2026-01-02")
    assert s["kind"] == "commodity"
    assert s["impl_carry"] == pytest.approx(0.05, abs=1e-9)
    assert s["theo_carry"] == pytest.approx(0.04 + 0.004 - 0.0, abs=1e-12)
    assert "carry" in fv.line("GC", asof="2026-01-02")


def test_degrades_when_inputs_missing(monkeypatch):
    monkeypatch.setattr(fv, "_ref_date", lambda s, a, fp: (5000.0, date(2026, 1, 2)))
    monkeypatch.setattr(fv, "_asof_value", lambda key, asof: (None, None))
    assert fv.summary("ES") is None
    assert fv.line("ES") is None
    assert fv.line("GC") is None


# --------------------------------------------------------------- integration


@pytest.mark.skipif(not (data_loader.RAW_DIR / "SPX_CASH.parquet").exists(),
                    reason="fair-value inputs not built (run fair_value --refresh)")
def test_es_summary_real_inputs():
    s = fv.summary("ES")
    assert s is not None and s["fair_value"] > 0 and s["dte"] > 0
