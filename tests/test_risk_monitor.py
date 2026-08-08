"""Tests for the previously-untested live-money path pieces: the event
stand-aside filter (parser + window + freshness warning), vol-target sizing,
RiskManager drawdown gates, and the monitor's live-log append contract."""
from __future__ import annotations

import warnings
from datetime import date, timedelta

import pandas as pd
import pytest

from futures_swing import execution, monitor, risk


# ------------------------------------------------------------- event filter


def _write_calendar(path, dates):
    lines = ["# comment line", "date,event,category,vol_mult"]
    lines += [f"{d},Test event,TEST,1.0" for d in dates]
    path.write_text("\n".join(lines))


def test_load_event_dates_parses_dates_and_skips_comments(tmp_path):
    p = tmp_path / "cal.csv"
    future = date.today() + timedelta(days=90)
    _write_calendar(p, [future.isoformat(), "not-a-date"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no freshness warning expected
        out = risk.load_event_dates(p)
    assert out == {future}


def test_load_event_dates_missing_file_is_empty(tmp_path):
    assert risk.load_event_dates(tmp_path / "absent.csv") == set()


def test_load_event_dates_warns_when_calendar_near_expiry(tmp_path):
    p = tmp_path / "cal.csv"
    _write_calendar(p, [(date.today() + timedelta(days=5)).isoformat()])
    with pytest.warns(UserWarning, match="event calendar ends"):
        risk.load_event_dates(p)


def test_real_calendar_parses():
    out = risk.load_event_dates()          # the committed configs file
    assert out, "configs/event_calendar.csv should yield dates"
    assert all(isinstance(d, date) for d in out)


def test_near_event_window():
    ev = {date(2026, 9, 16)}
    assert risk.near_event(date(2026, 9, 15), ev)
    assert risk.near_event(date(2026, 9, 16), ev)
    assert risk.near_event(date(2026, 9, 17), ev)
    assert not risk.near_event(date(2026, 9, 18), ev)
    assert not risk.near_event(date(2026, 9, 18), set())
    assert risk.near_event(pd.Timestamp("2026-09-14"), ev, window=2)


# ------------------------------------------------------------------- sizing


def test_vol_target_size_notional_math():
    # notional = equity * target_vol / ann_vol = 220k * 0.10 / 0.20 = 110k;
    # contract notional = price * pv = 4000 * 5 = 20k -> floor(5.5) = 5
    assert risk.vol_target_size(220_000, 0.20, 4000, 5.0, target_vol=0.10) == 5


def test_vol_target_size_conviction_and_cap():
    full = risk.vol_target_size(220_000, 0.20, 4000, 5.0, target_vol=0.10)
    half = risk.vol_target_size(220_000, 0.20, 4000, 5.0, target_vol=0.10, conviction_mult=0.5)
    assert half <= full // 2 + 1 and half < full
    assert risk.vol_target_size(1e9, 0.10, 100, 5.0) == risk.MAX_CONTRACTS


def test_vol_target_size_guards():
    assert risk.vol_target_size(220_000, 0.0, 4000, 5.0) == 0
    assert risk.vol_target_size(220_000, 0.2, 0.0, 5.0) == 0
    assert risk.vol_target_size(0.0, 0.2, 4000, 5.0) == 0


def test_position_size_stop_matches_execution():
    # sizing must assume the same stop distance execution actually places
    import inspect
    default = inspect.signature(risk.position_size).parameters["stop_mult"].default
    assert default == execution.ATR_STOP_MULT


# --------------------------------------------------------------- RiskManager


def test_risk_manager_drawdown_gates():
    rm = risk.RiskManager(equity0=100_000)
    assert rm.size_multiplier(100_000) == 1.0
    assert rm.size_multiplier(94_000) == 0.5          # >5% DD -> halve
    assert rm.size_multiplier(89_000) == 0.0          # >10% DD -> halt
    rm.update(120_000)                                 # new peak
    assert rm.size_multiplier(112_000) == 0.5          # DD measured off the peak


# ------------------------------------------------------------ live-log append


def _rows(asof, action="HOLD (1)", raw="HOLD (1)"):
    return [dict(symbol="ES", asof=asof, your_position=1, your_action=action,
                 model_position=1, model_action=action, sharpe=0.01,
                 exit_rule="mom20", raw_action=raw)]


def test_append_live_log_dedupes_and_keeps_last(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "LIVE_LOG", tmp_path / "live_log.csv")
    monitor.append_live_log(_rows("2026-08-07", action="BUY dip"))
    monitor.append_live_log(_rows("2026-08-07", action="HOLD (1)"))   # same key -> replace
    monitor.append_live_log(_rows("2026-08-08"))
    df = pd.read_csv(monitor.LIVE_LOG)
    assert len(df) == 2
    assert df.loc[df["asof"] == "2026-08-07", "your_action"].item() == "HOLD (1)"


def test_append_live_log_tolerates_legacy_schema(tmp_path, monkeypatch):
    """Old CSVs predate the raw_action column; appending must not lose rows."""
    log = tmp_path / "live_log.csv"
    monkeypatch.setattr(monitor, "LIVE_LOG", log)
    legacy = pd.DataFrame([dict(symbol="GC", asof="2026-08-06", your_position=1,
                                your_action="HOLD (1)", model_position=1,
                                model_action="HOLD (1)", sharpe=0.1, exit_rule="trail")])
    legacy.to_csv(log, index=False)
    monitor.append_live_log(_rows("2026-08-07", action="HALT-no entry", raw="BUY dip"))
    df = pd.read_csv(log)
    assert len(df) == 2
    assert df.loc[df["asof"] == "2026-08-07", "raw_action"].item() == "BUY dip"
    assert df.loc[df["asof"] == "2026-08-07", "your_action"].item() == "HALT-no entry"
