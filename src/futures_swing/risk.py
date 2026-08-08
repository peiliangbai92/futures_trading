"""Risk layer — position sizing, portfolio gates, and event filter.

Sizing is risk-based: each trade risks a fixed fraction of equity to its stop,
scaled by conviction (|Sharpe|), expressed in **micro contracts** (MES $5/pt,
MGC $10/pt) so a small account can size finely. Portfolio gates: cut size on
drawdown, and stand aside around macro events.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .execution import ATR_STOP_MULT

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_CSV = REPO_ROOT / "configs" / "event_calendar.csv"

RISK_FRAC = 0.0075        # per-trade risk as a fraction of equity (0.5-1%)
DD_HALVE = 0.05           # halve size beyond 5% drawdown
DD_HALT = 0.10            # stop trading beyond 10% drawdown
STRENGTH_SCALE = 0.5      # |Sharpe| at which the conviction multiplier saturates
MAX_CONTRACTS = 10
TARGET_VOL = 0.10         # annualized per-trade vol target for vol_target sizing


def conviction(sharpe: float, *, strength_scale: float = STRENGTH_SCALE) -> float:
    """Signal-strength multiplier in [0, 1] from |Sharpe|."""
    if not np.isfinite(sharpe):
        return 0.0
    return float(min(abs(sharpe) / strength_scale, 1.0))


def position_size(
    equity: float,
    sharpe: float,
    atr: float,
    point_value: float,
    *,
    risk_frac: float = RISK_FRAC,
    stop_mult: float = ATR_STOP_MULT,
    strength_scale: float = STRENGTH_SCALE,
    max_contracts: int = MAX_CONTRACTS,
) -> int:
    """Number of micro contracts for one trade (0 if untradeable)."""
    if not np.isfinite(sharpe) or not (atr > 0) or equity <= 0:
        return 0
    dollar_risk_per_contract = stop_mult * atr * point_value
    if dollar_risk_per_contract <= 0:
        return 0
    conv = conviction(sharpe, strength_scale=strength_scale)
    raw = (equity * risk_frac) / dollar_risk_per_contract * conv
    return int(min(np.floor(raw), max_contracts))


def vol_target_size(
    equity: float,
    forecast_vol_annual: float,
    price: float,
    point_value: float,
    *,
    target_vol: float = TARGET_VOL,
    conviction_mult: float = 1.0,
    max_contracts: int = MAX_CONTRACTS,
) -> int:
    """Micro contracts sizing the *notional* so a position's annualized vol hits
    ``target_vol`` (scaled by conviction). This actually deploys capital, unlike
    pure stop-risk sizing which stays tiny when ATR is large vs the risk budget."""
    if not (forecast_vol_annual > 0) or price <= 0 or equity <= 0:
        return 0
    notional = equity * target_vol / forecast_vol_annual * conviction_mult
    contracts = notional / (price * point_value)
    return int(min(np.floor(contracts), max_contracts))


@dataclass
class RiskManager:
    """Tracks equity peak and translates drawdown into a size multiplier."""

    equity0: float
    peak: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.peak = max(self.peak, self.equity0)

    def update(self, equity: float) -> None:
        self.peak = max(self.peak, equity)

    def drawdown(self, equity: float) -> float:
        return 1.0 - equity / self.peak if self.peak > 0 else 0.0

    def size_multiplier(self, equity: float) -> float:
        dd = self.drawdown(equity)
        if dd >= DD_HALT:
            return 0.0
        if dd >= DD_HALVE:
            return 0.5
        return 1.0


# --------------------------------------------------------------- event filter


def load_event_dates(path: Path | None = None) -> set[date]:
    """Parse the macro event calendar (skips comment lines).

    Warns when the calendar is close to running out: past its last row the
    stand-aside filter silently becomes a no-op, so the warning is the only
    signal that ``configs/event_calendar.csv`` needs next year's dates."""
    path = path or EVENT_CSV
    if not path.exists():
        return set()
    out: set[date] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("date,"):
            continue
        try:
            out.add(date.fromisoformat(line.split(",", 1)[0]))
        except ValueError:
            continue
    if out and (max(out) - date.today()).days < 30:
        warnings.warn(
            f"event calendar ends {max(out)} — extend configs/event_calendar.csv "
            "or the event stand-aside filter will silently stop blocking",
            stacklevel=2,
        )
    return out


def near_event(d, event_dates: set[date], *, window: int = 1) -> bool:
    """True if ``d`` is within ``window`` calendar days of a macro event."""
    if not event_dates:
        return False
    dd = pd.Timestamp(d).date()
    return any(abs((dd - e).days) <= window for e in event_dates)
