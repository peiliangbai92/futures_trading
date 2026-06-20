"""Execution rules — entry / stop / target levels and intrabar exit logic.

Alpha decides *direction*; execution decides *price levels* (doc Sections 10-11).
V1 keeps it simple and backtestable: enter at the next session's open, ATR-based
stop and target, plus a time stop (the horizon) and a signal-reversal exit
handled by the backtest loop.
"""
from __future__ import annotations

ATR_STOP_MULT = 2.0
ATR_TARGET_MULT = 3.0


def levels(
    side: int,
    entry: float,
    atr: float,
    *,
    stop_mult: float = ATR_STOP_MULT,
    target_mult: float = ATR_TARGET_MULT,
) -> dict[str, float]:
    """Stop/target for a position. ``side`` = +1 long, -1 short."""
    if side > 0:
        return {"entry": entry, "stop": entry - stop_mult * atr, "target": entry + target_mult * atr}
    return {"entry": entry, "stop": entry + stop_mult * atr, "target": entry - target_mult * atr}


def hit_exit(side: int, bar_high: float, bar_low: float, stop: float, target: float) -> tuple[str | None, float | None]:
    """Did this bar hit the stop or target? Conservative: if both are touched in
    the same bar (no intrabar path), assume the stop filled first."""
    if side > 0:
        hit_stop = bar_low <= stop
        hit_target = bar_high >= target
    else:
        hit_stop = bar_high >= stop
        hit_target = bar_low <= target
    if hit_stop:
        return "stop", stop
    if hit_target:
        return "target", target
    return None, None
