"""ES continuous-futures roll-jump cleaner (cash-anchored ratio back-adjustment).

yfinance's ES=F continuous series is NOT back-adjusted: at each quarterly roll the
front contract switches to the next, which trades at a different price (the
financing − dividend carry), injecting a SPURIOUS overnight return into the
continuous series. Alma (*Futures Trading Pt 1*) describes exactly this and notes
that professional platforms retroactively shift the history to remove the gap —
which is also what IBKR's back-adjusted ContFuture does (and yfinance does not).

We don't have both contracts at the roll, but we DO have the cash index (^GSPC):
ES is delta-1 to SPX, so the two move 1:1 in *return* terms, and the cash has NO
roll gap. So on each quarterly roll window we pick the day whose ES-vs-cash
overnight-return divergence is anomalously large (that's the roll gap) and
ratio-back-adjust all PRIOR bars to cancel it — replacing that one overnight
return with the cash return. Anchoring is at the recent end (the corrected bars
are all strictly BEFORE the roll), so the latest price, today's signal, and the
briefing are unchanged; only pre-roll history is de-gapped.

Validated against the IBKR back-adjusted ContFuture series (ground truth) over
2022-2026 (see the A/B in reports/diagnostics / the build notes).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

ROLL_MONTHS = (3, 6, 9, 12)   # ES quarterly expiries: 3rd Friday of Mar/Jun/Sep/Dec
PRE_DAYS = 10                 # roll window opens this many days before the 3rd Friday
POST_DAYS = 5                 # ... and closes this many days after
GAP_THRESHOLD = 0.004         # min |ES−cash overnight log-divergence| to call it a roll gap


def _third_friday(y: int, m: int) -> date:
    first = date(y, m, 1)
    return date(y, m, 1 + (4 - first.weekday()) % 7 + 14)   # weekday(): Mon=0 .. Fri=4


def _roll_windows(index: pd.DatetimeIndex, pre: int, post: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []
    wins = []
    for y in range(index[0].year, index[-1].year + 1):
        for m in ROLL_MONTHS:
            tf = pd.Timestamp(_third_friday(y, m))
            wins.append((tf - pd.Timedelta(days=pre), tf + pd.Timedelta(days=post)))
    return wins


def detect_roll_days(close: pd.Series, cash: pd.Series, *,
                     pre: int = PRE_DAYS, post: int = POST_DAYS,
                     threshold: float = GAP_THRESHOLD) -> dict[pd.Timestamp, float]:
    """{roll_day -> spurious overnight log-gap}. The roll day is the largest
    ES-vs-cash overnight-divergence day in each quarterly window, if it exceeds
    ``threshold`` (so windows with no real gap are left alone)."""
    ret_es = np.log(close / close.shift(1))
    c = cash.reindex(close.index).ffill()
    gap = (ret_es - np.log(c / c.shift(1))).dropna()
    rolls: dict[pd.Timestamp, float] = {}
    for lo, hi in _roll_windows(close.index, pre, post):
        sub = gap[(gap.index >= lo) & (gap.index <= hi)]
        if sub.empty:
            continue
        r = sub.abs().idxmax()
        if abs(sub[r]) > threshold:
            rolls[r] = float(sub[r])
    return rolls


def backadjust(ohlc: pd.DataFrame, cash: pd.Series, *,
               pre: int = PRE_DAYS, post: int = POST_DAYS,
               threshold: float = GAP_THRESHOLD, return_rolls: bool = False):
    """Ratio back-adjust an ES continuous OHLC frame to remove roll-jump gaps.

    For each detected roll at ``r`` with spurious log-gap ``g``, every bar strictly
    before ``r`` is scaled by ``exp(g)`` so the cleaned overnight return into ``r``
    equals the cash return. Bars at/after the last roll are unchanged (latest price
    preserved). Returns the cleaned frame (and the roll dict if ``return_rolls``).
    """
    if "close" not in ohlc.columns:
        return (ohlc, {}) if return_rolls else ohlc
    rolls = detect_roll_days(ohlc["close"], cash, pre=pre, post=post, threshold=threshold)
    log_adj = pd.Series(0.0, index=ohlc.index)
    for r, g in rolls.items():
        log_adj.loc[ohlc.index < r] += g
    factor = np.exp(log_adj)
    out = ohlc.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] * factor
    return (out, rolls) if return_rolls else out
