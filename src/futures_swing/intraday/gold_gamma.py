"""GC (gold) options-structure support/resistance from GLD dealer gamma.

Gold has no free listed-futures-options (OG) data, so we read the GLD ETF option
chain — the liquid, most-watched gold-gamma proxy — through QR's option_research
pipeline (reports/option_atm_flip/GLD/<date>/), then map every GLD strike to a GC
price with the live GLD->GC factor (~x10.9: GC_close / GLD_spot).

What we extract per snapshot:
  - dealer regime sign (net_gamma): <0 = net SHORT gamma -> moves AMPLIFIED, support
    is breakable (no dealer cushion); >0 = long gamma -> moves dampened, levels hold.
  - put walls below spot  -> support magnets (largest OI). In short gamma these are
    "defend-or-air-pocket" lines, not cushioned floors.
  - call walls above spot -> resistance caps (largest OI).
  - centroid / up-down pivots / zero-gamma flip / line-in-the-sand (Alma speed-profile).

DEPLOYMENT: computing needs QR + the GLD chain (local only). The briefing instead
reads the committed snapshot (tracking/gc_gamma.json) and degrades to "no data" in
CI, where QR isn't checked out. Regenerate + commit the snapshot where QR lives:
    python -m futures_swing.intraday.gold_gamma --date 2026-06-24

POINT-IN-TIME: a snapshot dated D reflects D's evening GLD chain (OI). It's a
forward-looking heads-up (where support sits now), not a backtest input — so using
the latest available date is correct here (no look-ahead concern for a digest).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data_loader
from . import gamma

SNAPSHOT = data_loader.REPO_ROOT / "tracking" / "gc_gamma.json"
WALL_WINDOW = 0.16     # ignore strikes >16% from spot (far-OTM LEAP OI isn't a near-term level)
N_WALLS = 3            # walls per side to surface
GLD_GC_FALLBACK = 10.9 # ~ GC/GLD ratio if a same-day GC bar is missing (holiday)


def _conversion(gld_spot: float, date: str) -> tuple[float, float]:
    """GLD->GC factor = GC_close / GLD_spot. Falls back to the latest available GC
    close / a constant ratio when the date has no GC bar. Returns (factor, gc_ref)."""
    try:
        close = data_loader.load_ohlc("GC")["close"]
    except Exception:                       # no GC cache (e.g. fresh CI checkout) -> fall back
        close = pd.Series(dtype=float)
    gc = None if close.empty else close.get(pd.Timestamp(date))
    if gc is None or not np.isfinite(gc):
        gc = float(close.iloc[-1]) if not close.empty else float("nan")  # most recent settle
    if not (gld_spot and np.isfinite(gc)):
        return GLD_GC_FALLBACK, float(gc)
    return float(gc / gld_spot), float(gc)


def _walls(strike_profile: pd.DataFrame, spot: float, conv: float, below: bool) -> list[dict]:
    """Walls on one side of spot, within WALL_WINDOW, nearest-first. Ranked by
    |gamma_exposure| (hedging pressure — weights moneyness/expiry, not just raw OI),
    UNION the single heaviest-OI strike so the headline wall is never dropped."""
    df = strike_profile
    lo, hi = spot * (1 - WALL_WINDOW), spot * (1 + WALL_WINDOW)
    side = df[(df["strike"] < spot) & (df["strike"] >= lo)] if below \
        else df[(df["strike"] > spot) & (df["strike"] <= hi)]
    if side.empty:
        return []
    side = side.assign(absg=side["gamma_exposure"].abs())   # not _absg: itertuples renames leading-underscore cols
    sel = pd.concat([side.nlargest(N_WALLS, "absg"), side.nlargest(1, "openinterest")])
    sel = sel.drop_duplicates("strike").sort_values("strike", ascending=not below)
    return [dict(gc=round(r.strike * conv / 5) * 5, gld=float(r.strike),
                 oi=int(r.openinterest) if np.isfinite(r.openinterest) else 0,
                 gex=float(r.gamma_exposure),
                 sign=1 if r.gamma_exposure > 0 else -1)   # +1 dealer-long (cushion/pin) · -1 short (accelerant)
            for r in sel.itertuples()]


def compute(date: str, gamma_symbol: str = "GLD") -> dict:
    """Build the GC gamma snapshot from QR's GLD profile for ``date``. Raises if the
    profile is missing (generate it first via gamma.ensure_profiles)."""
    d = gamma.PROFILE_DIR / gamma_symbol / str(date)
    cand_p, curve_p, strikes_p = d / "candidate.json", d / "gex_curve.csv", d / "strike_profile.csv"
    if not (cand_p.exists() and strikes_p.exists()):
        raise FileNotFoundError(f"no GLD profile for {date} at {d} — run gamma.ensure_profiles")
    c = json.loads(cand_p.read_text())
    raw_spot = c.get("snapshot_spot") or c.get("prior_close")
    if raw_spot is None or not np.isfinite(float(raw_spot)):
        raise ValueError(f"GLD profile {date} has no usable spot (snapshot_spot/prior_close)")
    spot = float(raw_spot)
    conv, gc_ref = _conversion(spot, date)

    def gc(x):
        return None if x is None or not np.isfinite(x) else round(float(x) * conv / 5) * 5

    flip = None
    if curve_p.exists():
        g = pd.read_csv(curve_p)
        flip = gamma.flip_from_curve(g["spot"], g["gamma_exposure"], near=spot)

    strikes = pd.read_csv(strikes_p)
    below = _walls(strikes, spot, conv, below=True)    # geometric: strikes UNDER spot
    above = _walls(strikes, spot, conv, below=False)   # geometric: strikes OVER spot

    # tag the structural line-in-the-sand onto its exact GLD strike (compare unrounded
    # GLD, not the ×conv-rounded GC, so two bracketing strikes can't both match).
    lis_gld = (c.get("line_in_the_sand") or [None])[0]
    for s in below:
        s["line_in_sand"] = (lis_gld is not None and abs(s["gld"] - float(lis_gld)) < 0.5)
    # call wall = heaviest-OI POSITIVE-gamma strike above spot (a real dealer cap).
    # A big-OI negative-gamma strike is an upside magnet, not a cap — don't mislabel it.
    pos_above = [r for r in above if r["sign"] > 0]
    if pos_above:
        max(pos_above, key=lambda r: r["oi"])["call_wall"] = True

    raw_net = c.get("net_gamma")
    net = float(raw_net) if raw_net is not None and np.isfinite(float(raw_net)) else None
    regime = 0 if net is None else (-1 if net < 0 else 1)
    label = {-1: "net SHORT gamma — moves amplified, levels are defend-or-break",
             1: "net LONG gamma — moves dampened, levels tend to hold",
             0: "gamma regime unknown (no net_gamma) — treat levels as unconfirmed"}[regime]
    return dict(
        asof=str(date), gamma_symbol=gamma_symbol,
        conv=round(conv, 3), gld_spot=round(spot, 2), gc_ref=round(gc_ref, 1),
        regime=regime, regime_label=label,
        speed_dir=c.get("speed_direction"),
        net_gamma=net,
        centroid_gc=gc(c.get("centroid_low")),
        flip_gc=gc(flip),
        upside_pivot_gc=gc(c.get("upside_pivot")),
        downside_pivot_gc=gc(c.get("downside_pivot")),
        line_in_sand_gc=gc(lis_gld),
        below=below, above=above,        # geometric sides; +g/-g sign per level says cushion vs accelerant
        tags=c.get("qualitative_tags"),
    )


def write_snapshot(date: str, path: Path = SNAPSHOT, ensure: bool = True) -> dict:
    """Compute and persist the snapshot. ``ensure`` first asks QR to generate the
    GLD profile if it's missing (needs the GLD chain.parquet present in QR)."""
    if ensure:
        gamma.ensure_profiles([str(date)], symbol="GLD")
    snap = compute(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False guarantees strict JSON; atomic replace so an interrupted regen
    # can't leave a half-written (partial-but-truthy) file for the briefing to read.
    body = json.dumps(snap, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, path)
    return snap


def load_snapshot(path: Path = SNAPSHOT) -> dict | None:
    """Read the committed snapshot for the briefing. None if absent/unreadable/not a
    JSON object (CI without QR, or a corrupt/partial file)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the GC gamma snapshot from GLD options")
    ap.add_argument("--date", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--no-ensure", action="store_true",
                    help="don't shell out to QR to generate a missing profile")
    ap.add_argument("--out", default=str(SNAPSHOT))
    args = ap.parse_args()
    date = str(pd.Timestamp(args.date).date())   # normalize any accepted date form to YYYY-MM-DD
    snap = write_snapshot(date, Path(args.out), ensure=not args.no_ensure)
    print(json.dumps(snap, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
