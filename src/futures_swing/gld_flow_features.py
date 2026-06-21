"""GLD options dealer-flow features for the GC model.

Reads the per-day GLD option-analysis output produced by the sibling
QuantitativeResearch ``option_research`` pipeline (``reports/option/GLD/<date>/``:
``candidate.json`` + ``iv_skew_term_structure.csv``) and turns each daily snapshot
into a row of stationary, spot-normalized features — the genuinely orthogonal
"options-flow" information (dealer GEX positioning + implied skew/risk-reversal)
that price+macro features can't see.

CAVEAT: GLD option-chain history starts 2026-06-21 (collection just began), so
there is not yet enough history to backtest these against GC's forward return.
This extractor is the ready-to-go plumbing: as the daily cron accrues snapshots
(or once historical chains are purchased), call ``feature_panel`` to get a daily
feature frame and evaluate it as GC features.

CLI:  python -m futures_swing.gld_flow_features --date 2026-06-21
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QR_OPTION_ROOT = Path("/Users/peiliangbai/Documents/GitHub/QuantitativeResearch/reports/option")
SYMBOL = "GLD"

_SPEED_DIR = {"negative": -1.0, "flat_to_negative": -0.5, "flat": 0.0,
              "flat_to_positive": 0.5, "positive": 1.0}


def extract_day(date: str, symbol: str = SYMBOL, root: Path = QR_OPTION_ROOT) -> dict | None:
    """One day's GLD dealer-flow feature row (spot-normalized, stationary)."""
    d = root / symbol / date
    cj, skew_csv = d / "candidate.json", d / "iv_skew_term_structure.csv"
    if not cj.exists():
        return None
    c = json.loads(cj.read_text())
    spot = c.get("snapshot_spot") or c.get("prior_close")
    if not spot:
        return None
    def dist(key):
        v = c.get(key)
        return (v - spot) / spot if v is not None else np.nan

    row = {
        "date": pd.Timestamp(date),
        # --- dealer-GEX positioning (where the hedging flows pull/repel price) ---
        "gex_centroid_dist": dist("centroid_low"),        # GEX center of mass vs spot
        "gex_up_pivot_dist": dist("upside_pivot"),
        "gex_dn_pivot_dist": dist("downside_pivot"),
        "gex_dn_target_dist": dist("downside_target"),    # downside tail magnet
        "gex_pin_strength": c.get("pin_strength"),
        "gex_speed_dir": _SPEED_DIR.get(c.get("speed_direction"), np.nan),  # dealer gamma regime
        "implied_move_1sig": ((c.get("sigma_1_upper", np.nan) - c.get("sigma_1_lower", np.nan)) / spot
                              if c.get("sigma_1_upper") else np.nan),
        "vol_bought": 1.0 if "vol_bought" in (c.get("qualitative_tags") or []) else 0.0,
        "defending_upside": 1.0 if "defending_upside" in (c.get("qualitative_tags") or []) else 0.0,
    }
    # --- implied skew / risk-reversal (directional sentiment) from the front expiry ---
    if skew_csv.exists():
        sk = pd.read_csv(skew_csv)
        if len(sk):
            front = sk.sort_values("days_to_expiry").iloc[0]
            for col in ("atm_iv", "atm_iv_skew", "risk_reversal_25d",
                        "atm_skew_power_law_exponent", "surface_h_proxy"):
                if col in sk.columns:
                    row[f"skew_{col}"] = float(front[col])
    return row


def feature_panel(symbol: str = SYMBOL, root: Path = QR_OPTION_ROOT) -> pd.DataFrame:
    """Daily feature frame over all collected GLD snapshots."""
    rows = [extract_day(p.name, symbol, root) for p in sorted((root / symbol).glob("20*")) if p.is_dir()]
    rows = [r for r in rows if r]
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="GLD options dealer-flow features for GC")
    ap.add_argument("--date", default=None, help="single date (default: show the full panel)")
    args = ap.parse_args()
    if args.date:
        row = extract_day(args.date)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in (row or {}).items() if k != "date"}, indent=2))
    else:
        panel = feature_panel()
        print(f"GLD dealer-flow feature panel: {panel.shape[0]} days x {panel.shape[1]} features")
        if not panel.empty:
            print(f"history: {panel.index[0].date()} .. {panel.index[-1].date()}")
            print(panel.tail().to_string())


if __name__ == "__main__":
    main()
