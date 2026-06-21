"""futures_swing — systematic swing-trading model for ES and GC futures.

Pipeline: data_loader -> features -> regime -> model (LightGBM alpha) ->
signal (risk-adjusted) -> execution (entry/stop/target) -> risk (sizing +
portfolio gates) -> backtest / pipeline (daily output).

V1 uses yfinance daily data only. See README.md for scope and the plan.
"""

__version__ = "0.1.0"

# Instrument specs (micros — V1 default per plan).
INSTRUMENTS = {
    "ES": {
        "yf_symbol": "ES=F",
        "micro_symbol": "MES",
        "point_value": 5.0,      # MES = $5 / index point
        "tick": 0.25,
        "horizon": 5,            # forward-return / holding horizon (trading days)
        "regime": "hmm",         # market HMM lifts ES OOS IC (V1.2); unused by the ridge sleeve
        # V1.4: ES is a linear short-horizon mean-reversion problem. A ridge sleeve
        # on ret_5/ret_20 gives a *significant* OOS IC (~+0.073, block-boot CI excludes
        # 0) where the 23-feature LightGBM diluted it to noise (+0.015). The reversion
        # forecast is small (signal std ~0.09) and broad/weak-per-trade, so we trade it
        # at a LOW threshold (Sharpe is a flat ~0.45-0.50 plateau across th 0.08-0.14)
        # and LONG-ONLY: shorting after rallies fights ES's secular up-drift and lost
        # money (short legs −$4.5k vs long legs +$38k in the 2-sided backtest).
        "alpha": {"kind": "ridge", "features": ["ret_5", "ret_20"], "ridge_alpha": 10.0},
        "signal_th": 0.12,
        "long_only": True,
        # Exit = ATR stop/target + 5d time-stop (the default). KEY finding from the
        # V1.5/1.6 exit study: quick exits are what let the strategy BUY THE NEXT LOW
        # (it cycled out and had dry powder — e.g. it bought the Apr-2025 crash bottom
        # @5097). "Hold-to-sell-high" variants get stuck fully invested and CANNOT buy
        # the bottoms (scale-in added 0 lots at the Apr-2025 low), and run -16/-25%
        # drawdowns. So "sell early" and "buy the bottom" are the same coin; the ATR
        # exit gives the best risk-adjusted result (Sharpe ~0.47, maxDD ~-4.3%).
    },
    "GC": {
        "yf_symbol": "GC=F",
        "micro_symbol": "MGC",
        "point_value": 10.0,     # MGC = $10 / $1 move
        "tick": 0.10,
        "horizon": 10,
        "regime": "rule",        # HMM (market or gold) did not help GC — rule is simplest
        # GC carries real nonlinearity only LightGBM captures (ridge kills the edge).
        "alpha": {"kind": "lgbm", "features": "all"},
        "signal_th": 0.35,
    },
}

# Cross-asset / macro feeds available from yfinance (V1).
MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "DXY": "DX-Y.NYB",
    "UST10Y": "^TNX",   # 10Y yield (e.g. 4.49 == 4.49%)
    "TIP": "TIP",       # iShares TIPS ETF (real-yield proxy)
    "OIL": "CL=F",      # WTI crude front month
    "HYG": "HYG",       # iShares high-yield credit ETF (V1.1 credit proxy)
    "LQD": "LQD",       # iShares IG credit ETF (V1.1 credit proxy)
}

# FRED macro series (V1.1) — keyless fredgraph.csv, full history. Keyed by our
# internal name -> FRED series id. (HY/IG OAS are omitted: the keyless CSV only
# serves ~3y for those; add via the FRED API key in a later pass.)
FRED_SERIES = {
    "UST2Y": "DGS2",          # 2Y Treasury constant-maturity yield (%)
    "CURVE_2S10S": "T10Y2Y",  # 10Y-2Y term-spread (%)
    "REAL_YIELD": "DFII10",   # 10Y TIPS real yield (%) — key gold driver
    "BREAKEVEN": "T10YIE",    # 10Y breakeven inflation (%)
}
