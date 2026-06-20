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
        "regime": "hmm",         # market HMM lifts ES OOS IC (V1.2)
    },
    "GC": {
        "yf_symbol": "GC=F",
        "micro_symbol": "MGC",
        "point_value": 10.0,     # MGC = $10 / $1 move
        "tick": 0.10,
        "horizon": 10,
        "regime": "rule",        # HMM (market or gold) did not help GC — rule is simplest
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
