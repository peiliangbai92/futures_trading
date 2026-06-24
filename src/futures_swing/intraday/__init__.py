"""Intraday ES sleeve (MVP) — dealer-gamma-regime-gated 5-minute strategy.

Hypothesis H1: on POSITIVE net-gamma days dealer hedging dampens intraday moves
(mean-reversion → fade extensions back toward VWAP/flip); on negative-gamma days
it amplifies them (don't fade). The MVP trades only the positive-gamma half.

Data (free, "test the waters"): yfinance 5-min ES (~60d) + daily SPY dealer-gamma
profiles produced by the sibling QuantitativeResearch option_research pipeline.
~20-25 usable days after the point-in-time shift → a FEASIBILITY PROBE, not a
validation. If the probe is encouraging, move to Databento + months of gamma.

Point-in-time discipline (we just got burned by a look-ahead in the daily model):
a gamma profile dated D reflects D's evening OI, so trading D's session uses the
D-1 profile; intraday features at bar t use only bars <= t.
"""
