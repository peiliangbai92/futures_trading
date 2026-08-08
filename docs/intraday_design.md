# Futures trading signal design (intraday GC — MTF RSI)

> **Status (2026-08):** implemented as a TradingView Pine indicator —
> `pine/gc_mtf_rsi.pine` (see `pine/README.md`). The IBKR 5-minute data
> pipeline described in Section 1 was **not** built: IBKR's API needs a live
> TWS/Gateway session (can't run in GitHub Actions) and 5-minute history is
> heavy, while TradingView serves every timeframe natively and pushes phone
> alerts. NQ was never in scope of the build. The Pine script extends this
> spec with a 1m rung, a symmetric short side, a 4h Bollinger-band trigger,
> a trend filter, and breakeven/ATR-trailing exits.

## 1. Data collection
Don't need to change this part, just use the same method we already used in this repo, i.e., use IBKR API to collect data. But I need you to confirm that whether it's legal to collect data from IBKR in 5min frequency basis. All I need is ES, NQ, and GC OHLCV data. 

## 2. One signal for GC
- Use 5min GC OHLCV data, then we fristly aggregate data to create 10min, 30min, 1h, and 4h frequency time series;
- Then we calculate RSI for each time frame, the parameter for RSI is using SMA and time length is 14;
- Then we calculate some other indicators: momentum, RSI divergence (for each time frame);
- The signal is simple: 
    - If we need to find an entry point, which is defined as: (5min RSI <= 30 && 10min RSI <= 30 && 30min RSI <= 35 && 1h RSI <= 37 && 4h RSI <= 40) OR ((5min RSI divergence is bull OR 10min RSI divergence is bull OR 30min RSI divergence is bull) AND (1h RSI and 4h RSI are both less than 40));
    - We also calculate momentum for each time frame, an entry point should require 5min, 10min, 30min momentum are all significant positive, 1h and 4h are not negative.
