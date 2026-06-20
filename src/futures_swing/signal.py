"""Signal generation — risk-adjusted (Sharpe) signal from the alpha forecast.

The tradeable quantity is not the raw predicted return but the *risk-adjusted*
forecast: ``sharpe = predicted_return / forecast_vol`` (doc Section 9), where
forecast vol is trailing close-to-close realized vol scaled to the forecast
horizon. Thresholds map the Sharpe signal to long / flat / short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import INSTRUMENTS, data_loader
from . import vol as volmod

DEFAULT_LONG_TH = 0.35
DEFAULT_SHORT_TH = -0.35
VOL_WINDOW = 21
TRADING_DAYS = 252


def horizon_forecast_vol(close: pd.Series, horizon: int, *, window: int = VOL_WINDOW) -> pd.Series:
    """Trailing realized vol scaled to the horizon (log-return units)."""
    ann = volmod.close_to_close_volatility(close, window=window)
    return (ann * np.sqrt(horizon / TRADING_DAYS)).rename("fc_vol")


def compute_signals(
    symbol: str,
    pred: pd.Series,
    *,
    long_th: float = DEFAULT_LONG_TH,
    short_th: float = DEFAULT_SHORT_TH,
    window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """Turn a forecast-return series into pred/fc_vol/sharpe/signal columns."""
    horizon = INSTRUMENTS[symbol]["horizon"]
    close = data_loader.load_ohlc(symbol)["close"]
    fc = horizon_forecast_vol(close, horizon, window=window).reindex(pred.index)
    sharpe = (pred / fc).replace([np.inf, -np.inf], np.nan)
    signal = pd.Series(0, index=pred.index, dtype=int)
    signal[sharpe >= long_th] = 1
    signal[sharpe <= short_th] = -1
    return pd.DataFrame(
        {"pred_ret": pred, "fc_vol": fc, "sharpe": sharpe, "signal": signal}
    )


def discretize(sharpe_value: float, *, long_th: float = DEFAULT_LONG_TH, short_th: float = DEFAULT_SHORT_TH) -> int:
    """Scalar Sharpe -> {-1, 0, +1}."""
    if not np.isfinite(sharpe_value):
        return 0
    if sharpe_value >= long_th:
        return 1
    if sharpe_value <= short_th:
        return -1
    return 0
