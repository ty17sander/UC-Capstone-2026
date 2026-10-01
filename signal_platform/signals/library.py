"""Starting signal library.

Deliberately restricted to price- and volume-derived signals. Free
fundamental data (yfinance included) reports *current, restated*
financials with no as-reported timestamp, so using it in a historical
backtest is lookahead bias. Price and volume carry unambiguous
timestamps.

Signals are drawn from distinct families on purpose. Per the Fundamental
Law of Active Management, adding a signal correlated 0.9 with one you
already hold buys you almost nothing; the value is in weak, uncorrelated
bets. Family labels make that structure explicit and testable.

These are all previously published hypotheses rather than inventions of
this project. Testing prior hypotheses is research; searching for the
best-performing formula in-sample is data mining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MarketData, Signal, SignalRegistry


class Momentum12_1(Signal):
    """Twelve-month return, skipping the most recent month.

    The skip matters: the omitted month is where short-term reversal
    lives, and including it contaminates a medium-term momentum signal
    with an effect that points the opposite way.
    """
    family = "momentum"
    rationale = ("Medium-term price trends persist; the one-month skip avoids "
                 "contamination from short-horizon reversal.")

    def __init__(self, lookback: int = 252, skip: int = 21):
        super().__init__(lookback=lookback, skip=skip)

    def compute(self, data: MarketData) -> pd.DataFrame:
        c, lb, sk = data.close, self.params["lookback"], self.params["skip"]
        return c.shift(sk) / c.shift(lb) - 1.0


class ShortTermReversal(Signal):
    """Negated one-month return. Recent losers tend to bounce."""
    family = "reversal"
    rationale = ("Short-horizon overreaction mean-reverts; sign flipped so that "
                 "higher score still means more attractive long.")

    def __init__(self, lookback: int = 21):
        super().__init__(lookback=lookback)

    def compute(self, data: MarketData) -> pd.DataFrame:
        c, lb = data.close, self.params["lookback"]
        return -(c / c.shift(lb) - 1.0)


class LowVolatility(Signal):
    """Negated realised volatility. The low-vol anomaly."""
    family = "risk"
    rationale = ("Low-volatility stocks have historically delivered better "
                 "risk-adjusted returns than CAPM predicts.")

    def __init__(self, window: int = 63):
        super().__init__(window=window)

    def compute(self, data: MarketData) -> pd.DataFrame:
        w = self.params["window"]
        vol = data.returns.rolling(w, min_periods=w // 2).std()
        return -vol


class TrendStrength(Signal):
    """Distance of price above its long moving average, vol-scaled.

    Vol-scaling is what stops this collapsing into a pure volatility bet:
    without it, high-vol names mechanically sit further from their mean.
    """
    family = "trend"
    rationale = "Price extension above trend, normalised so it is not a vol proxy."

    def __init__(self, window: int = 200, vol_window: int = 63):
        super().__init__(window=window, vol_window=vol_window)

    def compute(self, data: MarketData) -> pd.DataFrame:
        c, w, vw = data.close, self.params["window"], self.params["vol_window"]
        ma = c.rolling(w, min_periods=w // 2).mean()
        vol = data.returns.rolling(vw, min_periods=vw // 2).std()
        return ((c / ma) - 1.0) / vol.replace(0.0, np.nan)


class LiquidityTurnover(Signal):
    """Negated share turnover. Low-turnover names earn an illiquidity premium."""
    family = "liquidity"
    rationale = ("Illiquidity premium: less-traded names compensate holders for "
                 "transaction costs and attention scarcity.")

    def __init__(self, window: int = 63):
        super().__init__(window=window)

    def compute(self, data: MarketData) -> pd.DataFrame:
        data.require("volume")
        w = self.params["window"]
        dollar_vol = (data.close * data.volume).rolling(w, min_periods=w // 2).mean()
        return -np.log(dollar_vol.replace(0.0, np.nan))


class IdiosyncraticMomentum(Signal):
    """Momentum measured net of the equal-weight market return.

    Strips the market component out of each name's trend, leaving the
    stock-specific part. Structurally close to Momentum12_1 but usually
    correlated well below 1.0 with it, which is the interesting part.
    """
    family = "momentum"
    rationale = "Stock-specific trend after removing the common market factor."

    def __init__(self, lookback: int = 252, skip: int = 21):
        super().__init__(lookback=lookback, skip=skip)

    def compute(self, data: MarketData) -> pd.DataFrame:
        rets = data.returns
        market = rets.mean(axis=1)
        excess = rets.sub(market, axis=0)
        cum = (1.0 + excess.fillna(0.0)).cumprod()
        lb, sk = self.params["lookback"], self.params["skip"]
        return cum.shift(sk) / cum.shift(lb) - 1.0


class VolumeShock(Signal):
    """Negated recent volume spike relative to its own baseline.

    Abnormal volume clusters around news; the reversal tendency after
    attention-driven spikes motivates the negative sign.
    """
    family = "attention"
    rationale = "Attention-driven volume spikes tend to partially reverse."

    def __init__(self, short: int = 5, long: int = 63):
        super().__init__(short=short, long=long)

    def compute(self, data: MarketData) -> pd.DataFrame:
        data.require("volume")
        s, l = self.params["short"], self.params["long"]
        v = data.volume
        fast = v.rolling(s, min_periods=max(2, s // 2)).mean()
        slow = v.rolling(l, min_periods=l // 2).mean()
        return -np.log((fast / slow.replace(0.0, np.nan)).replace(0.0, np.nan))


def default_registry() -> SignalRegistry:
    """The starting library: seven signals across six families."""
    reg = SignalRegistry()
    for sig in (
        Momentum12_1(),
        IdiosyncraticMomentum(),
        ShortTermReversal(),
        LowVolatility(),
        TrendStrength(),
        LiquidityTurnover(),
        VolumeShock(),
    ):
        reg.add(sig)
    return reg
