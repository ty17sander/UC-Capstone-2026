"""Synthetic market data with a *known* embedded signal.

This exists for a specific reason beyond convenience. If you only ever
test on real data, you cannot distinguish "the harness is correct and the
signal is weak" from "the harness is broken". By generating data with a
planted, known-strength effect, you can verify the evaluation code
recovers it -- and, just as importantly, verify it reports ~zero IC on
pure noise. A harness that finds signal in noise is worse than no harness.

Also useful for CI: no network, deterministic, runs in a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(
    n_days: int = 1500,
    n_tickers: int = 120,
    n_sectors: int = 8,
    *,
    signal_strength: float = 0.03,
    market_vol: float = 0.011,
    sector_vol: float = 0.007,
    idio_vol: float = 0.016,
    seed: int = 7,
) -> dict:
    """Generate a panel with market, sector, and idiosyncratic components.

    ``signal_strength`` is the correlation planted between a lagged
    momentum-like feature and the next period's idiosyncratic return.
    Values around 0.02-0.05 are realistic for equity signals; anything
    much higher is fantasy.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_tickers)]
    sector_ids = rng.integers(0, n_sectors, size=n_tickers)
    sectors = pd.Series(
        [f"Sector_{s}" for s in sector_ids], index=tickers, name="sector"
    )

    market = rng.normal(0.0004, market_vol, size=n_days)
    sector_ret = rng.normal(0.0, sector_vol, size=(n_days, n_sectors))
    idio = rng.normal(0.0, idio_vol, size=(n_days, n_tickers))

    # Plant a predictive relationship: a persistent latent score that
    # tilts each name's future idiosyncratic return.
    latent = np.zeros((n_days, n_tickers))
    latent[0] = rng.normal(0, 1, n_tickers)
    for t in range(1, n_days):
        latent[t] = 0.97 * latent[t - 1] + rng.normal(0, 0.24, n_tickers)

    idio[1:] += signal_strength * idio_vol * latent[:-1] * 10.0

    total = (market[:, None] + sector_ret[:, sector_ids] + idio)

    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(total, axis=0)), index=dates, columns=tickers
    )
    base_vol = rng.lognormal(14.0, 0.8, size=n_tickers)
    vol_noise = rng.lognormal(0.0, 0.35, size=(n_days, n_tickers))
    volume = pd.DataFrame(base_vol * vol_noise, index=dates, columns=tickers).round()

    membership = pd.DataFrame(True, index=dates, columns=tickers)
    # Simulate entries and exits so survivorship handling gets exercised.
    for i, tk in enumerate(tickers):
        if i % 17 == 0:
            membership.iloc[: 200 + i, membership.columns.get_loc(tk)] = False
        if i % 23 == 0:
            membership.iloc[n_days - 150 - i:, membership.columns.get_loc(tk)] = False

    return {
        "close": close,
        "volume": volume,
        "sectors": sectors,
        "membership": membership,
        "latent": pd.DataFrame(latent, index=dates, columns=tickers),
    }


def make_pure_noise(n_days: int = 1500, n_tickers: int = 120, seed: int = 11) -> dict:
    """Same shape, zero planted signal. The harness must find nothing here."""
    return make_market(
        n_days=n_days, n_tickers=n_tickers, signal_strength=0.0, seed=seed
    )
