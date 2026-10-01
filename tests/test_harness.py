"""Validation of the evaluation harness itself.

The two tests that matter most are ``test_recovers_planted_signal`` and
``test_finds_nothing_in_noise``. Together they establish that the harness
has both power and correct size -- it detects real effects and does not
manufacture fake ones. Every other result the platform produces is only
trustworthy if these two pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_platform.data.prices import forward_returns
from signal_platform.data.synthetic import make_market, make_pure_noise
from signal_platform.research import evaluation as ev
from signal_platform.research import transforms as tf
from signal_platform.research.backtest import (
    purged_walk_forward_splits,
    run_backtest,
)
from signal_platform.signals import MarketData, default_registry


def _market_data(bundle: dict) -> MarketData:
    return MarketData(
        close=bundle["close"],
        volume=bundle["volume"],
        sectors=bundle["sectors"],
        membership=bundle["membership"],
    )


# ---------------------------------------------------------------------------
# Correctness of the lookahead guard
# ---------------------------------------------------------------------------
def test_forward_returns_have_no_lookahead():
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1, 146.41]},
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    fwd = forward_returns(close, horizon=1, lag=1)
    # At t=0, lag=1 means we trade at t=1 and hold to t=2: 110 -> 121.
    assert np.isclose(fwd["A"].iloc[0], 0.10)
    # The final two rows cannot be known and must be NaN.
    assert fwd["A"].iloc[-1] != fwd["A"].iloc[-1]
    assert fwd["A"].iloc[-2] != fwd["A"].iloc[-2]
    print("PASS  forward_returns lag alignment correct")


def test_zero_lag_would_leak():
    """Documents *why* TRADE_LAG exists by showing the leak it prevents."""
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    leaky = forward_returns(close, horizon=1, lag=0)
    # With lag=0 the value at t=0 is the return of the very bar on which
    # the signal was observed -- unattainable in practice.
    assert np.isclose(leaky["A"].iloc[0], 0.10)
    print("PASS  zero-lag leak demonstrated (and avoided by default)")


# ---------------------------------------------------------------------------
# Power and size
# ---------------------------------------------------------------------------
def test_recovers_planted_signal():
    bundle = make_market(n_days=1200, n_tickers=100, signal_strength=0.05, seed=3)
    latent = bundle["latent"]
    prepared = tf.prepare(latent, bundle["sectors"], neutralize=False)

    fwd = forward_returns(bundle["close"], horizon=1, lag=1)
    ic = ev.information_coefficient(prepared, fwd)
    stats = ev.ic_summary(ic, horizon=1)

    assert stats["mean_ic"] > 0.02, f"planted signal not recovered: {stats}"
    assert stats["t_stat"] > 3.0, f"IC not significant: {stats}"
    print(f"PASS  planted signal recovered: IC={stats['mean_ic']:.4f} "
          f"t={stats['t_stat']:.2f}")


def test_finds_nothing_in_noise():
    bundle = make_pure_noise(n_days=1200, n_tickers=100, seed=5)
    latent = bundle["latent"]
    prepared = tf.prepare(latent, bundle["sectors"], neutralize=False)

    fwd = forward_returns(bundle["close"], horizon=1, lag=1)
    ic = ev.information_coefficient(prepared, fwd)
    stats = ev.ic_summary(ic, horizon=1)

    assert abs(stats["mean_ic"]) < 0.02, f"harness found signal in noise: {stats}"
    assert abs(stats["t_stat"]) < 3.0, f"spurious significance: {stats}"
    print(f"PASS  no signal found in noise: IC={stats['mean_ic']:.4f} "
          f"t={stats['t_stat']:.2f}")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def test_sector_neutralization_removes_sector_mean():
    bundle = make_market(n_days=300, n_tickers=60, seed=9)
    sectors = bundle["sectors"]
    raw = bundle["latent"]
    neutral = tf.sector_neutralize(raw, sectors)

    last = neutral.iloc[-1]
    by_sector = last.groupby(sectors.reindex(last.index)).mean()
    assert by_sector.abs().max() < 1e-9, "sector means not removed"
    print("PASS  sector neutralization zeroes within-sector means")


def test_zscore_properties():
    bundle = make_market(n_days=200, n_tickers=50, seed=13)
    z = tf.zscore(bundle["latent"])
    row = z.iloc[-1].dropna()
    assert abs(row.mean()) < 1e-9
    assert abs(row.std(ddof=1) - 1.0) < 1e-9
    print("PASS  z-score produces mean 0, sd 1 cross-sectionally")


def test_combine_equal_weight_matches_manual():
    bundle = make_market(n_days=250, n_tickers=40, seed=17)
    a = tf.zscore(bundle["latent"])
    b = tf.zscore(-bundle["latent"])
    combined = tf.combine({"a": a, "b": b})
    # Perfectly opposed signals must cancel.
    assert combined.iloc[-1].abs().max() < 1e-9
    print("PASS  combiner cancels perfectly opposed signals")


# ---------------------------------------------------------------------------
# Signal library and full pipeline
# ---------------------------------------------------------------------------
def test_all_signals_compute():
    bundle = make_market(n_days=700, n_tickers=80, seed=21)
    data = _market_data(bundle)
    reg = default_registry()
    scores = reg.compute_all(data)

    assert len(scores) == len(reg)
    for name, frame in scores.items():
        assert frame.shape[1] == bundle["close"].shape[1], name
        coverage = frame.iloc[-1].notna().mean()
        assert coverage > 0.5, f"{name} has poor coverage: {coverage:.2f}"
    print(f"PASS  all {len(scores)} signals compute with adequate coverage")


def test_membership_mask_applied():
    bundle = make_market(n_days=600, n_tickers=60, seed=23)
    data = _market_data(bundle)
    scores = default_registry().signals[0](data)
    excluded = ~bundle["membership"]
    overlap = scores.where(excluded).notna().sum().sum()
    assert overlap == 0, "signal scored names outside the universe"
    print("PASS  universe membership mask enforced")


def test_correlation_matrix_wellformed():
    bundle = make_market(n_days=700, n_tickers=80, seed=27)
    data = _market_data(bundle)
    reg = default_registry()
    prepared = {
        n: tf.prepare(f, bundle["sectors"])
        for n, f in reg.compute_all(data).items()
    }
    corr = ev.signal_correlation(prepared)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T, equal_nan=True)
    off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
    assert np.nanmax(np.abs(off_diag)) <= 1.0001
    print("PASS  signal correlation matrix symmetric with unit diagonal")


def test_backtest_is_dollar_neutral_and_costs_reduce_return():
    bundle = make_market(n_days=900, n_tickers=100, signal_strength=0.05, seed=31)
    data = _market_data(bundle)
    prepared = tf.prepare(default_registry().signals[0](data), bundle["sectors"])

    res = run_backtest(prepared, bundle["close"], cost_bps=10.0)
    pos = res["positions"]
    active = pos[pos.abs().sum(axis=1) > 0]
    assert active.sum(axis=1).abs().max() < 1e-9, "book is not dollar neutral"
    assert res["stats"]["ann_return"] <= res["stats_gross"]["ann_return"] + 1e-12
    assert (res["costs"] >= 0).all()
    print(f"PASS  backtest dollar-neutral; costs cost "
          f"{(res['stats_gross']['ann_return'] - res['stats']['ann_return']):.4f}/yr")


def test_purged_splits_respect_embargo():
    dates = pd.bdate_range("2016-01-01", periods=1200)
    splits = purged_walk_forward_splits(dates, n_splits=4, embargo_days=21)
    assert len(splits) >= 3
    for train, test in splits:
        gap = (test[0] - train[-1]).days
        assert gap >= 21, f"embargo violated: {gap} days"
        assert train[-1] < test[0]
    print(f"PASS  {len(splits)} walk-forward splits, embargo respected")


def test_ic_decay_shape():
    bundle = make_market(n_days=900, n_tickers=80, signal_strength=0.05, seed=37)
    data = _market_data(bundle)
    prepared = tf.prepare(default_registry().signals[0](data), bundle["sectors"])
    decay = ev.ic_decay(prepared, bundle["close"], horizons=(1, 5, 21))
    assert list(decay.index) == [1, 5, 21]
    assert decay["n"].min() > 100
    print("PASS  IC decay computed across horizons")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
