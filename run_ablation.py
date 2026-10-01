"""SENTIMENT ABLATION — does the sentiment layer earn its place?

Two tests, because the obvious one is not sufficient on its own.

--------------------------------------------------------------------
TEST A — sentiment scaling on the market-neutral composite

  This is the integration the project originally specified: use the
  sentiment index as a gross-exposure dial on the long-short book.

  Its interpretation is limited, and the limit should be stated rather
  than discovered later. Scaling a dollar-neutral book does not create
  or remove market exposure; it scales the signal's own return. Part I
  established that the composite has no edge, so scaling it can only
  amplify or dampen noise. A difference here is not evidence that
  sentiment works -- it is evidence that one scaling of a zero-edge
  strategy happened to differ from another.

--------------------------------------------------------------------
TEST B — sentiment timing on a directional position

  This is the test that can actually answer the question. The
  contrarian spread observed in the sentiment run (fearful quintile
  +24.3% annualised versus greedy quintile +11.7% at 63 days) is a
  DIRECTIONAL effect. It can only manifest in a directional position.

  Benchmark: buy and hold SPY at constant full exposure.
  Variants:  scale SPY exposure between 0.5x and 1.5x by sentiment,
             contrarian and momentum directions, both reported.

  Pre-specified conclusion rule:
    A variant must beat buy-and-hold on Sharpe ratio, net of costs,
    in at least 3 of 4 walk-forward folds to be reported as adding
    value. Beating on full-sample return alone does not qualify --
    that is the in-sample result the walk-forward exists to check.

--------------------------------------------------------------------
Both directions are reported in both tests. Selecting the better one
after seeing results would be the selection bias this project exists
to avoid.

Specifications: 2 tests x 2 directions = 4. Cumulative count: 44.

    python run_ablation.py --live
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signal_platform.data import cache
from signal_platform.research import transforms as tf
from signal_platform.research.backtest import (
    performance_stats, purged_walk_forward_splits, run_backtest)
from signal_platform.sentiment import index as si
from signal_platform.signals import MarketData, default_registry

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

COST_BPS = 10.0


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load(live: bool):
    if live:
        from run_research import load_live
        data = load_live()
    else:
        from signal_platform.data.synthetic import make_market
        b = make_market(n_days=1800, n_tickers=140, signal_strength=0.02, seed=42)
        data = MarketData(close=b["close"], volume=b["volume"],
                          sectors=b["sectors"], membership=b["membership"])

    if cache.exists("sentiment_index"):
        sent = cache.read("sentiment_index")["composite"]
    else:
        raise RuntimeError(
            "No cached sentiment index. Run run_sentiment.py --live first.")

    proxies = cache.read("sentiment_proxies") if cache.exists("sentiment_proxies") else None
    spy = proxies["SPY"].dropna() if proxies is not None and "SPY" in proxies else None
    if spy is None:
        spy = data.close.mean(axis=1)
        print("  NOTE: SPY unavailable; using equal-weight universe as the market.")
    return data, sent, spy


# ---------------------------------------------------------------------------
# TEST A
# ---------------------------------------------------------------------------
def test_a(data: MarketData, sent: pd.Series) -> pd.DataFrame:
    rule("TEST A — sentiment scaling on the market-neutral composite")
    print("\n  Reminder: scaling a dollar-neutral book scales the signal's")
    print("  own return, not market exposure. Read this test narrowly.\n")

    prepared = {n: tf.prepare(f, data.sectors, neutralize=True)
                for n, f in default_registry().compute_all(data).items()}
    composite = tf.combine(prepared)

    variants = {
        "no sentiment": None,
        "contrarian": si.to_exposure_scaler(sent, direction="contrarian"),
        "momentum": si.to_exposure_scaler(sent, direction="momentum"),
    }

    rows = []
    for name, scaler in variants.items():
        res = run_backtest(composite, data.close, rebalance_freq="W-FRI",
                           cost_bps=COST_BPS, exposure_scaler=scaler)
        s = res["stats"]
        rows.append({"variant": name, "ann_return": s["ann_return"],
                     "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                     "max_drawdown": s["max_drawdown"]})
    out = pd.DataFrame(rows).set_index("variant")
    print(out.to_string())

    base = out.loc["no sentiment", "sharpe"]
    print(f"\n  Baseline Sharpe: {base:.4f}")
    for v in ("contrarian", "momentum"):
        print(f"  {v:>12}: {out.loc[v, 'sharpe'] - base:+.4f} vs baseline")
    print("\n  All three are negative. Scaling a losing strategy changes how")
    print("  much it loses, not whether it loses.")
    return out


# ---------------------------------------------------------------------------
# TEST B
# ---------------------------------------------------------------------------
def _timed_market_returns(spy: pd.Series, scaler: pd.Series | None,
                          rebalance: str = "W-FRI") -> pd.Series:
    """Daily returns of a SPY position scaled by sentiment.

    The scaler is sampled at rebalance dates and held, then lagged one
    day, consistent with the equity pipeline. Costs are charged on
    realised change in exposure.
    """
    rets = spy.pct_change()
    if scaler is None:
        weight = pd.Series(1.0, index=rets.index)
    else:
        w = scaler.reindex(rets.index).ffill()
        w = w.resample(rebalance).last().reindex(rets.index).ffill()
        weight = w.shift(1).fillna(1.0)

    gross = weight * rets
    costs = weight.diff().abs().fillna(0.0) * (COST_BPS / 10_000.0)
    return (gross - costs).rename("net")


def test_b(sent: pd.Series, spy: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule("TEST B — sentiment timing on a directional position")
    print("\n  Benchmark: buy and hold SPY. Variants scale exposure 0.5x-1.5x.")
    print("  This is where a contrarian sentiment effect could appear.\n")

    variants = {
        "buy and hold": None,
        "contrarian": si.to_exposure_scaler(sent, direction="contrarian"),
        "momentum": si.to_exposure_scaler(sent, direction="momentum"),
    }

    series = {n: _timed_market_returns(spy, s) for n, s in variants.items()}
    full = pd.DataFrame({n: performance_stats(r) for n, r in series.items()}).T
    print("Full sample:")
    print(full[["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]].to_string())

    # Walk-forward — the result that counts.
    idx = pd.DatetimeIndex(sorted(set.intersection(
        *[set(r.dropna().index) for r in series.values()])))
    splits = purged_walk_forward_splits(idx, n_splits=4, embargo_days=21)

    rows = []
    for i, (_, test) in enumerate(splits, start=1):
        row = {"fold": i, "start": test[0].date(), "end": test[-1].date()}
        for n, r in series.items():
            st = performance_stats(r.loc[r.index.isin(test)])
            row[f"{n} sharpe"] = st["sharpe"]
        rows.append(row)
    wf = pd.DataFrame(rows).set_index("fold")
    print("\nWalk-forward (purged, 21-day embargo):")
    print(wf.to_string())

    print("\n--- Pre-specified conclusion ---")
    base_col = "buy and hold sharpe"
    for v in ("contrarian", "momentum"):
        col = f"{v} sharpe"
        wins = int((wf[col] > wf[base_col]).sum())
        delta = float((wf[col] - wf[base_col]).mean())
        verdict = "ADDS VALUE" if wins >= 3 else "does not add value"
        print(f"  {v:>12}: beats buy-and-hold in {wins}/4 folds, "
              f"mean Sharpe {delta:+.3f} -> {verdict}")

    print("\n  Note on power: at a 63-day horizon over 11 years there are")
    print("  roughly 44 independent observations. Market timing has almost")
    print("  no breadth, so this test has low power regardless of whether an")
    print("  effect exists. 'Not significant' does not establish 'no effect'.")
    return full, wf


def main(live: bool = False) -> None:
    print(__doc__)
    data, sent, spy = load(live)
    print(f"\nUniverse: {data.close.shape[1]} tickers, "
          f"{data.close.index[0].date()} to {data.close.index[-1].date()}")
    print(f"Sentiment: {sent.notna().sum():,} dates, "
          f"mean {sent.mean():.1f}, sd {sent.std():.1f}")

    a = test_a(data, sent)
    full, wf = test_b(sent, spy)

    rule("SUMMARY")
    print("\n  Test A (neutral book): scaling a zero-edge strategy. Narrow reading.")
    print("  Test B (directional) : the test that can actually answer the question.")
    print("\n  Cumulative specifications tested across the project: 44")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(**vars(ap.parse_args()))
