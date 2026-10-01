"""End-to-end research run.

Defaults to synthetic data so it runs anywhere with no network. Pass
--live to ingest real prices via yfinance instead.

    python run_research.py                 # synthetic, offline
    python run_research.py --live          # real S&P 500 data
"""
from __future__ import annotations

import argparse

import pandas as pd

from signal_platform.data.synthetic import make_market
from signal_platform.research import evaluation as ev
from signal_platform.research import transforms as tf
from signal_platform.research.backtest import (
    deflated_sharpe_note,
    run_backtest,
    walk_forward_report,
)
from signal_platform.signals import MarketData, default_registry

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def load_synthetic():
    b = make_market(n_days=1800, n_tickers=150, signal_strength=0.02, seed=42)
    return MarketData(
        close=b["close"], volume=b["volume"],
        sectors=b["sectors"], membership=b["membership"],
    )


def load_live(start: str = "2015-01-01", end: str | None = None):
    from signal_platform.data import prices, universe
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start, end)

    membership = universe.build_universe(dates)
    tickers = list(membership.columns)
    data = prices.ingest(tickers, start, end)

    print("\nCleaning price data:")
    close, volume, report = prices.clean_prices(data["close"], data.get("volume"))

    sectors = universe.sector_map()
    mapped = sectors.reindex(close.columns).notna().sum()
    print(f"  sector coverage: {mapped}/{close.shape[1]} "
          f"({mapped / max(close.shape[1], 1):.0%})")
    if mapped / max(close.shape[1], 1) < 0.9:
        print("  WARNING: sector coverage below 90%. Unmapped names are")
        print("  pooled into one bucket, which makes neutralization harmful.")
        print("  Run: universe.rebuild_sector_map_from_snapshots()")

    membership = membership.reindex(columns=close.columns).fillna(False)
    membership = universe.apply_liquidity_filter(
        membership, close, volume,
        min_dollar_volume=5_000_000.0, min_price=5.0,
    )
    return MarketData(
        close=close, volume=volume, sectors=sectors, membership=membership,
    )


def main(live: bool = False) -> None:
    data = load_live() if live else load_synthetic()
    print(f"Universe: {data.close.shape[1]} tickers, "
          f"{data.close.shape[0]} days, "
          f"{data.close.index[0].date()} to {data.close.index[-1].date()}\n")

    # Cross-sectional statistics are meaningless on a handful of names:
    # IC requires at least 20 per date and quantile bucketing needs at
    # least 10. Failing loudly here beats emitting a full report of NaNs
    # that looks like a modelling result rather than a data problem.
    if data.close.shape[1] < 30:
        raise SystemExit(
            f"Only {data.close.shape[1]} tickers available -- too few for "
            "cross-sectional research (need 30+, ideally 300+).\n"
            "Most likely a stale price cache from a smaller run. Fix with:\n"
            "  from signal_platform.data import cache\n"
            "  import shutil, signal_platform.config as cfg\n"
            "  shutil.rmtree(cfg.CURATED_DIR)\n"
            "or rerun with prices.ingest(..., refresh=True)."
        )

    registry = default_registry()
    raw = registry.compute_all(data)
    prepared = {n: tf.prepare(f, data.sectors, neutralize=True)
                for n, f in raw.items()}

    print("=" * 78)
    print("SIGNAL LEAGUE TABLE  (5-day horizon, sector-neutral, 10bps/side)")
    print("=" * 78)
    table, details = ev.evaluate_registry(
        prepared, data.close, primary_horizon=5, cost_bps=10.0
    )
    print(table[["mean_ic", "ic_t_stat", "ic_ir", "monotonicity",
                 "ann_spread_gross", "ann_cost_drag", "ann_spread_net",
                 "avg_turnover"]].to_string())

    print("\n" + "=" * 78)
    print("SIGNAL CORRELATION  (avg cross-sectional; <0.3 means it adds breadth)")
    print("=" * 78)
    corr = ev.signal_correlation(prepared)
    short = {n: n.replace("_", "")[:14] for n in corr.columns}
    print(corr.rename(index=short, columns=short).to_string())

    print("\n" + "=" * 78)
    print("IC DECAY  (best signal by IC-IR)")
    print("=" * 78)
    best = table.index[0]
    print(f"Signal: {best}")
    print(details[best]["ic_decay"][
        ["n", "mean_ic", "t_stat", "p_value", "ic_ir", "hit_rate"]].to_string())

    print("\n" + "=" * 78)
    print("COMPOSITE BACKTEST  (equal-weight combination, long-short neutral)")
    print("=" * 78)
    composite = tf.combine(prepared)
    res = run_backtest(composite, data.close, rebalance_freq="W-FRI", cost_bps=10.0)
    stats = pd.DataFrame({
        "gross": res["stats_gross"], "net": res["stats"]
    })
    print(stats.to_string())

    print("\n" + "=" * 78)
    print("WALK-FORWARD  (purged, 21-day embargo) -- consistency is the finding")
    print("=" * 78)
    wf = walk_forward_report(composite, data.close, n_splits=4, embargo_days=21)
    print(wf[["test_start", "test_end", "ann_return", "ann_vol",
              "sharpe", "max_drawdown"]].to_string())

    print("\n" + "=" * 78)
    print("MULTIPLE-TESTING HAIRCUT")
    print("=" * 78)
    note = deflated_sharpe_note(
        n_trials=len(prepared),
        observed_sharpe=res["stats"]["sharpe"],
        n_obs=res["stats"]["n_periods"],
    )
    for k, v in note.items():
        print(f"  {k:>28}: {v}")
    print("\n  Interpretation: an observed Sharpe below the null hurdle is")
    print("  indistinguishable from having searched hard enough to find one.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use yfinance data")
    main(**vars(ap.parse_args()))
