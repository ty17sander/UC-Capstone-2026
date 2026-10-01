"""Diagnostics -- run before believing any null result.

A flat league table has two very different explanations:
  (a) the signals genuinely have no edge in this universe and period, or
  (b) something upstream is destroying the signal before it is measured.

These checks separate the two. Run:  python diagnose.py --live
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signal_platform.research import evaluation as ev
from signal_platform.research import transforms as tf
from signal_platform.signals import MarketData, default_registry

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def load(live: bool):
    if not live:
        from signal_platform.data.synthetic import make_market
        b = make_market(n_days=1800, n_tickers=150, signal_strength=0.02, seed=42)
        return MarketData(close=b["close"], volume=b["volume"],
                          sectors=b["sectors"], membership=b["membership"])
    from run_research import load_live
    return load_live()


def main(live: bool = False) -> None:
    data = load(live)
    close = data.close

    print("=" * 78)
    print("1. UNIVERSE AND COVERAGE")
    print("=" * 78)
    print(f"  tickers in price matrix : {close.shape[1]}")
    print(f"  trading days            : {close.shape[0]}")

    all_nan = close.columns[close.isna().all()].tolist()
    print(f"  all-NaN columns         : {len(all_nan)}  {all_nan[:10]}")

    per_day = close.notna().sum(axis=1)
    print(f"  names with price/day    : min={per_day.min()}, "
          f"median={int(per_day.median())}, max={per_day.max()}")

    if data.membership is not None:
        mem = data.membership.reindex(index=close.index, columns=close.columns).fillna(False)
        both = (mem & close.notna()).sum(axis=1)
        print(f"  in-universe AND priced  : min={both.min()}, "
              f"median={int(both.median())}, max={both.max()}")

    print("\n" + "=" * 78)
    print("2. SECTOR MAPPING  <-- the suspected problem")
    print("=" * 78)
    sectors = data.sectors
    if sectors is None or len(sectors) == 0:
        print("  NO SECTOR MAP AT ALL -- neutralization is a no-op.")
    else:
        mapped = sectors.reindex(close.columns)
        n_mapped = mapped.notna().sum()
        print(f"  tickers with a sector   : {n_mapped} / {close.shape[1]} "
              f"({n_mapped / close.shape[1]:.0%})")
        print(f"  UNMAPPED (one bucket)   : {close.shape[1] - n_mapped}")
        print("\n  bucket sizes as neutralization sees them:")
        filled = mapped.fillna("_UNMAPPED")
        print(filled.value_counts().to_string())
        print("\n  If _UNMAPPED is one of the largest buckets, sector")
        print("  neutralization is demeaning unrelated delisted companies")
        print("  against each other. That is noise injection, not control.")

    print("\n" + "=" * 78)
    print("3. IC WITH vs WITHOUT SECTOR NEUTRALIZATION")
    print("=" * 78)
    print("  If neutralized ICs collapse while raw ICs survive, the")
    print("  neutralization step is the problem -- not the signals.\n")

    registry = default_registry()
    raw_scores = registry.compute_all(data)

    rows = []
    for name, raw in raw_scores.items():
        neut = tf.prepare(raw, sectors, neutralize=True)
        plain = tf.prepare(raw, sectors, neutralize=False)

        d_neut = ev.ic_decay(neut, close, horizons=(5,))
        d_plain = ev.ic_decay(plain, close, horizons=(5,))

        rows.append({
            "signal": name[:34],
            "ic_neutral": d_neut.loc[5, "mean_ic"],
            "t_neutral": d_neut.loc[5, "t_stat"],
            "ic_raw": d_plain.loc[5, "mean_ic"],
            "t_raw": d_plain.loc[5, "t_stat"],
        })
    comp = pd.DataFrame(rows).set_index("signal")
    comp["ic_lost_pct"] = (
        1 - comp["ic_neutral"].abs() / comp["ic_raw"].abs().replace(0, np.nan)
    ) * 100
    print(comp.to_string())

    print("\n" + "=" * 78)
    print("4. CROSS-SECTION SIZE PER DATE (after all masking)")
    print("=" * 78)
    prepared = tf.prepare(raw_scores[list(raw_scores)[0]], sectors, neutralize=True)
    n_scored = prepared.notna().sum(axis=1)
    print(f"  names scored per date   : min={n_scored.min()}, "
          f"median={int(n_scored.median())}, max={n_scored.max()}")
    print(f"  dates with <20 names    : {(n_scored < 20).sum()} "
          f"(IC is skipped on these)")

    print("\n" + "=" * 78)
    print("5. RETURN SANITY")
    print("=" * 78)
    rets = close.pct_change()
    print(f"  median daily return     : {rets.stack().median():.6f}")
    print(f"  daily vol (pooled)      : {rets.stack().std():.4f}")
    print(f"  |return| > 50% count    : {(rets.abs() > 0.5).sum().sum()}")
    print(f"  |return| > 100% count   : {(rets.abs() > 1.0).sum().sum()}")
    worst = rets.abs().max().sort_values(ascending=False).head(8)
    print("\n  largest single-day moves (check for unadjusted splits):")
    print(worst.to_string())

    ew = rets.mean(axis=1)
    ann = (1 + ew.fillna(0)).prod() ** (252 / len(ew)) - 1
    print(f"\n  equal-weight universe return, annualised: {ann:.2%}")
    print("  (Should be roughly in line with the market over the period.")
    print("   Far below it suggests dead/delisted names dragging the mean.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(**vars(ap.parse_args()))
