"""Build and evaluate the market sentiment composite index.

    python run_sentiment.py            # synthetic, offline
    python run_sentiment.py --live     # real data

PRE-REGISTERED QUESTION, fixed before execution:

  Is the composite index PREDICTIVE of forward market returns, or merely
  COINCIDENT with recent ones?

  Most published sentiment measures are coincident: they report that the
  market has fallen, which the reader already knows. The test is whether
  forward returns vary systematically across sentiment levels once the
  index is constructed causally.

  Conclusion rule, fixed in advance:
    - |t| > 3.0 on the forward-return slope at any horizon
      -> report as predictive
    - correlation with trailing returns exceeding correlation with
      forward returns by more than 3x
      -> report as coincident
    - otherwise -> report as inconclusive

  Specifications tested here: 3 horizons x 1 composite = 3, plus 2
  exposure directions in the ablation = 5. Added to the running total
  of 35 from Part I, the cumulative count becomes 40.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signal_platform.sentiment import index as si

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load(live: bool):
    if not live:
        rng = np.random.default_rng(3)
        from signal_platform.data.synthetic import make_market
        b = make_market(n_days=1800, n_tickers=140, seed=3)
        idx = b["close"].index
        proxies = pd.DataFrame({
            "^VIX": 15 + 8 * np.abs(rng.normal(0, 1, len(idx))),
            "SPY": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, len(idx)))),
            "TLT": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.007, len(idx)))),
            "HYG": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, len(idx)))),
            "LQD": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, len(idx)))),
        }, index=idx)
        return proxies, b["close"], b["membership"]

    from run_research import load_live
    data = load_live()
    start = data.close.index[0].strftime("%Y-%m-%d")
    end = data.close.index[-1].strftime("%Y-%m-%d")
    print("\nFetching sentiment proxy series:")
    proxies = si.fetch_proxies(start, end)
    return proxies, data.close, data.membership


def main(live: bool = False) -> None:
    print(__doc__)
    proxies, close, membership = load(live)

    rule("1. COMPONENT CONSTRUCTION")
    print()
    frame = si.build_index(proxies, close, membership)

    print("\nComponent summary (0 = extreme fear, 100 = extreme greed):")
    comps = [c for c in frame.columns if c not in ("composite", "n_components")]
    print(frame[comps + ["composite"]].describe().loc[
        ["count", "mean", "std", "min", "max"]].to_string())

    print("\nComponent correlation:")
    print(frame[comps].corr().to_string())
    print("\n  Components correlated above 0.8 are largely redundant; the")
    print("  index gains little from carrying both.")

    latest = frame["composite"].dropna()
    if len(latest):
        v = latest.iloc[-1]
        print(f"\nMost recent reading: {v:.1f} -- {si.label(v)} "
              f"({latest.index[-1].date()})")

    rule("2. TIME IN EACH SENTIMENT REGIME")
    labels = frame["composite"].dropna().map(si.label)
    dist = labels.value_counts(normalize=True).reindex(
        ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]).dropna()
    for k, pct in dist.items():
        bar = "#" * int(pct * 50)
        print(f"  {k:<15} {pct:>6.1%}  {bar}")

    rule("3. PRE-REGISTERED TEST: predictive or coincident?")
    market = proxies["SPY"] if "SPY" in proxies.columns else close.mean(axis=1)

    pred = si.predictive_test(frame["composite"], market)
    print("\nForward-return regression (one-day execution lag):\n")
    print(pred[["n", "corr", "slope_t", "p_value", "r_squared",
                "low_bucket_ann", "high_bucket_ann", "spread_ann"]].to_string())
    print("\n  low_bucket_ann  = annualised forward return when sentiment is most fearful")
    print("  high_bucket_ann = the same when sentiment is most greedy")
    print("  A contrarian effect would show low > high (positive spread).")

    coin = si.coincident_test(frame["composite"], market)
    print(f"\nCoincident check:")
    print(f"  correlation with TRAILING 21-day return : {coin['corr_with_trailing']:+.4f}")
    print(f"  correlation with FORWARD  21-day return : {coin['corr_with_forward']:+.4f}")
    ratio = (abs(coin["corr_with_trailing"]) /
             max(abs(coin["corr_with_forward"]), 1e-9))
    print(f"  ratio                                   : {ratio:.1f}x")

    print("\n--- Pre-specified conclusion ---")
    max_t = pred["slope_t"].abs().max() if len(pred) else np.nan
    best_h = int(pred["slope_t"].abs().idxmax()) if len(pred) else 63

    # The theoretical null does not apply to this regressor. Calibration on
    # synthetic data with no relationship showed |t| > 2 occurring 8-20% of
    # the time rather than 5%, because the index is persistent and partly
    # derived from the same price series it is asked to predict (Stambaugh,
    # 1999). A bootstrap null is therefore mandatory, not supplementary.
    print("\n  Running bootstrap null calibration (the theoretical null")
    print("  is known to be anti-conservative for this regressor)...")
    boot = si.null_calibration(frame["composite"], market,
                               horizon=best_h, n_sims=300, seed=1)

    if "error" in boot:
        print(f"  calibration failed: {boot['error']}")
        emp_p, boot_sig = np.nan, False
    else:
        print(f"\n    observed t at {best_h}d      : {boot['observed_t']:+.3f}")
        print(f"    null sd of t            : {boot['null_std_t']:.3f} "
              f"(theory says 1.000)")
        print(f"    null 95th pct of |t|    : {boot['null_p95']:.3f} "
              f"(theory says 1.96)")
        print(f"    null 99th pct of |t|    : {boot['null_p99']:.3f} "
              f"(theory says 2.58)")
        print(f"    empirical p-value       : {boot['empirical_p_value']:.4f}")
        emp_p, boot_sig = boot["empirical_p_value"], boot["significant"]

    if not np.isnan(max_t) and max_t > 3.0 and boot_sig:
        print(f"\n  PREDICTIVE: slope t = {max_t:.2f} clears the hurdle AND")
        print(f"  survives bootstrap calibration (empirical p = {emp_p:.4f}).")
    elif not np.isnan(max_t) and max_t > 3.0:
        print(f"\n  NOT PREDICTIVE despite t = {max_t:.2f}: the statistic fails")
        print("  bootstrap calibration. Under the empirical null this value is")
        print("  unremarkable. Reporting it as a finding would be a false positive.")
    elif ratio > 3.0:
        print(f"\n  COINCIDENT: the index tracks the recent past ({ratio:.1f}x")
        print("  stronger than it tracks the future). It describes where the")
        print("  market has been, not where it is going. This is the expected")
        print("  result for sentiment measures and is a legitimate finding.")
    else:
        print(f"\n  INCONCLUSIVE: max slope t = {max_t:.2f}, trailing/forward")
        print(f"  ratio = {ratio:.1f}x. Neither threshold is met.")

    rule("4. EXPOSURE SCALER (the hook into the trading layer)")
    for direction in ("contrarian", "momentum"):
        sc = si.to_exposure_scaler(frame["composite"], direction=direction)
        print(f"\n  {direction}: range {sc.min():.2f} to {sc.max():.2f}, "
              f"mean {sc.mean():.2f}")
    print("\n  Both directions must be reported. Selecting whichever performs")
    print("  better after seeing results is the selection bias this project")
    print("  exists to avoid; the ablation in Part II tests both.")

    from signal_platform.data import cache
    cache.write("sentiment_index", frame,
                meta={"components": len(comps), "live": bool(live)})
    print("\n  Index cached as 'sentiment_index'.")

    rule("MULTIPLE-TESTING ACCOUNTING")
    print("\n  Specifications in this run   : 5 (3 horizons + 2 directions)")
    print("  Carried forward from Part I  : 35")
    print("  Cumulative                   : 40")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(**vars(ap.parse_args()))
