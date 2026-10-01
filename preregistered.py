"""PRE-REGISTERED ANALYSIS -- three tests, specified before execution.

Committing to a fixed set of tests before running them is what separates
analysis from searching. Every result below is reported whether it
supports the hypothesis or not, and the trial count used in the
multiple-testing hurdle reflects everything run, not everything kept.

--------------------------------------------------------------------
TEST 1 -- Is LiquidityTurnover a real signal or a static size bet?

  H1: LiquidityTurnover's apparent significance (t = -2.31, perfect
      monotonicity) reflects a persistent large-vs-small positioning
      rather than a repeated, independent cross-sectional bet.

  Evidence that would support H1:
    - high rank autocorrelation at long lags
    - high name overlap in the long book one year apart
    - PnL largely explained by a mega-cap spread portfolio
    - effective breadth far below the nominal name count

  Pre-specified conclusion rule: if 1-year book overlap exceeds 70%
  AND the spanning regression alpha is insignificant, the signal is
  reported as a factor exposure, not a discovery.

--------------------------------------------------------------------
TEST 2 -- Does any signal work at a longer horizon?

  H2: The library is momentum-adjacent and momentum does not operate
      at 5 days. ICs should be stronger at 21 and 63 days.

  All 7 signals x 4 horizons = 28 cells. ALL are reported. The
  multiple-testing hurdle is recomputed with the full trial count.

--------------------------------------------------------------------
TEST 3 -- Does walk-forward sign determination fix the composite?

  H3: The composite loses because equal weighting includes wrong-signed
      signals. Determining signs from prior folds only should improve
      out-of-sample performance.

  This is a methodology correction, not a new hypothesis, so it adds
  no trials. Setting signs on the full sample would be p-hacking; signs
  here come only from data preceding each test fold.
--------------------------------------------------------------------

Run:  python preregistered.py --live
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signal_platform.data.prices import forward_returns
from signal_platform.research import evaluation as ev
from signal_platform.research import transforms as tf
from signal_platform.research.backtest import (
    build_positions,
    deflated_sharpe_note,
    performance_stats,
    purged_walk_forward_splits,
)
from signal_platform.signals import MarketData, default_registry

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

HORIZONS = (1, 5, 21, 63)
TARGET = "LiquidityTurnover_window63"


def load(live: bool) -> MarketData:
    if not live:
        from signal_platform.data.synthetic import make_market
        b = make_market(n_days=1800, n_tickers=150, signal_strength=0.02, seed=42)
        return MarketData(close=b["close"], volume=b["volume"],
                          sectors=b["sectors"], membership=b["membership"])
    from run_research import load_live
    return load_live()


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# TEST 1
# ---------------------------------------------------------------------------
def test_static_bet(data: MarketData, prepared: dict[str, pd.DataFrame],
                    target: str) -> dict:
    rule(f"TEST 1 -- Is {target} a static size bet?")

    if target not in prepared:
        print(f"  {target} not in library; skipping.")
        return {}

    sig = prepared[target]
    close = data.close

    # (a) Rank autocorrelation: how quickly does the ordering change?
    print("\n(a) Cross-sectional rank autocorrelation")
    print("    A genuine repeated bet re-ranks over time. A static")
    print("    position keeps the same order for years.\n")
    ranks = sig.rank(axis=1)
    ac_rows = []
    for lag in (1, 21, 63, 252):
        if lag >= len(ranks):
            continue
        a, b = ranks, ranks.shift(lag)
        valid = a.notna() & b.notna()
        am = a.where(valid).sub(a.where(valid).mean(axis=1), axis=0)
        bm = b.where(valid).sub(b.where(valid).mean(axis=1), axis=0)
        num = (am * bm).sum(axis=1)
        den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
        ac_rows.append({"lag_days": lag,
                        "rank_autocorr": float((num / den.replace(0, np.nan)).mean())})
    ac = pd.DataFrame(ac_rows).set_index("lag_days")
    print(ac.to_string())

    # (b) Book overlap one year apart -- the decisive, non-circular test.
    print("\n(b) Long-book name overlap across time")
    print("    If the same names are held a year later, breadth is")
    print("    an illusion regardless of which factor explains it.\n")
    pos = build_positions(sig.resample("W-FRI").last().dropna(how="all"))
    longs = {d: set(row[row > 0].index) for d, row in pos.iterrows()
             if (row > 0).any()}
    dates = sorted(longs)
    overlaps = {}
    for label, weeks in (("1 week", 1), ("13 weeks", 13), ("52 weeks", 52)):
        vals = []
        for i in range(len(dates) - weeks):
            a, b = longs[dates[i]], longs[dates[i + weeks]]
            if a and b:
                vals.append(len(a & b) / len(a))
        if vals:
            overlaps[label] = float(np.mean(vals))
    for k, v in overlaps.items():
        print(f"    {k:>10} later: {v:.1%} of long book unchanged")

    # (c) Spanning regression on a mega-cap proxy.
    print("\n(c) Spanning regression of signal PnL")
    print("    Caveat: the mega-cap proxy is built from dollar volume,")
    print("    the same input as the signal, so this is partly circular.")
    print("    Treat (b) as the stronger evidence.\n")

    daily = close.pct_change()
    held = pos.reindex(daily.index).ffill().shift(1).fillna(0.0)
    pnl = (held * daily).sum(axis=1)

    if data.volume is not None:
        dv = (close * data.volume).rolling(63, min_periods=20).mean()
        dv_rank = dv.rank(axis=1, pct=True)
        mega = daily.where(dv_rank > 0.9).mean(axis=1)
        small = daily.where(dv_rank < 0.5).mean(axis=1)
        megacap_spread = (mega - small).rename("megacap_spread")
    else:
        megacap_spread = pd.Series(0.0, index=daily.index, name="megacap_spread")

    market = daily.mean(axis=1).rename("market")
    X = pd.concat([market, megacap_spread], axis=1)
    df = pd.concat([pnl.rename("pnl"), X], axis=1).dropna()

    alpha_t = np.nan
    if len(df) > 100:
        try:
            import statsmodels.api as sm
            model = sm.OLS(df["pnl"], sm.add_constant(df[["market", "megacap_spread"]])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 21})
            out = pd.DataFrame({
                "coef": model.params, "t_stat": model.tvalues,
                "p_value": model.pvalues})
            out.index = ["alpha", "market_beta", "megacap_beta"]
            print(out.to_string())
            print(f"\n    R-squared: {model.rsquared:.4f}")
            alpha_t = float(model.tvalues.iloc[0])
        except Exception as exc:  # noqa: BLE001
            print(f"    regression failed: {exc}")

    # (d) Effective breadth.
    print("\n(d) Effective breadth")
    n_names = int(pos.abs().gt(0).sum(axis=1).mean())
    ann_overlap = overlaps.get("52 weeks", np.nan)
    if not np.isnan(ann_overlap) and ann_overlap < 1:
        eff_bets_per_year = (1 - ann_overlap) * n_names
    else:
        eff_bets_per_year = np.nan
    print(f"    nominal positions       : {n_names}")
    print(f"    independent bets / year : {eff_bets_per_year:.1f}")
    print("    (nominal count assumes each name is a fresh decision;")
    print("     persistent books make that assumption false)")

    # Pre-specified conclusion rule.
    print("\n--- Pre-specified conclusion ---")
    supports = (not np.isnan(ann_overlap) and ann_overlap > 0.70)
    alpha_insig = np.isnan(alpha_t) or abs(alpha_t) < 2.0
    if supports and alpha_insig:
        print("  H1 SUPPORTED: report as a factor exposure, not a discovery.")
    elif supports:
        print("  MIXED: book is persistent but alpha survives the spanning")
        print("  regression. Report both; do not claim a novel signal.")
    else:
        print("  H1 NOT SUPPORTED: the book turns over enough that breadth")
        print("  is real. The signal warrants further investigation.")

    return {"overlaps": overlaps, "alpha_t": alpha_t,
            "eff_bets": eff_bets_per_year}


# ---------------------------------------------------------------------------
# TEST 2
# ---------------------------------------------------------------------------
def test_horizons(data: MarketData, prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rule("TEST 2 -- Full horizon sweep (all cells reported)")

    rows = []
    for name, sig in prepared.items():
        decay = ev.ic_decay(sig, data.close, horizons=HORIZONS)
        for h in HORIZONS:
            if h not in decay.index:
                continue
            rows.append({
                "signal": name[:32], "horizon": h,
                "mean_ic": decay.loc[h, "mean_ic"],
                "t_stat": decay.loc[h, "t_stat"],
                "p_value": decay.loc[h, "p_value"],
                "ic_ir": decay.loc[h, "ic_ir"],
            })
    grid = pd.DataFrame(rows)

    print("\nIC t-statistic by signal and horizon:\n")
    pivot = grid.pivot(index="signal", columns="horizon", values="t_stat")
    print(pivot.to_string())

    print("\nMean IC by signal and horizon:\n")
    print(grid.pivot(index="signal", columns="horizon", values="mean_ic").to_string())

    n_trials = len(grid)
    print(f"\nTotal cells tested: {n_trials}")

    # Bonferroni is conservative but transparent; Harvey/Liu/Zhu argue for
    # a t-hurdle near 3.0 in this literature for similar reasons.
    hurdle = 3.0
    survivors = grid[grid["t_stat"].abs() > hurdle]
    print(f"Cells with |t| > {hurdle}: {len(survivors)}")
    if len(survivors):
        print(survivors.to_string(index=False))
    else:
        print("  None. No signal/horizon combination clears the hurdle")
        print("  appropriate for the number of specifications tested.")

    best = grid.loc[grid["t_stat"].abs().idxmax()]
    print(f"\nStrongest cell (still reported even if it fails the hurdle):")
    print(f"  {best['signal']} at {int(best['horizon'])}d: "
          f"IC={best['mean_ic']:.4f}, t={best['t_stat']:.2f}, "
          f"p={best['p_value']:.4f}")

    return grid


# ---------------------------------------------------------------------------
# TEST 3
# ---------------------------------------------------------------------------
def test_walkforward_signs(data: MarketData,
                           prepared: dict[str, pd.DataFrame],
                           horizon: int = 5) -> pd.DataFrame:
    rule("TEST 3 -- Walk-forward sign determination")
    print("\n  Signs are fixed using ONLY data before each test fold.")
    print("  Compared against naive equal weighting on the same folds.\n")

    close = data.close
    idx = sorted(set.intersection(*[set(f.index) for f in prepared.values()]))
    idx = pd.DatetimeIndex(idx)
    splits = purged_walk_forward_splits(idx, n_splits=4, embargo_days=21)

    daily = close.pct_change()
    rows = []

    for i, (train, test) in enumerate(splits, start=1):
        signs = {}
        for name, sig in prepared.items():
            fwd = forward_returns(close, horizon=horizon, lag=1)
            tr = sig.loc[sig.index.isin(train)]
            ic = ev.information_coefficient(tr, fwd.reindex(tr.index))
            m = ic.mean()
            signs[name] = 0.0 if (pd.isna(m) or abs(m) < 1e-6) else float(np.sign(m))

        def _run(weights: dict[str, float]) -> dict:
            comp = tf.combine(prepared, weights=weights)
            comp_test = comp.loc[comp.index.isin(test)]
            if comp_test.empty:
                return {}
            pos = build_positions(comp_test.resample("W-FRI").last().dropna(how="all"))
            held = pos.reindex(daily.index).ffill().shift(1).fillna(0.0)
            held = held.loc[held.index.isin(test)]
            gross = (held * daily.reindex(held.index)).sum(axis=1)
            costs = held.diff().abs().sum(axis=1).fillna(0.0) * (10.0 / 10_000.0)
            return performance_stats(gross - costs)

        signed = _run(signs) or {}
        naive = _run({n: 1.0 for n in prepared}) or {}

        rows.append({
            "fold": i,
            "test_start": test[0].date(), "test_end": test[-1].date(),
            "naive_sharpe": naive.get("sharpe", np.nan),
            "signed_sharpe": signed.get("sharpe", np.nan),
            "naive_return": naive.get("ann_return", np.nan),
            "signed_return": signed.get("ann_return", np.nan),
            "n_positive_signs": int(sum(1 for v in signs.values() if v > 0)),
        })

    res = pd.DataFrame(rows).set_index("fold")
    print(res.to_string())

    imp = (res["signed_sharpe"] - res["naive_sharpe"]).dropna()
    print(f"\n  Mean Sharpe improvement: {imp.mean():+.4f}")
    print(f"  Folds improved: {int((imp > 0).sum())} of {len(imp)}")
    print("\n--- Pre-specified conclusion ---")
    if len(imp) and imp.mean() > 0 and (imp > 0).sum() >= len(imp) * 0.75:
        print("  H3 SUPPORTED: sign determination helps out of sample.")
    elif len(imp) and imp.mean() > 0:
        print("  WEAK SUPPORT: mean improvement positive but inconsistent")
        print("  across folds. Do not claim this as a robust improvement.")
    else:
        print("  H3 NOT SUPPORTED: sign determination does not help. This")
        print("  is evidence the signals carry no stable directional edge.")
    return res


# ---------------------------------------------------------------------------
def main(live: bool = False) -> None:
    print(__doc__)
    data = load(live)
    print(f"\nUniverse: {data.close.shape[1]} tickers, {data.close.shape[0]} days, "
          f"{data.close.index[0].date()} to {data.close.index[-1].date()}")

    registry = default_registry()
    prepared = {n: tf.prepare(f, data.sectors, neutralize=True)
                for n, f in registry.compute_all(data).items()}

    t1 = test_static_bet(data, prepared, TARGET)
    grid = test_horizons(data, prepared)
    t3 = test_walkforward_signs(data, prepared)

    rule("MULTIPLE-TESTING ACCOUNTING")
    n_trials = len(grid)
    print(f"\n  Specifications tested in this run : {n_trials}")
    print(f"  Plus prior 5-day league table     : 7")
    print(f"  Cumulative trial count            : {n_trials + 7}")
    best_t = grid["t_stat"].abs().max()
    print(f"\n  Largest |t| observed              : {best_t:.2f}")
    note = deflated_sharpe_note(n_trials + 7, observed_sharpe=0.0,
                                n_obs=len(data.close))
    print(f"  Expected max Sharpe under null    : "
          f"{note['expected_max_sharpe_null']:.4f}")
    print("\n  Report the cumulative count in the writeup. A t-statistic")
    print("  that would impress at one trial is unremarkable at 35.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(**vars(ap.parse_args()))
