"""Simplified long-short market-neutral backtest.

Deliberately simple: dollar-neutral, equal-weight within the top and
bottom buckets, flat per-side cost. No optimiser, no leverage targeting,
no borrow-cost model. That simplicity is a feature for a capstone -- every
assumption is inspectable and defensible, and the headline number is not
the research finding anyway. The evaluation harness is.

Two things here are not optional and are frequently omitted elsewhere:
costs are charged on realised turnover, and the walk-forward split is
purged so that training and test windows cannot overlap through the
forward-return horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COST_BPS_PER_SIDE, TRADE_LAG


def build_positions(
    signal: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Dollar-neutral weights: long the top slice, short the bottom.

    Longs sum to +gross/2 and shorts to -gross/2, so the book nets to zero
    by construction. That is *dollar* neutrality, not beta neutrality --
    worth stating plainly, since a dollar-neutral book can still carry
    meaningful net beta if the long leg is higher-beta than the short.
    """
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date, row in signal.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 20:
            continue
        n_long = max(int(np.floor(n * top_frac)), 1)
        n_short = max(int(np.floor(n * bottom_frac)), 1)

        ordered = valid.sort_values(ascending=False)
        longs = ordered.index[:n_long]
        shorts = ordered.index[-n_short:]

        out.loc[date, longs] = (gross_exposure / 2.0) / n_long
        out.loc[date, shorts] = -(gross_exposure / 2.0) / n_short

    return out


def run_backtest(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    rebalance_freq: str = "W-FRI",
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    cost_bps: float = COST_BPS_PER_SIDE,
    lag: int = TRADE_LAG,
    exposure_scaler: pd.Series | None = None,
) -> dict:
    """Run the strategy and return equity curve plus performance stats.

    ``exposure_scaler`` is the hook for the sentiment layer: a series
    indexed by date, values typically in [0, 1.5], scaling gross exposure
    up or down with the market regime. One parameter, easy to defend --
    and easy to ablate, which is how you show it actually contributed.
    """
    daily_ret = close.pct_change()

    # Signal observed at period end -> traded with a lag -> held forward.
    sig_rebal = signal.resample(rebalance_freq).last().dropna(how="all")
    positions = build_positions(sig_rebal, top_frac=top_frac, bottom_frac=bottom_frac)

    if exposure_scaler is not None:
        scaler = exposure_scaler.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.mul(scaler, axis=0)

    # Hold each rebalance's weights until the next one, then apply the
    # execution lag. Both shifts are essential: without them the strategy
    # earns the return of the day it decided to trade on.
    held = positions.reindex(daily_ret.index).ffill().shift(lag).fillna(0.0)

    gross_pnl = (held * daily_ret).sum(axis=1)

    traded = held.diff().abs().sum(axis=1).fillna(0.0)
    costs = traded * (cost_bps / 10_000.0)
    net_pnl = gross_pnl - costs

    equity = (1.0 + net_pnl.fillna(0.0)).cumprod()

    return {
        "equity_curve": equity,
        "returns_net": net_pnl,
        "returns_gross": gross_pnl,
        "costs": costs,
        "positions": held,
        "stats": performance_stats(net_pnl),
        "stats_gross": performance_stats(gross_pnl),
    }


def performance_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Standard performance summary."""
    r = returns.dropna()
    if len(r) < 20:
        return {k: np.nan for k in
                ("ann_return", "ann_vol", "sharpe", "max_drawdown",
                 "calmar", "hit_rate", "skew", "n_periods")}

    ann_ret = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    equity = (1.0 + r).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": dd,
        "calmar": ann_ret / abs(dd) if dd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "n_periods": int(len(r)),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------
def purged_walk_forward_splits(
    dates: pd.DatetimeIndex,
    *,
    n_splits: int = 4,
    embargo_days: int = 21,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window splits with a gap between train and test.

    The embargo is the part people skip. If a signal uses a 21-day forward
    return, the last few training observations overlap the start of the
    test window, and information leaks across the boundary. Dropping an
    embargo period at least as long as the forward horizon closes that
    leak. Without it, walk-forward results are optimistic in a way that is
    invisible in the output.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    n = len(dates)
    if n < (n_splits + 1) * (embargo_days + 20):
        raise ValueError("Not enough history for the requested splits.")

    fold = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_start = train_end + embargo_days
        test_end = min(train_end + fold, n)
        if test_start >= test_end:
            continue
        splits.append((dates[:train_end], dates[test_start:test_end]))
    return splits


def walk_forward_report(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n_splits: int = 4,
    embargo_days: int = 21,
    **bt_kwargs,
) -> pd.DataFrame:
    """Out-of-sample performance for each fold.

    Consistency across folds is the finding worth reporting. One
    spectacular fold and three flat ones is a signal that worked in one
    regime, which is a very different claim from a signal that works.
    """
    splits = purged_walk_forward_splits(
        signal.index, n_splits=n_splits, embargo_days=embargo_days
    )
    rows = []
    for i, (train, test) in enumerate(splits, start=1):
        res = run_backtest(signal.loc[test], close.loc[close.index >= test[0]], **bt_kwargs)
        rows.append({
            "fold": i,
            "train_end": train[-1].date(),
            "test_start": test[0].date(),
            "test_end": test[-1].date(),
            **res["stats"],
        })
    return pd.DataFrame(rows).set_index("fold")


def deflated_sharpe_note(n_trials: int, observed_sharpe: float, n_obs: int) -> dict:
    """Rough haircut for having searched over many strategy variants.

    If you test 40 configurations and report the best, the winner's Sharpe
    is partly a selection artefact. This applies an approximate
    multiple-testing adjustment: the expected maximum Sharpe under the
    null of zero skill grows with the number of trials, and your observed
    value must clear that bar rather than merely clearing zero.

    This is an approximation of the Bailey and Lopez de Prado deflated
    Sharpe ratio, not a faithful implementation -- verify the reference
    before citing it, and state in the writeup that it is indicative.
    """
    from scipy.stats import norm

    if n_trials < 1 or n_obs < 20:
        return {"expected_max_sharpe_null": np.nan, "clears_hurdle": None}

    euler = 0.5772156649
    e = max(n_trials, 2)
    # Expected maximum of n_trials draws from a standard normal.
    z = (1 - euler) * norm.ppf(1 - 1.0 / e) + euler * norm.ppf(1 - 1.0 / (e * np.e))
    expected_max = float(z / np.sqrt(n_obs) * np.sqrt(252))

    return {
        "n_trials": n_trials,
        "observed_sharpe": observed_sharpe,
        "expected_max_sharpe_null": expected_max,
        "clears_hurdle": bool(observed_sharpe > expected_max),
    }
