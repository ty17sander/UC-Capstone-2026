"""The evaluation harness -- the research core of the platform.

This is deliberately separated from portfolio backtesting. A backtest
produces one noisy number that blends signal quality, portfolio
construction, and cost assumptions together; when it disappoints you
cannot tell which component failed. Signal evaluation isolates the first.

The metrics here answer four questions about any signal:

1. Does it predict?            -> information coefficient and its t-stat
2. Over what horizon?          -> IC decay across horizons
3. Is the relationship monotone? -> quantile spread
4. What will it cost to trade? -> turnover

And one question about the library as a whole:

5. Is this signal telling me anything the others are not? -> correlation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORWARD_HORIZONS, N_QUANTILES, TRADE_LAG, newey_west_lags
from ..data.prices import forward_returns


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------
def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """Pearson correlation computed independently for each row."""
    a, b = a.align(b, join="inner")
    valid = a.notna() & b.notna()
    a = a.where(valid)
    b = b.where(valid)

    n = valid.sum(axis=1)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)

    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))

    out = num / den.replace(0.0, np.nan)
    return out.where(n >= min_obs)


def information_coefficient(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    *,
    method: str = "spearman",
    min_obs: int = 20,
) -> pd.Series:
    """Cross-sectional correlation of signal against forward return, per date.

    Spearman (rank) is the default because it is robust to the fat tails
    that dominate equity returns -- a single 300% mover would otherwise
    drive a Pearson IC on its own.
    """
    if method == "spearman":
        s = signal.rank(axis=1)
        r = fwd_ret.rank(axis=1)
    elif method == "pearson":
        s, r = signal, fwd_ret
    else:
        raise ValueError(f"Unknown method: {method}")
    return _rowwise_corr(s, r, min_obs=min_obs).dropna()


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """Summarise an IC series with an autocorrelation-robust t-statistic.

    Overlapping forward returns make consecutive IC observations highly
    dependent. A naive t-stat on a 21-day-horizon IC series can be
    inflated several-fold. Newey-West HAC standard errors correct for
    this, and omitting the correction is one of the most common ways a
    backtest reports significance that is not there.
    """
    ic = ic.dropna()
    n = len(ic)
    if n < 10:
        return {"n": n, "mean_ic": np.nan, "ic_std": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "ic_ir": np.nan, "hit_rate": np.nan}

    mean, std = ic.mean(), ic.std(ddof=1)
    try:
        import statsmodels.api as sm
        model = sm.OLS(ic.values, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": max(newey_west_lags(horizon), 1)},
        )
        t_stat = float(model.tvalues[0])
        p_value = float(model.pvalues[0])
    except Exception:  # pragma: no cover - fallback if statsmodels absent
        t_stat = float(mean / (std / np.sqrt(n))) if std > 0 else np.nan
        p_value = np.nan

    return {
        "n": n,
        "mean_ic": float(mean),
        "ic_std": float(std),
        "t_stat": t_stat,
        "p_value": p_value,
        # IC information ratio: consistency of the signal, not its size.
        "ic_ir": float(mean / std) if std > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()),
    }


def ic_decay(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
    lag: int = TRADE_LAG,
    method: str = "spearman",
) -> pd.DataFrame:
    """IC across several horizons -- reveals the natural holding period.

    A signal whose IC peaks at 1 day and vanishes by 21 needs daily
    rebalancing and will be eaten by transaction costs. One that holds up
    at 21 days can be traded weekly and survive.
    """
    rows = []
    for h in horizons:
        fwd = forward_returns(close, horizon=h, lag=lag)
        ic = information_coefficient(signal, fwd, method=method)
        rows.append({"horizon": h, **ic_summary(ic, h)})
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Quantile analysis
# ---------------------------------------------------------------------------
def quantile_returns(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    horizon: int = 5,
    lag: int = TRADE_LAG,
    n_quantiles: int = N_QUANTILES,
) -> pd.DataFrame:
    """Mean forward return by signal quantile, per date.

    Monotonicity across buckets matters more than the top bucket's return.
    A signal where only the extreme decile works, with the middle
    unordered, is usually picking up an outlier effect rather than a
    pervasive one -- and it will not survive out of sample.
    """
    fwd = forward_returns(close, horizon=horizon, lag=lag)
    sig, fwd = signal.align(fwd, join="inner")

    def _bucket(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < n_quantiles * 2:
            return pd.Series(np.nan, index=row.index)
        try:
            q = pd.qcut(valid, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=row.index)
        return q.reindex(row.index)

    buckets = sig.apply(_bucket, axis=1)

    records = []
    for q in range(n_quantiles):
        mask = buckets == q
        masked = fwd.where(mask)
        records.append(masked.mean(axis=1).rename(f"Q{q + 1}"))

    out = pd.concat(records, axis=1)
    out["spread"] = out[f"Q{n_quantiles}"] - out["Q1"]
    return out.dropna(how="all")


def quantile_summary(qret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Annualised mean return per bucket, plus monotonicity diagnostics."""
    periods_per_year = 252.0 / horizon
    means = qret.mean() * periods_per_year
    stds = qret.std() * np.sqrt(periods_per_year)
    summary = pd.DataFrame({
        "ann_mean": means,
        "ann_vol": stds,
        "sharpe": means / stds.replace(0.0, np.nan),
    })
    bucket_means = means.drop("spread", errors="ignore")
    # Spearman correlation of bucket index against bucket mean return: 1.0
    # is perfectly monotone increasing.
    summary.attrs["monotonicity"] = float(
        pd.Series(range(len(bucket_means))).corr(
            pd.Series(bucket_means.values), method="spearman"
        )
    ) if len(bucket_means) > 2 else np.nan
    return summary


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------
def turnover(signal: pd.DataFrame, *, normalize: bool = True) -> pd.Series:
    """Per-period fraction of the book that would change.

    Computed on the prepared signal rather than on realised positions, so
    it is a property of the signal itself. Multiply by round-trip cost to
    get the annual drag a signal must overcome before it is worth trading.
    """
    sig = signal.copy()
    if normalize:
        abs_sum = sig.abs().sum(axis=1).replace(0.0, np.nan)
        sig = sig.div(abs_sum, axis=0)
    return (sig - sig.shift(1)).abs().sum(axis=1).rename("turnover")


def cost_drag(turnover_series: pd.Series, cost_bps: float, periods_per_year: float) -> float:
    """Annualised return drag implied by a signal's turnover."""
    return float(turnover_series.mean() * (cost_bps / 10_000.0) * periods_per_year)


# ---------------------------------------------------------------------------
# Cross-signal correlation
# ---------------------------------------------------------------------------
def signal_correlation(prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average pairwise cross-sectional correlation between signals.

    This is the most decision-relevant table in the platform. Information
    ratio scales with the square root of the number of *independent*
    bets, so a signal correlated 0.9 with an incumbent adds almost
    nothing regardless of how good its standalone IC looks. Correlation
    below roughly 0.3 is where a new signal starts earning its place.
    """
    names = sorted(prepared)
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        out.loc[a, a] = 1.0
        for b in names[i + 1:]:
            c = _rowwise_corr(prepared[a], prepared[b]).mean()
            out.loc[a, b] = out.loc[b, a] = c
    return out


def ic_correlation(ics: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of signals' IC *time series*.

    Distinct from score correlation and arguably more important: two
    signals can rank stocks differently yet still succeed and fail in the
    same market regimes, which means they diversify far less than their
    score correlation suggests.
    """
    return pd.DataFrame(ics).corr()


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------
def evaluate_signal(
    name: str,
    prepared: pd.DataFrame,
    close: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
    primary_horizon: int = 5,
    lag: int = TRADE_LAG,
    cost_bps: float = 10.0,
) -> dict:
    """Run the full battery on one signal."""
    decay = ic_decay(prepared, close, horizons=horizons, lag=lag)
    qret = quantile_returns(prepared, close, horizon=primary_horizon, lag=lag)
    qsum = quantile_summary(qret, primary_horizon)
    to = turnover(prepared)

    periods_per_year = 252.0 / primary_horizon
    drag = cost_drag(to, cost_bps, periods_per_year)
    spread_ann = float(qret["spread"].mean() * periods_per_year)

    return {
        "name": name,
        "ic_decay": decay,
        "quantile_returns": qret,
        "quantile_summary": qsum,
        "turnover": to,
        "headline": {
            "mean_ic": decay.loc[primary_horizon, "mean_ic"],
            "ic_t_stat": decay.loc[primary_horizon, "t_stat"],
            "ic_ir": decay.loc[primary_horizon, "ic_ir"],
            "monotonicity": qsum.attrs.get("monotonicity", np.nan),
            "ann_spread_gross": spread_ann,
            "ann_cost_drag": drag,
            "ann_spread_net": spread_ann - drag,
            "avg_turnover": float(to.mean()),
        },
    }


def evaluate_registry(
    prepared: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate every signal; return a league table plus full detail."""
    details = {n: evaluate_signal(n, f, close, **kwargs) for n, f in prepared.items()}
    table = pd.DataFrame([d["headline"] for d in details.values()],
                         index=list(details)).sort_values("ic_ir", ascending=False)
    return table, details
