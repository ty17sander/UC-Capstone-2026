"""Cross-sectional transforms applied before signals are compared or combined.

Order matters and is fixed: winsorize -> sector-neutralize -> z-score.

Winsorizing first stops a single bad print from dominating the sector mean.
Neutralizing before the final z-score means the output has unit dispersion
*after* the sector effect is removed, which is what the combiner expects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import WINSOR_LIMITS


def winsorize(df: pd.DataFrame, limits: tuple[float, float] = WINSOR_LIMITS) -> pd.DataFrame:
    """Clip each row to its own cross-sectional quantiles."""
    lo = df.quantile(limits[0], axis=1)
    hi = df.quantile(limits[1], axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise standardisation across the cross-section."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map each row to roughly N(0,1) via ranks.

    More robust than z-scoring when the raw signal is badly skewed, at the
    cost of discarding magnitude information.
    """
    ranks = df.rank(axis=1, pct=True)
    # Nudge off the open interval boundary before the normal quantile map.
    eps = 1e-6
    return pd.DataFrame(
        _norm_ppf(ranks.clip(eps, 1 - eps).to_numpy()),
        index=df.index, columns=df.columns,
    ).where(df.notna())


def _norm_ppf(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    return norm.ppf(x)


def sector_neutralize(df: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Demean each row within sector.

    This is the step that makes "market neutral" mean something. A raw
    momentum rank in 2020 would have been overwhelmingly long technology
    and short energy -- that is a sector bet wearing a stock-selection
    costume. Demeaning within sector removes it, leaving only the
    relative-to-peers view the signal was supposed to express.

    Tickers with no sector mapping are grouped into a residual bucket
    rather than dropped, so coverage is not silently reduced.
    """
    if sectors is None or sectors.empty:
        return df

    mapped = sectors.reindex(df.columns).fillna("_UNMAPPED")
    groups = mapped.groupby(mapped).groups

    out = df.copy()
    for _, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            # A one-name sector demeans to exactly zero, destroying the
            # signal for that name. Leave it untouched instead.
            continue
        block = df[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def prepare(
    raw: pd.DataFrame,
    sectors: pd.Series | None = None,
    *,
    neutralize: bool = True,
    method: str = "zscore",
) -> pd.DataFrame:
    """Full standard pipeline from raw signal to comparable scores."""
    out = winsorize(raw)
    if neutralize and sectors is not None:
        out = sector_neutralize(out, sectors)
    if method == "zscore":
        return zscore(out)
    if method == "rank":
        return rank_normalize(out)
    raise ValueError(f"Unknown normalisation method: {method}")


def to_rebalance_dates(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Down-sample a daily signal to rebalance dates.

    Uses the last observation on or before each period end -- never a
    future one. ``resample().last()`` is safe here precisely because it
    looks backwards within the period.
    """
    return df.resample(freq).last().dropna(how="all")


def combine(
    prepared: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weighted composite of prepared signals.

    Defaults to equal weight. This is not a placeholder for something
    cleverer -- equal weighting is a genuinely strong baseline that
    optimised weights frequently fail to beat out of sample, because
    weight estimation itself overfits.
    """
    if not prepared:
        raise ValueError("No signals to combine.")
    names = list(prepared)
    w = {n: 1.0 for n in names} if weights is None else dict(weights)

    missing = [n for n in names if n not in w]
    if missing:
        raise ValueError(f"No weight supplied for: {missing}")
    total = sum(abs(v) for v in w.values())
    if total == 0:
        raise ValueError("Weights sum to zero.")

    idx = sorted(set().union(*[set(f.index) for f in prepared.values()]))
    cols = sorted(set().union(*[set(f.columns) for f in prepared.values()]))

    acc = pd.DataFrame(0.0, index=pd.DatetimeIndex(idx), columns=cols)
    cnt = pd.DataFrame(0.0, index=pd.DatetimeIndex(idx), columns=cols)

    for n in names:
        f = prepared[n].reindex(index=acc.index, columns=acc.columns)
        contrib = f * w[n]
        acc = acc.add(contrib.fillna(0.0))
        cnt = cnt.add(f.notna().astype(float) * abs(w[n]))

    # Divide by realised weight so partial coverage is not penalised.
    composite = acc.div(cnt.replace(0.0, np.nan))
    return composite.where(cnt > 0)
