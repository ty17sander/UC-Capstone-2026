"""Market sentiment composite index.

A five-component reconstruction in the spirit of CNN's Fear & Greed
index, restricted to inputs obtainable reliably without an API key.

Components (each scaled so that 100 = extreme greed, 0 = extreme fear):

  1. Volatility      VIX relative to its own 50-day average, inverted
  2. Credit appetite HYG versus LQD relative return (junk vs investment grade)
  3. Safe-haven      SPY versus TLT relative return (stocks vs treasuries)
  4. Momentum        SPY relative to its 125-day average
  5. Breadth         share of the universe above its 200-day average

Two components of CNN's published index are deliberately omitted. The
McClellan volume summation requires NYSE advance-decline volume, and the
put/call ratio requires CBOE history; neither is reliably free. Their
absence is a documented limitation rather than a silent substitution.

THE CRITICAL DESIGN CHOICE is how each component is scaled to 0-100.
Normalising against the full sample -- the obvious approach, and the one
most published reconstructions use -- means today's reading depends on
data from years in the future. Any predictive test built on such an index
is contaminated from the outset. Every component here is normalised
against a trailing window only, so the value at date t uses nothing
after t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import cache

# Tickers needed beyond the equity universe already held.
PROXY_TICKERS = ["^VIX", "SPY", "TLT", "HYG", "LQD"]

# Trailing window for percentile normalisation. Three years is long
# enough to span a full sentiment cycle without being so long that the
# index stops responding to regime change.
NORM_WINDOW = 756
MIN_NORM_OBS = 252


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def causal_percentile(series: pd.Series, window: int = NORM_WINDOW,
                      min_obs: int = MIN_NORM_OBS) -> pd.Series:
    """Map a series to 0-100 by its rank within a trailing window.

    The value at date t is the percentile of observation t among the
    preceding `window` observations *including t itself but nothing
    after it*. This is what makes the index usable in a predictive test.

    A full-sample percentile would be trivially easy to compute and
    silently wrong: it would tell you that March 2020 was an extreme
    only because you already know what 2021 looked like.
    """
    def _rank(x: np.ndarray) -> float:
        if len(x) < min_obs:
            return np.nan
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) < min_obs:
            return np.nan
        return 100.0 * (valid < last).sum() / len(valid)

    return series.rolling(window, min_periods=min_obs).apply(_rank, raw=True)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def component_volatility(vix: pd.Series, window: int = 50) -> pd.Series:
    """VIX relative to its own average. Inverted: high VIX is fear."""
    ratio = vix / vix.rolling(window, min_periods=window // 2).mean()
    return 100.0 - causal_percentile(ratio)


def component_credit(hyg: pd.Series, lqd: pd.Series, window: int = 20) -> pd.Series:
    """Junk versus investment-grade bond performance.

    Proxy for the high-yield option-adjusted spread. When investors are
    confident they reach for yield and junk outperforms. Weaker than the
    actual OAS series but obtainable without an API key.
    """
    rel = (hyg / hyg.shift(window)) - (lqd / lqd.shift(window))
    return causal_percentile(rel)


def component_safe_haven(spy: pd.Series, tlt: pd.Series, window: int = 20) -> pd.Series:
    """Stocks versus long treasuries. Equity outperformance is greed."""
    rel = (spy / spy.shift(window)) - (tlt / tlt.shift(window))
    return causal_percentile(rel)


def component_momentum(spy: pd.Series, window: int = 125) -> pd.Series:
    """Index level relative to its own medium-term average."""
    ratio = spy / spy.rolling(window, min_periods=window // 2).mean()
    return causal_percentile(ratio)


def component_breadth(close: pd.DataFrame, window: int = 200,
                      membership: pd.DataFrame | None = None) -> pd.Series:
    """Share of the universe trading above its own long-run average.

    Computed from the equity panel already held, so it costs nothing
    extra and is the component most likely to carry information the
    index-level measures miss: a market can rise on a handful of names
    while most decline.
    """
    ma = close.rolling(window, min_periods=window // 2).mean()
    above = (close > ma)
    if membership is not None:
        mask = membership.reindex(index=close.index, columns=close.columns).fillna(False)
        above = above.where(mask)
        denom = mask.sum(axis=1).replace(0, np.nan)
        share = above.sum(axis=1) / denom
    else:
        share = above.sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    return causal_percentile(share * 100.0)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def fetch_proxies(start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
    """Download the index-level series the sentiment components need."""
    from .. data import prices

    key = "sentiment_proxies"
    if not refresh and cache.exists(key):
        cached = cache.read(key)
        have = set(cached.columns)
        if set(PROXY_TICKERS).issubset(have):
            return cached

    data = prices.download_prices(PROXY_TICKERS, start, end)
    if "close" not in data:
        raise RuntimeError("Could not download sentiment proxy series.")
    close = data["close"]
    missing = [t for t in PROXY_TICKERS if t not in close.columns
               or close[t].isna().all()]
    if missing:
        print(f"  WARNING: no data for {missing}; those components will be skipped")
    cache.write(key, close, meta={"start": start, "end": end})
    return close


def build_index(
    proxies: pd.DataFrame,
    close: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble the composite index and return it with its components.

    The composite is the equal-weighted mean of whichever components are
    available on each date. Equal weighting is chosen over anything
    fitted because fitting component weights against realised returns
    would convert a descriptive index into an in-sample optimisation.
    """
    comps: dict[str, pd.Series] = {}

    def _col(name: str) -> pd.Series | None:
        if name in proxies.columns and not proxies[name].isna().all():
            return proxies[name].dropna()
        return None

    vix, spy, tlt = _col("^VIX"), _col("SPY"), _col("TLT")
    hyg, lqd = _col("HYG"), _col("LQD")

    if vix is not None:
        comps["volatility"] = component_volatility(vix)
    if hyg is not None and lqd is not None:
        comps["credit"] = component_credit(hyg, lqd)
    if spy is not None and tlt is not None:
        comps["safe_haven"] = component_safe_haven(spy, tlt)
    if spy is not None:
        comps["momentum"] = component_momentum(spy)
    if close is not None and not close.empty:
        comps["breadth"] = component_breadth(close, membership=membership)

    if not comps:
        raise RuntimeError("No sentiment components could be constructed.")

    frame = pd.DataFrame(comps).sort_index()
    frame["composite"] = frame[list(comps)].mean(axis=1, skipna=True)
    frame["n_components"] = frame[list(comps)].notna().sum(axis=1)

    if verbose:
        print(f"  components built: {', '.join(comps)}")
        n = frame["composite"].notna().sum()
        print(f"  composite available on {n:,} dates "
              f"({frame.index[0].date()} to {frame.index[-1].date()})")
        missing = {"volatility", "credit", "safe_haven", "momentum", "breadth"} - set(comps)
        if missing:
            print(f"  NOT built (missing data): {', '.join(sorted(missing))}")
    return frame


def label(value: float) -> str:
    """CNN-style categorical label for a 0-100 reading."""
    if pd.isna(value):
        return "unavailable"
    if value < 25:
        return "Extreme Fear"
    if value < 45:
        return "Fear"
    if value < 55:
        return "Neutral"
    if value < 75:
        return "Greed"
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# Does it predict anything?
# ---------------------------------------------------------------------------
def null_calibration(
    composite: pd.Series,
    market: pd.Series,
    *,
    horizon: int = 63,
    n_sims: int = 500,
    block: int = 63,
    seed: int = 0,
) -> dict:
    """Empirical null distribution for the predictive slope t-statistic.

    WHY THIS IS NECESSARY. A predictive regression of forward returns on
    a persistent, price-derived regressor is biased even when no
    relationship exists (Stambaugh, 1999). The sentiment index is exactly
    such a regressor: it is highly autocorrelated and several of its
    components are functions of the same price series whose future it is
    being asked to predict. Newey-West standard errors correct for
    overlapping returns; they do not correct for this.

    The consequence is not theoretical. Running the standard test on
    synthetic data containing no relationship by construction produced a
    slope t-statistic of -3.04, which would have cleared a conventional
    hurdle and been reported as a discovery.

    This function establishes what the test statistic actually does under
    the null. Market returns are resampled in blocks, preserving their
    autocorrelation and volatility clustering while destroying any
    relationship with sentiment. The observed statistic is then compared
    against that empirical distribution rather than against a theoretical
    one that does not apply.
    """
    rng = np.random.default_rng(seed)
    df = pd.concat([composite.rename("s"), market.rename("px")], axis=1).dropna()
    if len(df) < 300:
        return {"error": "insufficient overlap"}

    rets = df["px"].pct_change().dropna()
    sentiment = df["s"].reindex(rets.index)

    def _slope_t(sent: pd.Series, r: pd.Series) -> float:
        import statsmodels.api as sm
        px = (1.0 + r).cumprod()
        fwd = px.shift(-1 - horizon) / px.shift(-1) - 1.0
        sub = pd.concat([sent.rename("s"), fwd.rename("f")], axis=1).dropna()
        if len(sub) < 200:
            return np.nan
        model = sm.OLS(sub["f"].values,
                       sm.add_constant(sub["s"].values)).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 1)})
        return float(model.tvalues[1])

    observed = _slope_t(sentiment, rets)

    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    null_ts = []
    for _ in range(n_sims):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        shuffled = np.concatenate([rets.values[s:s + block] for s in starts])[:n]
        null_ts.append(_slope_t(sentiment, pd.Series(shuffled, index=rets.index)))

    null_ts = np.array([t for t in null_ts if not np.isnan(t)])
    if len(null_ts) < 50:
        return {"error": "too few successful simulations"}

    p_emp = float((np.abs(null_ts) >= abs(observed)).mean())
    return {
        "observed_t": observed,
        "null_mean_t": float(null_ts.mean()),
        "null_std_t": float(null_ts.std(ddof=1)),
        "null_p05": float(np.percentile(np.abs(null_ts), 5)),
        "null_p95": float(np.percentile(np.abs(null_ts), 95)),
        "null_p99": float(np.percentile(np.abs(null_ts), 99)),
        "empirical_p_value": p_emp,
        "n_sims": int(len(null_ts)),
        "significant": bool(p_emp < 0.05),
    }


def predictive_test(
    composite: pd.Series,
    market: pd.Series,
    *,
    horizons: tuple[int, ...] = (5, 21, 63),
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Test whether the index predicts forward market returns.

    This is the question that decides whether the index is worth
    anything beyond description. Most sentiment measures are
    *coincident* -- they tell you the market fell, which you already
    knew. A predictive relationship would show forward returns varying
    systematically across sentiment buckets.

    Forward returns are lagged one day, consistent with the equity
    pipeline: a reading computed on the close of t is tradeable at t+1.
    """
    import statsmodels.api as sm

    df = pd.concat([composite.rename("sentiment"), market.rename("px")], axis=1).dropna()
    rows = []

    for h in horizons:
        fwd = df["px"].shift(-1 - h) / df["px"].shift(-1) - 1.0
        sub = pd.concat([df["sentiment"], fwd.rename("fwd")], axis=1).dropna()
        if len(sub) < 200:
            continue

        # Correlation with a Newey-West t-statistic on the regression.
        X = sm.add_constant(sub["sentiment"].values)
        model = sm.OLS(sub["fwd"].values, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 1)})

        buckets = pd.qcut(sub["sentiment"], n_buckets, labels=False, duplicates="drop")
        by_bucket = sub.groupby(buckets)["fwd"].mean() * (252.0 / h)

        rows.append({
            "horizon": h,
            "n": len(sub),
            "corr": float(sub["sentiment"].corr(sub["fwd"])),
            "slope_t": float(model.tvalues[1]),
            "p_value": float(model.pvalues[1]),
            "r_squared": float(model.rsquared),
            "low_bucket_ann": float(by_bucket.iloc[0]) if len(by_bucket) else np.nan,
            "high_bucket_ann": float(by_bucket.iloc[-1]) if len(by_bucket) else np.nan,
        })

    out = pd.DataFrame(rows).set_index("horizon")
    out["spread_ann"] = out["low_bucket_ann"] - out["high_bucket_ann"]
    return out


def coincident_test(composite: pd.Series, market: pd.Series,
                    window: int = 21) -> dict:
    """Measure how much of the index is simply describing the recent past.

    A high correlation with *trailing* returns alongside a negligible
    correlation with *forward* returns is the signature of a coincident
    indicator: accurate, widely reported, and useless for decisions.
    """
    df = pd.concat([composite.rename("s"), market.rename("px")], axis=1).dropna()
    trailing = df["px"] / df["px"].shift(window) - 1.0
    forward = df["px"].shift(-1 - window) / df["px"].shift(-1) - 1.0
    joined = pd.concat([df["s"], trailing.rename("t"), forward.rename("f")],
                       axis=1).dropna()
    return {
        "corr_with_trailing": float(joined["s"].corr(joined["t"])),
        "corr_with_forward": float(joined["s"].corr(joined["f"])),
        "n": int(len(joined)),
    }


def to_exposure_scaler(
    composite: pd.Series,
    *,
    direction: str = "contrarian",
    floor: float = 0.5,
    ceiling: float = 1.5,
) -> pd.Series:
    """Convert the index into a gross-exposure multiplier.

    This is the single hook by which sentiment enters the trading layer,
    and it is deliberately one parameter rather than many. A richer
    scheme -- letting sentiment regime-switch which signals receive
    weight -- multiplies the specification count and is where
    overfitting takes hold.

    direction='contrarian' scales exposure up when sentiment is fearful;
    'momentum' does the reverse. Both should be tested and reported, and
    the choice justified before results are seen rather than after.
    """
    s = composite.clip(0, 100) / 100.0
    if direction == "contrarian":
        raw = 1.0 - s
    elif direction == "momentum":
        raw = s
    else:
        raise ValueError("direction must be 'contrarian' or 'momentum'")
    return (floor + raw * (ceiling - floor)).rename("exposure")
