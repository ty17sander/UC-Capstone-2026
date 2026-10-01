"""Price ingestion.

yfinance is free and undocumented-by-design: it scrapes an endpoint that
Yahoo can change without notice, rate-limits aggressively, and silently
returns empty frames on failure. Treat every response as suspect. The
batching, retry, and validation here is not defensive paranoia -- it is
the difference between a pipeline that runs unattended and one that
quietly caches a week of NaNs.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..config import MIN_PRICE
from . import cache

_BATCH = 50
_MAX_RETRIES = 3
_BACKOFF = 2.0


def _import_yf():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for live ingestion: pip install yfinance"
        ) from exc
    return yf


def download_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download one batch with retries. Returns {field: wide DataFrame}."""
    yf = _import_yf()
    last_err: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    tickers=" ".join(tickers),
                    start=start,
                    end=end,
                    auto_adjust=True,       # split & dividend adjusted
                    progress=False,
                    group_by="column",
                    threads=True,
                )
            if raw is None or raw.empty:
                raise ValueError("empty response")

            out: dict[str, pd.DataFrame] = {}
            for field in ("Close", "Volume", "Open", "High", "Low"):
                if isinstance(raw.columns, pd.MultiIndex):
                    if field not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[field].copy()
                else:
                    if field not in raw.columns:
                        continue
                    sub = raw[[field]].copy()
                    sub.columns = tickers[:1]
                out[field.lower()] = sub
            if not out:
                raise ValueError("no recognised price fields")
            return out

        except Exception as exc:  # noqa: BLE001 - yfinance raises anything
            last_err = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF * (attempt + 1))

    raise RuntimeError(f"Failed to download batch after {_MAX_RETRIES} tries: {last_err}")


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    batch_size: int = _BATCH,
    pause: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Download many tickers in batches, concatenating on the column axis."""
    tickers = sorted(set(t for t in tickers if isinstance(t, str) and t))
    collected: dict[str, list[pd.DataFrame]] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            got = download_batch(batch, start, end)
        except RuntimeError:
            # A dead batch should not kill the whole run; record and continue.
            continue
        for field, frame in got.items():
            collected.setdefault(field, []).append(frame)
        time.sleep(pause)

    return {
        field: pd.concat(frames, axis=1).sort_index()
        for field, frames in collected.items()
    }


def validate(close: pd.DataFrame, *, max_daily_move: float = 0.60) -> pd.DataFrame:
    """Flag data-quality problems rather than trusting the vendor.

    Returns a per-ticker report. Anything with a high ``pct_missing`` or a
    nonzero ``extreme_moves`` count deserves inspection before it enters a
    backtest -- unadjusted splits show up here as 50%+ single-day drops.
    """
    rets = close.pct_change()
    report = pd.DataFrame({
        "n_obs": close.notna().sum(),
        "pct_missing": close.isna().mean().round(4),
        "first_date": close.apply(lambda s: s.first_valid_index()),
        "last_date": close.apply(lambda s: s.last_valid_index()),
        "extreme_moves": (rets.abs() > max_daily_move).sum(),
        "zero_var_days": (rets == 0).sum(),
        "min_price": close.min(),
    })
    report["suspect"] = (
        (report["pct_missing"] > 0.30)
        | (report["extreme_moves"] > 0)
        | (report["min_price"] < MIN_PRICE / 5)
    )
    return report.sort_values("suspect", ascending=False)


def clean_prices(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    *,
    max_daily_move: float = 1.0,
    min_obs: int = 252,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Remove empty columns and neutralise suspected bad prints.

    Two distinct problems:

    1. Columns with no data at all. yfinance returns an all-NaN column for
       tickers it cannot resolve -- typically long-delisted names -- while
       still counting them as retrieved. Left in place they inflate the
       apparent universe size and silently reintroduce survivorship bias,
       since the names it drops are disproportionately the failures.

    2. Unadjusted corporate actions. A merger or reverse split that Yahoo
       has not adjusted shows up as a several-hundred-percent single-day
       move. One such print in a forward-return window can dominate an
       entire cross-section.

    Bad prints are NaN'd at the return level rather than the ticker being
    discarded, so a single bad day does not cost the whole history. The
    threshold is deliberately loose: genuine 100%+ days do occur (GME in
    January 2021), so this only catches the clearly non-economic.
    """
    report: dict = {}

    empty = close.columns[close.isna().all()].tolist()
    report["empty_columns"] = empty
    close = close.drop(columns=empty)
    if volume is not None:
        volume = volume.drop(columns=[c for c in empty if c in volume.columns])

    thin = [c for c in close.columns if close[c].notna().sum() < min_obs]
    report["thin_columns"] = thin
    close = close.drop(columns=thin)
    if volume is not None:
        volume = volume.drop(columns=[c for c in thin if c in volume.columns])

    rets = close.pct_change()
    bad = rets.abs() > max_daily_move
    n_bad = int(bad.sum().sum())
    report["bad_prints"] = n_bad
    report["bad_print_tickers"] = (
        bad.sum()[bad.sum() > 0].sort_values(ascending=False).to_dict()
    )

    if n_bad:
        # Blank the price on the offending day so the return either side
        # of it is undefined rather than fabricated.
        close = close.mask(bad)

    report["n_tickers_final"] = close.shape[1]

    if verbose:
        print(f"  dropped {len(empty)} all-empty columns")
        print(f"  dropped {len(thin)} columns with <{min_obs} observations")
        print(f"  masked {n_bad} suspected bad prints "
              f"(|move| > {max_daily_move:.0%})")
        print(f"  final universe: {close.shape[1]} tickers")

    return close, volume, report


def ingest(
    tickers: list[str],
    start: str,
    end: str,
    *,
    name_prefix: str = "px",
    refresh: bool = False,
    min_coverage: float = 0.95,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Full ingestion path: download, validate, cache.

    The cache is only reused when it actually covers what was asked for.
    Checking merely that a cache file exists is a trap: a five-ticker
    smoke test writes the same key as a full universe pull, and every
    later run silently inherits the small one. Coverage of both the
    ticker set and the date range is verified before reuse.
    """
    fields = ("close", "volume", "open", "high", "low")
    requested = sorted(set(t for t in tickers if isinstance(t, str) and t))

    if not refresh and cache.exists(f"{name_prefix}_close"):
        cached = cache.read(f"{name_prefix}_close")
        have = set(cached.columns)
        overlap = len(have & set(requested)) / max(len(requested), 1)

        want_start, want_end = pd.Timestamp(start), pd.Timestamp(end)
        tol = pd.Timedelta(days=7)
        covers_dates = (
            len(cached.index) > 0
            and cached.index.min() <= want_start + tol
            and cached.index.max() >= want_end - tol
        )

        if overlap >= min_coverage and covers_dates:
            if verbose:
                print(f"Using cached prices: {len(have)} tickers, "
                      f"{cached.index.min().date()} to {cached.index.max().date()}")
            return {
                f: cache.read(f"{name_prefix}_{f}")
                for f in fields
                if cache.exists(f"{name_prefix}_{f}")
            }

        if verbose:
            reason = []
            if overlap < min_coverage:
                reason.append(f"ticker coverage {overlap:.0%} (need {min_coverage:.0%})")
            if not covers_dates:
                reason.append("date range insufficient")
            print(f"Cache miss -- refetching: {'; '.join(reason)}")

    if verbose:
        print(f"Downloading {len(requested)} tickers, {start} to {end}...")

    data = download_prices(requested, start, end)
    if "close" not in data:
        raise RuntimeError("Ingestion returned no close prices.")

    got = data["close"].shape[1]
    if verbose:
        print(f"Retrieved {got} of {len(requested)} tickers "
              f"({got / max(len(requested), 1):.0%})")

    report = validate(data["close"])
    cache.write(f"{name_prefix}_quality", report)

    for field, frame in data.items():
        cache.write(f"{name_prefix}_{field}", frame,
                    meta={"start": start, "end": end, "n_tickers": frame.shape[1]})
    return data


def to_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, with inf guarded."""
    r = close.pct_change()
    return r.replace([np.inf, -np.inf], np.nan)


def forward_returns(close: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    """Return realised over ``horizon`` days, starting ``lag`` days ahead.

    This is the most correctness-critical function in the codebase. A
    signal observed on the close of day t cannot be traded until day t+lag.
    The return it earns therefore spans t+lag to t+lag+horizon.

    With lag=1 and horizon=5, the value at index t is the return from the
    close of t+1 to the close of t+6. Nothing in that window is knowable
    at t, which is exactly what we need.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd = close.shift(-lag - horizon) / close.shift(-lag) - 1.0
    return fwd.replace([np.inf, -np.inf], np.nan)
