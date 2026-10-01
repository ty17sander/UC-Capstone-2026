"""Point-in-time universe construction.

The single most common fatal error in student backtests is pulling
*today's* index membership and running it over ten years of history. That
silently guarantees profits, because you only ever hold companies that
survived to the present.

This module reconstructs approximate historical S&P 500 membership by
taking the current constituent list and walking Wikipedia's change log
backwards in time. It is imperfect -- Wikipedia's change table is
incomplete before roughly the mid-2010s, and ticker renames are handled
crudely -- and those limitations belong in the writeup rather than being
quietly ignored. It is nonetheless a large improvement over the naive
approach, and moves the bias from "fatal and undisclosed" to "bounded and
measured".
"""
from __future__ import annotations

import pandas as pd

from ..config import SECTOR_ETF_INCEPTION, SECTOR_ETFS, WIKI_SP500_URL
from . import cache


# ---------------------------------------------------------------------------
# Sector ETF universe -- zero survivorship bias, no reconstruction needed
# ---------------------------------------------------------------------------
def sector_etf_universe(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean membership matrix (date x ETF) respecting inception dates."""
    inception = {k: pd.Timestamp(v) for k, v in SECTOR_ETF_INCEPTION.items()}
    frame = pd.DataFrame(False, index=dates, columns=sorted(SECTOR_ETFS))
    for etf, start in inception.items():
        if etf in frame.columns:
            frame.loc[frame.index >= start, etf] = True
    frame.index.name = "date"
    return frame


# ---------------------------------------------------------------------------
# S&P 500 point-in-time reconstruction
# ---------------------------------------------------------------------------
# Wikipedia rejects requests carrying a generic library User-Agent with a
# 403. Their robot policy asks for a descriptive agent identifying the
# tool and a contact. Edit the contact below to your own address.
WIKI_USER_AGENT = (
    "SignalResearchPlatform/0.1 (academic capstone project; "
    "contact: your.email@example.edu)"
)


def _fetch_html(url: str, *, timeout: int = 30) -> str:
    """Retrieve a page with headers Wikipedia will accept."""
    import requests

    resp = requests.get(
        url,
        headers={
            "User-Agent": WIKI_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex header into readable single-level names.

    Wikipedia's change log uses a two-level header where the first level
    repeats ('Date'/'Date') or pairs a group with a field
    ('Added'/'Ticker'). Naive joining yields 'Date_Date'; de-duplicating
    the parts yields 'Date' and 'Added_Ticker', which is what the field
    matching below expects.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).strip() for c in df.columns]
        return df

    names = []
    for tup in df.columns:
        parts = []
        for p in tup:
            p = str(p).strip()
            if not p or p.lower().startswith("unnamed") or p in parts:
                continue
            parts.append(p)
        names.append("_".join(parts) if parts else "col")
    df.columns = names
    return df


def _score_changes_table(df: pd.DataFrame) -> int:
    cols = " ".join(str(c).lower() for c in df.columns)
    return sum([
        2 if "date" in cols else 0,
        2 if "added" in cols else 0,
        2 if "removed" in cols else 0,
        1 if "ticker" in cols or "symbol" in cols else 0,
    ])


def _score_current_table(df: pd.DataFrame) -> int:
    cols = " ".join(str(c).lower() for c in df.columns)
    return sum([
        2 if "symbol" in cols or "ticker" in cols else 0,
        2 if "gics" in cols or "sector" in cols else 0,
        1 if "security" in cols or "company" in cols else 0,
        -3 if "added" in cols and "removed" in cols else 0,
    ])


def describe_tables(url: str = WIKI_SP500_URL) -> None:
    """Diagnostic: print every table found and its columns.

    Call this when parsing fails. Wikipedia reorganises pages
    periodically, and seeing the actual structure beats guessing.
    """
    from io import StringIO

    tables = pd.read_html(StringIO(_fetch_html(url)))
    print(f"Found {len(tables)} tables at {url}\n")
    for i, t in enumerate(tables):
        flat = _flatten_columns(t.copy())
        print(f"--- table[{i}]  shape={t.shape}")
        print(f"    columns: {list(flat.columns)}")
        print(f"    changes_score={_score_changes_table(flat)} "
              f"current_score={_score_current_table(flat)}")
        print(flat.head(2).to_string()[:400])
        print()


def fetch_sp500_tables(url: str = WIKI_SP500_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape the current-constituents table and the change log.

    Tables are located by *content* rather than position. Wikipedia
    reorders and reformats pages without notice, so indexing tables[0]
    and tables[1] is a latent breakage waiting for the next page edit.
    Scoring each table on the fields it actually contains survives that.

    Note also the two-step fetch: letting pandas make the HTTP call would
    use urllib's default User-Agent, which Wikipedia blocks with a 403.
    """
    from io import StringIO

    raw_tables = [_flatten_columns(t) for t in pd.read_html(StringIO(_fetch_html(url)))]
    if not raw_tables:
        raise RuntimeError(f"No tables found at {url}")

    current = max(raw_tables, key=_score_current_table).copy()
    changes = max(raw_tables, key=_score_changes_table).copy()

    if _score_current_table(current) < 3 or _score_changes_table(changes) < 4:
        raise RuntimeError(
            "Could not identify the constituent and change tables. "
            "Run universe.describe_tables() to inspect the page layout."
        )

    # ---- current constituents -------------------------------------------
    col_lookup = {str(c).lower(): c for c in current.columns}

    def _pick(*keys: str) -> str | None:
        for want in keys:
            for c_low, c_orig in col_lookup.items():
                if want in c_low:
                    return c_orig
        return None

    rename = {}
    for target, keys in {
        "ticker": ("symbol", "ticker"),
        "name": ("security", "company"),
        "sector": ("gics sector", "sector"),
        "sub_industry": ("sub-industry", "sub industry"),
        "date_added": ("date added", "date first added"),
    }.items():
        found = _pick(*keys)
        if found is not None:
            rename[found] = target
    current = current.rename(columns=rename)

    keep = [c for c in ["ticker", "name", "sector", "sub_industry", "date_added"]
            if c in current.columns]
    if "ticker" not in keep:
        raise RuntimeError(
            f"No ticker column in the constituents table. Columns: "
            f"{list(current.columns)}"
        )
    current = current[keep].copy()
    current["ticker"] = current["ticker"].astype(str).str.replace(".", "-", regex=False)

    # ---- change log ------------------------------------------------------
    col = {str(c).lower(): c for c in changes.columns}

    def _find(*keys: str) -> str | None:
        for c_low, c_orig in col.items():
            if all(k in c_low for k in keys):
                return c_orig
        return None

    date_col = _find("date")
    added_col = _find("added", "ticker") or _find("added")
    removed_col = _find("removed", "ticker") or _find("removed")

    if date_col is None or (added_col is None and removed_col is None):
        raise RuntimeError(
            f"Change table missing expected fields. Columns: "
            f"{list(changes.columns)}. Run universe.describe_tables()."
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(changes[date_col], errors="coerce"),
        "added": changes[added_col] if added_col else None,
        "removed": changes[removed_col] if removed_col else None,
    }).dropna(subset=["date"])

    for c in ("added", "removed"):
        out[c] = (out[c].astype(str)
                  .str.replace(".", "-", regex=False)
                  .replace({"nan": None, "": None, "None": None}))

    return current, out.sort_values("date").reset_index(drop=True)


def reconstruct_membership(
    current_tickers: set[str],
    changes: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Walk the change log backwards to produce a membership matrix.

    Parameters
    ----------
    current_tickers : the index constituents as of today
    changes : columns ``date``, ``added``, ``removed``
    dates : the trading-day index to produce membership for

    Returns
    -------
    Boolean DataFrame indexed by date, columns are every ticker that was
    ever a member over the window -- including names that have since been
    delisted or acquired, which is the entire point.
    """
    changes = changes.sort_values("date", ascending=False)
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))

    # Snapshot membership immediately *after* each change date, moving back.
    snapshots: list[tuple[pd.Timestamp, set[str]]] = []
    members = set(current_tickers)
    snapshots.append((pd.Timestamp.max, set(members)))

    for _, row in changes.iterrows():
        # Reverse the change to get membership just before it happened.
        if row["added"] and row["added"] in members:
            members.discard(row["added"])
        if row["removed"]:
            members.add(row["removed"])
        snapshots.append((pd.Timestamp(row["date"]), set(members)))

    # snapshots[i] = (effective_date, membership *before* that date)
    all_tickers = sorted({t for _, s in snapshots for t in s})
    frame = pd.DataFrame(False, index=dates, columns=all_tickers)

    ordered = sorted(snapshots, key=lambda x: x[0])
    for d in dates:
        chosen = ordered[-1][1]
        for eff_date, snap in ordered:
            if d < eff_date:
                chosen = snap
                break
        frame.loc[d, list(chosen)] = True

    frame.index.name = "date"
    return frame


def _parse_constituents(html: str) -> pd.DataFrame:
    """Extract the constituent table from a rendered page revision."""
    from io import StringIO

    tables = [_flatten_columns(t) for t in pd.read_html(StringIO(html))]
    if not tables:
        raise RuntimeError("No tables in revision HTML")
    best = max(tables, key=_score_current_table)

    lookup = {str(c).lower(): c for c in best.columns}

    def _pick(*keys: str) -> str | None:
        for want in keys:
            for c_low, c_orig in lookup.items():
                if want in c_low:
                    return c_orig
        return None

    rename = {}
    for target, keys in {
        "ticker": ("symbol", "ticker"),
        "name": ("security", "company"),
        "sector": ("gics sector", "sector"),
        "sub_industry": ("sub-industry", "sub industry"),
        "date_added": ("date added", "date first added"),
    }.items():
        found = _pick(*keys)
        if found is not None:
            rename[found] = target
    best = best.rename(columns=rename)

    if "ticker" not in best.columns:
        raise RuntimeError(f"No ticker column; found {list(best.columns)}")

    keep = [c for c in ["ticker", "name", "sector", "sub_industry", "date_added"]
            if c in best.columns]
    out = best[keep].copy()
    out["ticker"] = (out["ticker"].astype(str)
                     .str.replace(".", "-", regex=False)
                     .str.strip()
                     .str.upper())
    # Strip footnote markers and obvious junk rows.
    out = out[out["ticker"].str.match(r"^[A-Z][A-Z0-9\-]{0,6}$", na=False)]
    return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def _revision_id_at(timestamp: str, title: str = "List of S&P 500 companies") -> int | None:
    """Latest revision id at or before a timestamp, via the MediaWiki API."""
    import requests

    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "prop": "revisions", "titles": title,
            "rvlimit": 1, "rvdir": "older", "rvstart": timestamp,
            "rvprop": "ids|timestamp", "format": "json", "formatversion": 2,
        },
        headers={"User-Agent": WIKI_USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages:
        return None
    revs = pages[0].get("revisions", [])
    return int(revs[0]["revid"]) if revs else None


def snapshot_membership(
    dates: pd.DatetimeIndex,
    *,
    freq: str = "QE",
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Point-in-time membership from historical Wikipedia revisions.

    For each period end, fetch the revision of the constituents page as it
    stood on that date and read the table. This is a direct observation of
    what the index looked like then, rather than an inference from a
    change log -- and it survives the change log being moved or deleted,
    which is exactly what happened.

    Quarterly sampling means membership changes are dated to within a
    quarter. For weekly-rebalance research that imprecision is immaterial;
    for anything event-driven it would not be. Pass freq='ME' for monthly
    at three times the request count.

    Snapshots are cached individually, so an interrupted run resumes
    instead of starting over.
    """
    import time

    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    periods = pd.date_range(dates[0], dates[-1], freq=freq)
    if len(periods) == 0 or periods[-1] < dates[-1]:
        periods = periods.append(pd.DatetimeIndex([dates[-1]]))

    snapshots: dict[pd.Timestamp, set[str]] = {}
    sector_hist: dict[str, str] = {}
    cache_key = "universe_snapshots"
    cached: dict[str, list[str]] = {}
    if cache.exists(cache_key):
        prev = cache.read(cache_key)
        for col in prev.columns:
            cached[col] = [t for t in prev[col].dropna().tolist() if t]

    for p in periods:
        key = p.strftime("%Y-%m-%d")
        if key in cached and cached[key]:
            snapshots[p] = set(cached[key])
            continue
        try:
            rev = _revision_id_at(p.strftime("%Y-%m-%dT%H:%M:%SZ"))
            if rev is None:
                continue
            html = _fetch_html(
                f"https://en.wikipedia.org/w/index.php?oldid={rev}"
            )
            parsed = _parse_constituents(html)
            tickers = set(parsed["ticker"])
            if len(tickers) < 400:
                if verbose:
                    print(f"  {key}: only {len(tickers)} tickers, skipping")
                continue
            snapshots[p] = tickers
            # Capture each name's sector as of when it was a member. This
            # is the only place delisted companies' sectors are available
            # -- the current constituents page has long since dropped them.
            if "sector" in parsed.columns:
                for tk, sec in zip(parsed["ticker"], parsed["sector"]):
                    if isinstance(sec, str) and sec.strip():
                        sector_hist.setdefault(tk, sec.strip())
            if verbose:
                print(f"  {key}: {len(tickers)} constituents (rev {rev})")
            time.sleep(pause)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  {key}: failed ({type(exc).__name__}: {exc})")

    if sector_hist:
        # Merge rather than overwrite. Cached snapshots are skipped above
        # without re-fetching their HTML, so a run that fetches only one
        # new quarter would otherwise replace a full historical map with
        # that single snapshot's ~503 live names -- silently undoing the
        # delisted-name coverage this map exists to provide.
        existing: dict[str, str] = {}
        if cache.exists("sp500_sectors_hist"):
            prev = cache.read("sp500_sectors_hist")
            if "sector" in prev.columns:
                existing = prev["sector"].dropna().to_dict()
        # Earlier observations win: sector is recorded as first seen.
        merged = {**sector_hist, **existing}
        if len(merged) < len(existing):
            raise RuntimeError("Sector map would shrink; refusing to write.")
        cache.write("sp500_sectors_hist",
                    pd.DataFrame({"sector": pd.Series(merged)}),
                    meta={"n": len(merged)})
        if verbose:
            print(f"  historical sector map: {len(merged)} tickers "
                  f"({len(merged) - len(existing)} new)")

    if not snapshots:
        raise RuntimeError(
            "No revision snapshots retrieved. Check network access to "
            "en.wikipedia.org and that WIKI_USER_AGENT names a contact."
        )

    # Persist for resumability.
    maxlen = max(len(v) for v in snapshots.values())
    cache.write(cache_key, pd.DataFrame({
        k.strftime("%Y-%m-%d"): sorted(v) + [None] * (maxlen - len(v))
        for k, v in snapshots.items()
    }))

    all_tickers = sorted(set().union(*snapshots.values()))
    frame = pd.DataFrame(False, index=dates, columns=all_tickers)
    ordered = sorted(snapshots.items())

    for d in dates:
        # Use the most recent snapshot at or before this date; before the
        # first snapshot, fall back to the earliest one available.
        chosen = ordered[0][1]
        for snap_date, snap in ordered:
            if snap_date <= d:
                chosen = snap
            else:
                break
        frame.loc[d, sorted(chosen)] = True

    frame.index.name = "date"
    return frame


def build_universe(
    dates: pd.DatetimeIndex,
    *,
    refresh: bool = False,
    method: str = "snapshots",
) -> pd.DataFrame:
    """Fetch, reconstruct, and cache the point-in-time membership matrix.

    method='snapshots' reads historical revisions of the constituents page
    (robust, ~36 requests for a decade at quarterly sampling).
    method='changelog' uses the older backward-walk approach, retained for
    reference but dependent on a change table Wikipedia has since removed
    from the main page.
    """
    if not refresh and cache.exists("universe_sp500"):
        cached = cache.read("universe_sp500")
        if set(pd.DatetimeIndex(dates)).issubset(set(cached.index)):
            return cached.loc[dates]

    if method == "snapshots":
        membership = snapshot_membership(dates)
        try:
            current, _ = fetch_sp500_tables()
            cache.write("sp500_meta", current.set_index("ticker"))
        except Exception:
            pass
    elif method == "changelog":
        current, changes = fetch_sp500_tables()
        membership = reconstruct_membership(set(current["ticker"]), changes, dates)
        cache.write("sp500_meta", current.set_index("ticker"))
    else:
        raise ValueError(f"Unknown method: {method}")

    cache.write("universe_sp500", membership,
                meta={"source": f"wikipedia:{method}",
                      "n_tickers": membership.shape[1]})
    return membership


def sector_map(*, refresh: bool = False) -> pd.Series:
    """ticker -> GICS sector, covering delisted names where possible.

    Prefers the historical map accumulated from revision snapshots, which
    includes companies that have since left the index. Falls back to the
    current constituents page, which covers only ~500 live names and
    leaves every departed company unmapped -- and an unmapped bucket
    holding a third of the universe makes sector neutralization actively
    harmful rather than merely incomplete.

    Limitation to disclose: a ticker's sector is recorded as first
    observed, so GICS reclassifications (notably the 2018 creation of
    Communication Services) are not tracked through time.
    """
    parts = []
    if cache.exists("sp500_sectors_hist") and not refresh:
        hist = cache.read("sp500_sectors_hist")
        if "sector" in hist.columns:
            parts.append(hist["sector"])

    if refresh or not cache.exists("sp500_meta"):
        try:
            current, _ = fetch_sp500_tables()
            cache.write("sp500_meta", current.set_index("ticker"))
        except Exception:
            pass
    if cache.exists("sp500_meta"):
        meta = cache.read("sp500_meta")
        if "sector" in meta.columns:
            parts.append(meta["sector"])

    if not parts:
        return pd.Series(dtype=object)

    combined = pd.concat(parts)
    return combined[~combined.index.duplicated(keep="first")].rename("sector")


def rebuild_sector_map_from_snapshots(verbose: bool = True) -> pd.Series:
    """Re-derive the historical sector map without refetching membership.

    Use when snapshots were cached by an earlier version that discarded
    sector labels. Reads the cached snapshot ticker lists, refetches only
    the revisions needed, and rewrites the sector map.
    """
    if not cache.exists("universe_snapshots"):
        raise RuntimeError("No cached snapshots -- run build_universe first.")
    snaps = cache.read("universe_snapshots")
    sector_hist: dict[str, str] = {}
    import time

    for col in snaps.columns:
        try:
            rev = _revision_id_at(f"{col}T00:00:00Z")
            if rev is None:
                continue
            parsed = _parse_constituents(
                _fetch_html(f"https://en.wikipedia.org/w/index.php?oldid={rev}")
            )
            if "sector" not in parsed.columns:
                continue
            for tk, sec in zip(parsed["ticker"], parsed["sector"]):
                if isinstance(sec, str) and sec.strip():
                    sector_hist.setdefault(tk, sec.strip())
            if verbose:
                print(f"  {col}: {len(sector_hist)} tickers mapped so far")
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  {col}: failed ({exc})")

    if not sector_hist:
        raise RuntimeError("No sector labels recovered.")
    existing: dict[str, str] = {}
    if cache.exists("sp500_sectors_hist"):
        prev = cache.read("sp500_sectors_hist")
        if "sector" in prev.columns:
            existing = prev["sector"].dropna().to_dict()
    merged = {**existing, **sector_hist}
    cache.write("sp500_sectors_hist",
                pd.DataFrame({"sector": pd.Series(merged)}),
                meta={"n": len(merged)})
    if verbose:
        print(f"Historical sector map rebuilt: {len(merged)} tickers")
    return pd.Series(merged)


def apply_liquidity_filter(
    membership: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    min_dollar_volume: float,
    min_price: float,
    window: int = 21,
) -> pd.DataFrame:
    """Intersect index membership with a tradeable-liquidity screen.

    Shorting requires locatable borrow. Restricting to liquid names is the
    cheap proxy for that, and prevents the backtest from assuming trades
    that could not have been executed.
    """
    dollar_vol = (close * volume).rolling(window, min_periods=max(5, window // 2)).mean()
    liquid = (dollar_vol >= min_dollar_volume) & (close >= min_price)
    aligned = liquid.reindex(index=membership.index, columns=membership.columns).fillna(False)
    return membership & aligned
