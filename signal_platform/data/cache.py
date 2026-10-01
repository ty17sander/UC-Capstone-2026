"""Parquet-backed cache with a DuckDB query surface.

Rationale: Parquet files on disk are cheap, portable, diffable-by-hash,
and readable by pandas, polars, R (arrow), and DuckDB alike. DuckDB gives
SQL over them with zero server to run. This is the right weight class for
a project of this size -- Postgres would be operational overhead with no
corresponding benefit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from ..config import CURATED_DIR

_MANIFEST = CURATED_DIR / "_manifest.json"


def _load_manifest() -> dict:
    if _MANIFEST.exists():
        try:
            return json.loads(_MANIFEST.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_manifest(m: dict) -> None:
    _MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def path_for(name: str) -> Path:
    return CURATED_DIR / f"{name}.parquet"


def write(name: str, df: pd.DataFrame, *, meta: dict | None = None) -> Path:
    """Persist a dataframe and record when it was written."""
    p = path_for(name)
    df.to_parquet(p, index=True)
    man = _load_manifest()
    man[name] = {
        "written_at": time.time(),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns][:200],
        **(meta or {}),
    }
    _save_manifest(man)
    return p


def read(name: str) -> pd.DataFrame:
    p = path_for(name)
    if not p.exists():
        raise FileNotFoundError(
            f"No cached dataset '{name}'. Run the ingestion step first."
        )
    return pd.read_parquet(p)


def exists(name: str) -> bool:
    return path_for(name).exists()


def age_seconds(name: str) -> float | None:
    """Seconds since this dataset was written, or None if unknown."""
    entry = _load_manifest().get(name)
    if not entry:
        return None
    return time.time() - float(entry["written_at"])


def is_stale(name: str, max_age_hours: float) -> bool:
    age = age_seconds(name)
    if age is None:
        return True
    return age > max_age_hours * 3600.0


def sql(query: str) -> pd.DataFrame:
    """Run DuckDB SQL against the curated parquet directory.

    Datasets are exposed as views named after their file stem, so
    ``sql("SELECT * FROM prices LIMIT 5")`` works with no setup.
    """
    con = duckdb.connect()
    try:
        for p in CURATED_DIR.glob("*.parquet"):
            con.execute(
                f"CREATE OR REPLACE VIEW {p.stem} AS "
                f"SELECT * FROM read_parquet('{p.as_posix()}')"
            )
        return con.execute(query).fetchdf()
    finally:
        con.close()


def summary() -> pd.DataFrame:
    """What is cached, how big, and how old -- for the dashboard footer."""
    man = _load_manifest()
    if not man:
        return pd.DataFrame(columns=["dataset", "rows", "age_hours"])
    rows = [
        {
            "dataset": k,
            "rows": v.get("rows"),
            "age_hours": round((time.time() - float(v["written_at"])) / 3600.0, 2),
        }
        for k, v in man.items()
    ]
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
