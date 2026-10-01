"""Copy the Parquet cache into the repo so the deployed app has data.

Streamlit Community Cloud has ephemeral storage and cannot run the
~30 minute ingestion on startup, so the cache must travel with the repo.

Only the datasets the app actually reads are copied. The OHLC series
(px_high, px_low, px_open) are ingested but never used -- no signal in
the library references them -- and account for roughly two thirds of the
cache by size. Copying them would triple the repository for no benefit.

GitHub warns above 50 MB per file and rejects above 100 MB.
"""
import shutil
from pathlib import Path

from signal_platform.config import CURATED_DIR

# Datasets the app and research pipeline actually read.
NEEDED = {
    "_manifest.json",
    "px_close.parquet",
    "px_volume.parquet",
    "px_quality.parquet",
    "universe_sp500.parquet",
    "universe_snapshots.parquet",
    "sp500_sectors_hist.parquet",
    "sp500_meta.parquet",
    "sentiment_index.parquet",
    "sentiment_proxies.parquet",
}

DEST = Path(__file__).parent / "data" / "curated"
DEST.mkdir(parents=True, exist_ok=True)

# Clear anything staged by an earlier run so removals take effect.
for old in DEST.glob("*"):
    if old.is_file():
        old.unlink()

copied = skipped = 0.0
for p in sorted(CURATED_DIR.glob("*")):
    if not p.is_file():
        continue
    mb = p.stat().st_size / 1e6
    if p.name not in NEEDED:
        skipped += mb
        print(f"  skip  {p.name:<30} {mb:>7.2f} MB  (unused)")
        continue
    shutil.copy2(p, DEST / p.name)
    copied += mb
    flag = "  <-- exceeds GitHub's 50 MB warning" if mb > 50 else ""
    print(f"  copy  {p.name:<30} {mb:>7.2f} MB{flag}")

print(f"\nCopied {copied:.1f} MB to {DEST}")
print(f"Skipped {skipped:.1f} MB of unused OHLC series")
if copied > 100:
    print("WARNING: over 100 MB. Use Git LFS or shorten the sample period.")
