"""Central configuration for the signal research platform.

Every tunable assumption lives here so that a reviewer can audit the
methodology in one file instead of hunting through the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
# DATA_DIR can be pointed at a mounted volume / object-store cache in
# deployment without touching code.
# Resolution order:
#   1. SIGPLAT_DATA_DIR environment variable
#   2. a repo-local ./data directory, if present -- this is how a deployed
#      app (Streamlit Cloud, which has ephemeral storage and cannot run a
#      30-minute ingestion on startup) finds data committed alongside the code
#   3. ~/.sigplat for local development
_REPO_DATA = Path(__file__).resolve().parent.parent / "data"
if "SIGPLAT_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["SIGPLAT_DATA_DIR"])
elif (_REPO_DATA / "curated").exists():
    DATA_DIR = _REPO_DATA
else:
    DATA_DIR = Path.home() / ".sigplat"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"

for _d in (RAW_DIR, CURATED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# SPDR sector ETFs. These have no survivorship bias -- they are not
# delisted, acquired, or renamed. Inception dates matter: XLRE and XLC
# are late arrivals and must not appear in backtests before they existed.
SECTOR_ETFS: dict[str, str] = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

SECTOR_ETF_INCEPTION: dict[str, str] = {
    "XLB": "1998-12-22", "XLE": "1998-12-22", "XLF": "1998-12-22",
    "XLI": "1998-12-22", "XLK": "1998-12-22", "XLP": "1998-12-22",
    "XLU": "1998-12-22", "XLV": "1998-12-22", "XLY": "1998-12-22",
    "XLRE": "2015-10-08", "XLC": "2018-06-19",
}

# Wikipedia is the free source of point-in-time S&P 500 membership.
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Benchmarks and sentiment-component proxies.
MARKET_PROXY = "SPY"
BOND_PROXY = "TLT"          # safe-haven demand
JUNK_PROXY = "HYG"          # junk bond demand
IG_PROXY = "LQD"            # investment grade, paired with HYG

# --------------------------------------------------------------------------
# Research defaults
# --------------------------------------------------------------------------
# Signals are computed on the close of day t. The earliest realistic
# execution is the close of day t+1. TRADE_LAG enforces that gap and is
# the single most important guard against lookahead bias in this codebase.
TRADE_LAG = 1

FORWARD_HORIZONS = (1, 5, 21)      # trading days
REBALANCE_FREQ = "W-FRI"           # weekly, per project scope
N_QUANTILES = 5
WINSOR_LIMITS = (0.01, 0.99)       # cross-sectional, applied pre-z-score

# Cost model. 10bps per side is a defensible round-number assumption for
# liquid US large caps; document it rather than pretending it is precise.
COST_BPS_PER_SIDE = 10.0

# Newey-West lag for IC t-statistics. Overlapping forward returns induce
# autocorrelation; naive t-stats are badly overstated without this.
def newey_west_lags(horizon: int) -> int:
    """Standard choice: one fewer lag than the overlap length."""
    return max(int(horizon) - 1, 0)

# Liquidity floor for the shortable universe. Shorting illiquid names in
# a backtest assumes borrow that may not have existed.
MIN_DOLLAR_VOLUME = 5_000_000.0
MIN_PRICE = 5.0
