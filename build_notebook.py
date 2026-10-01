"""Build the research notebook.

Generating the .ipynb from a script keeps the notebook's source under
version control as plain Python rather than as JSON with embedded
outputs, and makes it regenerable when the underlying API changes.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Signal Research Platform — Interactive Research Notebook

**Tynan Sander · M.S. Business Analytics Capstone**

This notebook runs the full research pipeline with visible output at every
stage. It reads from the local Parquet cache, so it executes in seconds
once data has been ingested — re-run any cell to see results update.

**Before first use**, populate the cache by running `python run_research.py --live`
from the project root. Afterwards this notebook is fully offline.

---
### Contents
1. Setup and cache inspection
2. Universe — survivorship bias made visible
3. Price data quality
4. Signal computation
5. Evaluation — the league table
6. Signal correlation
7. IC decay
8. Quantile returns
9. Backtest and equity curve
10. Walk-forward validation
11. Parameter sandbox
""")

md("## 1. Setup and cache inspection")

code("""
import warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")

# Locate the project root whether this notebook sits in the root or a subfolder
_root = Path.cwd()
while not (_root / "signal_platform").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from signal_platform.data import cache
from signal_platform.research import evaluation as ev
from signal_platform.research import transforms as tf
from signal_platform.research.backtest import (
    build_positions, run_backtest, walk_forward_report, performance_stats)
from signal_platform.signals import MarketData, default_registry

# Plot styling — consistent across the notebook
mpl.rcParams.update({
    "figure.figsize": (11, 4.5), "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})
NAVY, RED, GREY = "#1F3864", "#C0392B", "#7F8C8D"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

print("What is currently cached:\\n")
display(cache.summary())
""")

md("""
The cache manifest records row counts and age in hours. If `age_hours` is
large, re-run the ingestion step to refresh.
""")

md("## 2. Universe — survivorship bias made visible")

code("""
membership = cache.read("universe_sp500")
print(f"Membership matrix: {membership.shape[0]:,} dates x {membership.shape[1]} tickers")

per_date = membership.sum(axis=1)
ever = membership.any(axis=0).sum()

fig, ax = plt.subplots()
ax.plot(per_date.index, per_date.values, color=NAVY, lw=1.4,
        label="Members on this date")
ax.axhline(ever, color=RED, ls="--", lw=1.4,
           label=f"Distinct companies ever a member ({ever})")
ax.fill_between(per_date.index, per_date.values, ever, alpha=0.08, color=RED)
ax.set_title("Index membership over time")
ax.set_ylabel("Number of companies")
ax.legend(loc="center right", frameon=False)
plt.tight_layout(); plt.show()

gap = ever - int(per_date.median())
print(f"\\nMedian members on any date : {int(per_date.median())}")
print(f"Distinct companies ever    : {ever}")
print(f"Survivorship gap           : {gap} companies "
      f"({gap / ever:.0%} of the universe)")
print("\\nThe shaded region is the bias a naive backtest silently removes:")
print("companies that were acquired, went bankrupt, or left the index.")
""")

md("## 3. Price data quality")

code("""
close_raw = cache.read("px_close")
volume_raw = cache.read("px_volume")

from signal_platform.data.prices import clean_prices
close, volume, report = clean_prices(close_raw, volume_raw, verbose=False)

print(f"Raw columns                    : {close_raw.shape[1]}")
print(f"  dropped, entirely empty      : {len(report['empty_columns'])}")
print(f"  dropped, fewer than 252 obs  : {len(report['thin_columns'])}")
print(f"  bad prints masked            : {report['bad_prints']}")
print(f"Final usable universe          : {report['n_tickers_final']}")

if report["bad_print_tickers"]:
    print("\\nTickers with suspected unadjusted corporate actions:")
    for tk, n in list(report["bad_print_tickers"].items())[:10]:
        print(f"  {tk:<6} {n} bad print(s)")
""")

code("""
coverage = close.notna().sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].plot(coverage.index, coverage.values, color=NAVY, lw=1.2)
axes[0].set_title("Names with a price each day")
axes[0].set_ylabel("Count")

rets = close.pct_change().stack()
axes[1].hist(rets[rets.abs() < 0.15], bins=120, color=NAVY, alpha=0.8)
axes[1].set_yscale("log")
axes[1].set_title("Daily return distribution (log scale, trimmed at ±15%)")
axes[1].set_xlabel("Daily return")
plt.tight_layout(); plt.show()

print(f"Pooled daily volatility: {close.pct_change().stack().std():.4f}")
print(f"Annualised equal-weight universe return: "
      f"{((1 + close.pct_change().mean(axis=1).fillna(0)).prod() ** (252/len(close)) - 1):.2%}")
""")

md("## 4. Signal computation")

code("""
from signal_platform.data import universe as uni

sectors = uni.sector_map()
mem = membership.reindex(index=close.index, columns=close.columns).fillna(False)
mem = uni.apply_liquidity_filter(mem, close, volume,
                                 min_dollar_volume=5_000_000.0, min_price=5.0)

data = MarketData(close=close, volume=volume, sectors=sectors, membership=mem)

cov = sectors.reindex(close.columns).notna().mean()
print(f"Sector coverage: {cov:.0%}")
if cov < 0.9:
    print("WARNING: below 90%. Run universe.rebuild_sector_map_from_snapshots()")

registry = default_registry()
display(registry.describe())
""")

code("""
raw = registry.compute_all(data)
prepared = {n: tf.prepare(f, sectors, neutralize=True) for n, f in raw.items()}

print(f"{len(prepared)} signals computed.\\n")
print("Cross-section size after all masking:")
for name, f in prepared.items():
    n = f.notna().sum(axis=1)
    print(f"  {name[:38]:<40} median {int(n.median()):>4} names")
""")

md("## 5. Evaluation — the league table")

code("""
HORIZON = 5   # change and re-run to see results update

table, details = ev.evaluate_registry(
    prepared, close, primary_horizon=HORIZON, cost_bps=10.0)

cols = ["mean_ic", "ic_t_stat", "ic_ir", "monotonicity",
        "ann_spread_gross", "ann_cost_drag", "ann_spread_net", "avg_turnover"]
display(table[cols].style.format("{:.4f}")
        .background_gradient(subset=["ic_t_stat"], cmap="RdYlGn", vmin=-3, vmax=3))
""")

code("""
fig, ax = plt.subplots(figsize=(10, 4))
t = table["ic_t_stat"].sort_values()
colors = [RED if abs(v) < 2 else NAVY for v in t.values]
ax.barh([n[:32] for n in t.index], t.values, color=colors, alpha=0.85)
for x, lbl, c in [(2, "|t| = 2", GREY), (-2, None, GREY),
                  (3, "|t| = 3 (multiple-testing hurdle)", RED), (-3, None, RED)]:
    ax.axvline(x, color=c, ls="--", lw=1.1, label=lbl)
ax.set_xlabel("IC t-statistic (Newey-West)")
ax.set_title(f"Signal significance at {HORIZON}-day horizon")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

md("## 6. Signal correlation")

code("""
corr = ev.signal_correlation(prepared)
short = {n: n.split("_")[0][:18] for n in corr.columns}
c = corr.rename(index=short, columns=short)

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
for i in range(len(c)):
    for j in range(len(c)):
        v = c.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.55 else "black")
ax.set_title("Average cross-sectional correlation")
plt.colorbar(im, shrink=0.8); ax.grid(False)
plt.tight_layout(); plt.show()

print("Pairs above 0.80 — redundant, drop one:")
found = False
for i, a in enumerate(corr.columns):
    for b in corr.columns[i+1:]:
        if abs(corr.loc[a, b]) > 0.8:
            print(f"  {a[:34]:<36} <-> {b[:34]:<36} {corr.loc[a, b]:.4f}")
            found = True
if not found:
    print("  none")
""")

md("## 7. IC decay — what is the natural holding period?")

code("""
HORIZONS = (1, 5, 21, 63)

decay = {}
for name, sig in prepared.items():
    decay[name] = ev.ic_decay(sig, close, horizons=HORIZONS)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for name, d in decay.items():
    lbl = name.split("_")[0][:20]
    axes[0].plot(d.index, d["mean_ic"], marker="o", ms=4, lw=1.4, label=lbl)
    axes[1].plot(d.index, d["t_stat"], marker="o", ms=4, lw=1.4, label=lbl)

axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_title("Mean IC by horizon"); axes[0].set_xlabel("Horizon (days)")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].axhline(3, color=RED, ls="--", lw=1); axes[1].axhline(-3, color=RED, ls="--", lw=1)
axes[1].set_title("t-statistic by horizon (red = hurdle)")
axes[1].set_xlabel("Horizon (days)")
axes[1].legend(fontsize=7, frameon=False, ncol=2)
plt.tight_layout(); plt.show()

grid = pd.DataFrame({n.split("_")[0][:22]: d["t_stat"] for n, d in decay.items()}).T
grid.columns = [f"{h}d" for h in HORIZONS]
print("IC t-statistic grid:"); display(grid.style.format("{:.2f}")
    .background_gradient(cmap="RdYlGn", vmin=-3, vmax=3))
print(f"\\nCells tested: {grid.size}.  Cells with |t| > 3: "
      f"{int((grid.abs() > 3).sum().sum())}")
""")

md("## 8. Quantile returns — is the effect monotone?")

code("""
PICK = table.index[0]     # change to inspect any signal

qret = details[PICK]["quantile_returns"]
qsum = details[PICK]["quantile_summary"]
buckets = [c for c in qsum.index if c.startswith("Q")]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
vals = qsum.loc[buckets, "ann_mean"]
axes[0].bar(buckets, vals.values,
            color=[RED if v < 0 else NAVY for v in vals.values], alpha=0.85)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_title(f"Annualised return by quantile\\n{PICK[:40]}")
axes[0].set_ylabel("Annualised return")

eq = (1 + qret["spread"].fillna(0)).cumprod()
axes[1].plot(eq.index, eq.values, color=NAVY, lw=1.4)
axes[1].axhline(1, color=GREY, ls="--", lw=1)
axes[1].set_title("Cumulative top-minus-bottom spread (gross)")
plt.tight_layout(); plt.show()

print(f"Monotonicity: {qsum.attrs.get('monotonicity', float('nan')):.3f}")
print("(1.00 = perfectly increasing across buckets; near 0 = no ordering)")
""")

md("## 9. Backtest and equity curve")

code("""
composite = tf.combine(prepared)
res = run_backtest(composite, close, rebalance_freq="W-FRI", cost_bps=10.0)

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [2.4, 1]})
eq_net = (1 + res["returns_net"].fillna(0)).cumprod()
eq_gross = (1 + res["returns_gross"].fillna(0)).cumprod()
axes[0].plot(eq_gross.index, eq_gross.values, color=GREY, lw=1.2, label="Gross")
axes[0].plot(eq_net.index, eq_net.values, color=NAVY, lw=1.5, label="Net of costs")
axes[0].axhline(1, color="k", lw=0.8)
axes[0].set_title("Equal-weighted composite — long-short market-neutral")
axes[0].set_ylabel("Growth of $1"); axes[0].legend(frameon=False)

dd = eq_net / eq_net.cummax() - 1
axes[1].fill_between(dd.index, dd.values, 0, color=RED, alpha=0.4)
axes[1].set_ylabel("Drawdown"); axes[1].set_title("Drawdown")
plt.tight_layout(); plt.show()

display(pd.DataFrame({"gross": res["stats_gross"], "net": res["stats"]}))
print(f"\\nAnnual cost drag: "
      f"{res['stats_gross']['ann_return'] - res['stats']['ann_return']:.2%}")
""")

md("## 10. Walk-forward validation")

code("""
wf = walk_forward_report(composite, close, n_splits=4, embargo_days=21)
display(wf[["test_start", "test_end", "ann_return", "ann_vol",
            "sharpe", "max_drawdown"]])

fig, ax = plt.subplots(figsize=(9, 3.6))
colors = [NAVY if v > 0 else RED for v in wf["sharpe"]]
ax.bar(wf.index.astype(str), wf["sharpe"], color=colors, alpha=0.85)
ax.axhline(0, color="k", lw=0.9)
ax.set_title("Out-of-sample Sharpe by fold — consistency is the finding")
ax.set_xlabel("Fold"); ax.set_ylabel("Sharpe ratio")
plt.tight_layout(); plt.show()

pos = int((wf["sharpe"] > 0).sum())
print(f"Folds positive: {pos} of {len(wf)}")
print("Four of four would be a signal. Two of four is a coin flip.")
""")

md("""
## 11. Parameter sandbox

Change any value below and re-run to see the full effect. The most
instructive experiment is `TRADE_LAG`: setting it to 0 lets the strategy
trade on the same bar it observed, which is impossible in practice. The
difference between 0 and 1 measures directly how much lookahead bias
would have inflated the results.
""")

code("""
TRADE_LAG_TEST = [0, 1]      # 0 = leaks future information
NEUTRALIZE_TEST = [True, False]

rows = []
for lag in TRADE_LAG_TEST:
    for neut in NEUTRALIZE_TEST:
        prep = {n: tf.prepare(f, sectors, neutralize=neut) for n, f in raw.items()}
        d = ev.ic_decay(prep[table.index[0]], close, horizons=(5,), lag=lag)
        rows.append({"trade_lag": lag, "sector_neutral": neut,
                     "mean_ic": d.loc[5, "mean_ic"], "t_stat": d.loc[5, "t_stat"]})

sandbox = pd.DataFrame(rows)
print(f"Signal under test: {table.index[0]}\\n")
display(sandbox)

leak = sandbox[(sandbox.trade_lag == 0) & (sandbox.sector_neutral)]["t_stat"].iloc[0]
real = sandbox[(sandbox.trade_lag == 1) & (sandbox.sector_neutral)]["t_stat"].iloc[0]
print(f"\\nt-statistic with lookahead (lag=0): {leak:.2f}")
print(f"t-statistic without      (lag=1): {real:.2f}")
print(f"Inflation attributable to lookahead: {leak - real:+.2f}")
""")

md("""
---

### Interpreting what you see

| Observation | Reading |
|---|---|
| IC between 0.01 and 0.05 | Plausible range for real equity signals |
| IC above 0.10 on live data | Almost certainly a bug — check for lookahead |
| \\|t\\| below 2 | Not evidence of anything |
| Gross positive, net negative | Real but untradeable at this frequency |
| Monotonicity below 0.7 | Outlier effect, not a pervasive one |
| Net Sharpe above 2.0 | A bug, not a discovery |
| Folds disagreeing in sign | One regime, not an edge |

Every result here is conditional on the number of specifications tested.
Keep a running count and report it.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("/home/claude/sigplat/research_notebook.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"notebook written: {len(cells)} cells")
