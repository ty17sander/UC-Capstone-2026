# Signal Research Platform

A platform for evaluating cross-sectional equity trading signals under point-in-time
data, realistic transaction costs, and multiple-testing correction.

**M.S. Business Analytics capstone · Tynan Sander**

---

## The headline result

Across **613 companies**, **eleven years** (2015–2026), **seven signals**, and **four
horizons** — 44 specifications in total — **no result survived correction for the number
of tests performed.** A market sentiment index built alongside it tested as *coincident*
with recent returns rather than predictive, and adds no value as a regime filter.

This is the expected outcome, not a failed project. The deliverable is the platform and
its validation methodology; whether a given signal survives is a finding. Large-cap U.S.
equities are the most heavily arbitraged segment of the market, and the replication
literature consistently finds published anomalies decay after publication.

The harder thing to earn is trust in a *null* result. That requires showing the measuring
instrument works — see [Validating the instrument](#validating-the-instrument).

---

## Quick start

```bash
pip install -r requirements.txt

python run_research.py --live     # universe + prices + signal evaluation (~30 min)
python run_sentiment.py --live    # sentiment index (~2 min)
python run_ablation.py --live     # sentiment ablation (~2 min)

streamlit run app.py              # interactive dashboard
jupyter notebook research_notebook.ipynb
```

Everything after the first ingestion reads a local Parquet cache and runs in seconds.

---

## Architecture

```
signal_platform/
├── config.py              Every tunable assumption, in one auditable file
├── data/
│   ├── cache.py           Parquet store with a DuckDB SQL surface
│   ├── universe.py        Point-in-time membership via Wikipedia revisions
│   ├── prices.py          Ingestion, validation, cleaning
│   └── synthetic.py       Data with a known planted signal, for validation
├── signals/
│   ├── base.py            The interface every signal implements
│   └── library.py         Seven price/volume signals across six families
├── research/
│   ├── transforms.py      Winsorize → sector-neutralize → z-score
│   ├── evaluation.py      IC, IC decay, quantiles, turnover, correlation
│   └── backtest.py        Long-short backtest, purged walk-forward
└── sentiment/
    └── index.py           Five-component index, causal normalization, bootstrap

app.py                     Streamlit dashboard
research_notebook.ipynb    Interactive analysis with charts
diagnose.py                Data-quality diagnostics
preregistered.py           Pre-registered hypothesis tests
tests/test_harness.py      13 tests — all passing
```

---

## The central design decision

**Signal evaluation is separated from portfolio backtesting.**

A backtest returns one noisy number blending signal quality, portfolio construction, and
cost assumptions. When it disappoints you cannot tell which failed. The evaluation harness
isolates signal quality and answers five questions about any signal:

| Metric | Question |
|---|---|
| Information coefficient | Does it predict? |
| IC decay across horizons | Over what holding period? |
| Quantile monotonicity | Is the effect pervasive, or an outlier artifact? |
| Turnover → cost drag | What will it cost to trade? |
| Cross-signal correlation | Does it tell me anything the others don't? |

The last matters most. Per Grinold's Fundamental Law (IR ≈ IC × √breadth), a signal
correlated 0.9 with an incumbent adds almost nothing regardless of standalone strength.

### Adding a signal

```python
class MyIdea(Signal):
    family = "mean_reversion"
    rationale = "Why this should work, stated before testing it."

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -(data.close / data.close.rolling(10).mean() - 1)
```

Evaluation, correlation, combination, and backtesting all work immediately. The
`rationale` field is required by convention — a signal you cannot justify *before*
testing is a data-mining result waiting to happen.

---

## Validating the instrument

A flat league table admits two readings: the signals have no edge, or the harness is
broken. Distinguishing them requires establishing both **power** and **correct size**.

```python
def test_recovers_planted_signal():
    # synthetic data with a known embedded effect — the harness must find it
    assert stats["mean_ic"] > 0.02 and stats["t_stat"] > 3.0

def test_finds_nothing_in_noise():
    # identical generator, effect removed — it must NOT find what is absent
    assert abs(stats["mean_ic"]) < 0.02 and abs(stats["t_stat"]) < 3.0
```

Both pass: IC 0.376 at t = 147 with the effect planted, IC 0.006 at t = 2.15 without.
Eleven further tests cover lookahead alignment, sector demeaning, dollar neutrality, cost
monotonicity, and embargo enforcement. Synthetic runs are byte-identical across operating
systems.

---

## Guards against standard failure modes

| Failure mode | Guard |
|---|---|
| Survivorship bias | Point-in-time membership from quarterly Wikipedia revisions; delisted names retained |
| Lookahead bias | One-day execution lag enforced in `forward_returns`; unit-tested |
| Restated fundamentals | Library restricted to price/volume — unambiguous timestamps |
| Overlapping-return t-stats | Newey-West HAC standard errors throughout |
| Train/test leakage | Purged walk-forward, 21-day embargo |
| Multiple testing | Deflated Sharpe hurdle scaled by cumulative trial count |
| Unmodeled costs | 10 bps/side on realized turnover; gross and net both reported |
| Unintended sector bets | Signals demeaned within GICS sector |
| Anti-conservative tests | Block-bootstrap null calibration (below) |

---

## A test that was wrong, and how it was caught

The predictive regression for the sentiment index declared a significant relationship
**on synthetic data containing no relationship by construction** (t = −3.04).

Across forty independent synthetic datasets, the conventional test produced |t| > 2 in
**8–20% of cases** rather than the nominal 5%. The cause is Stambaugh bias: the index is
highly persistent and partly derived from the same price series whose future it predicts.
Newey-West corrects for overlapping returns — not for this.

A block-bootstrap null was implemented:

| | Empirical | Theoretical |
|---|---|---|
| Std dev of t | 1.250 | 1.000 |
| 95th pct of \|t\| | 2.496 | 1.960 |
| 99th pct of \|t\| | 3.194 | 2.576 |
| p-value, observed result | **0.073** | 0.026 |

The naive test would have reported a finding. The calibrated test correctly does not.

---

## Point-in-time universe construction

The most common fatal error in retrospective equity research is building a universe from
*current* index membership. That silently guarantees you only ever hold survivors.

Membership here is reconstructed from historical revisions of the Wikipedia S&P 500
constituents page, sampled quarterly via the MediaWiki API. Each snapshot is a direct
observation rather than an inference, and carries an auditable revision ID — anyone can
retrieve the exact page version used.

**779 distinct companies** passed through the index over eleven years against ~503 at any
single date. That 276-name gap is the survivorship bias being removed.

---

## Known limitations

- **164 companies unavailable** from the price vendor, disproportionately failures and
  acquisitions — reintroduces some survivorship bias; the largest residual data limitation
- **Quarterly membership sampling** dates index changes only to within a quarter
- **Sectors are first-observed**, so GICS reclassifications are not tracked through time
- **Dollar-neutral is not beta-neutral**
- **No borrow-cost or market-impact model**; the liquidity screen is a proxy
- **Two sentiment components omitted** (put/call, McClellan volume); credit uses an ETF
  proxy rather than the option-adjusted spread
- **Market timing has little breadth** — ~44 independent observations at a 63-day horizon
  — so negative results there have limited power. *Not significant* ≠ *no effect*
- **Free data is fragile.** Three scraper breakages occurred during development (HTTP 403,
  header parsing, source table removal). Pipelines built on free sources decay.

---

## Disclaimer

Academic research, not investment advice. The research found no statistically significant
edge. Nothing here should be used to make investment decisions.

## License

MIT
