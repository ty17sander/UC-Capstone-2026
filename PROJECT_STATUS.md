# Signal Research Platform — Project Status

**Author:** Tynan Sander
**Program:** M.S. Business Analytics
**Last updated:** September 2026

---

## 1. What this project is

A platform for evaluating cross-sectional equity trading signals under a
uniform, honest methodology. It ingests point-in-time market data,
measures whether a signal predicts forward returns, and reports whether
any edge survives transaction costs and multiple-testing adjustment.

**The deliverable is the platform and the methodology, not a profitable
strategy.** Whether any particular signal survives evaluation is a
finding. This framing is deliberate: a capstone claiming to have found
reliable alpha invites — and usually fails — exactly the scrutiny it
should attract, whereas a rigorous evaluation framework that honestly
reports null results is defensible regardless of outcome.

### How the scope was narrowed

The project began as two separate ambitions: a market-wide sentiment
index (a Fear & Greed analogue) and a market-wide stock-selection model.
Each is a full capstone. They were unified into a single three-layer
architecture with one testable thesis:

| Layer | Question it answers | Status |
|---|---|---|
| Sentiment | How much risk to take | Not yet built |
| Sector | Remove unintended factor bets | Built |
| Stock selection | Which names to hold long/short | Built |

Two further scope decisions:

- **Long-short market-neutral**, not long-only. Isolates signal quality
  from market direction; a long-only backtest over 2015–2026 would be
  dominated by the bull market regardless of signal quality.
- **Weekly rebalance**, price/volume signals only. Free fundamental data
  (yfinance included) reports *current, restated* financials with no
  as-reported timestamp, so using it historically is lookahead bias.
  Price and volume carry unambiguous timestamps.

---

## 2. What has been built

```
signal_platform/
├── config.py              All tunable assumptions in one auditable file
├── data/
│   ├── cache.py           Parquet store + DuckDB SQL surface
│   ├── universe.py        Point-in-time membership via Wikipedia revisions
│   ├── prices.py          yfinance ingestion, validation, cleaning
│   └── synthetic.py       Known-signal data for validating the harness
├── signals/
│   ├── base.py            The Signal interface every signal implements
│   └── library.py         Seven price/volume signals across six families
└── research/
    ├── transforms.py      Winsorize → sector-neutralize → z-score
    ├── evaluation.py      IC, IC decay, quantiles, turnover, correlation
    └── backtest.py        Long-short backtest, purged walk-forward
diagnose.py                Data-quality diagnostics
run_research.py            End-to-end research run
tests/test_harness.py      13 tests, all passing
```

### The central design decision

**Signal evaluation is separated from portfolio backtesting.** A backtest
produces one number blending signal quality, portfolio construction, and
cost assumptions; when it disappoints you cannot tell which failed. The
evaluation harness isolates signal quality and reports:

1. **Information coefficient** — rank correlation of score vs forward
   return, with Newey-West standard errors
2. **IC decay** across 1/5/21-day horizons — reveals natural holding period
3. **Quantile spread and monotonicity** — is the effect pervasive or an
   outlier artifact?
4. **Turnover** — converted to annual cost drag
5. **Cross-signal correlation** — does a new signal add breadth?

Item 5 matters most. Per Grinold's Fundamental Law
(IR ≈ IC × √breadth), a signal correlated 0.9 with an incumbent adds
almost nothing regardless of standalone IC.

### Adding a signal

```python
class MyIdea(Signal):
    family = "mean_reversion"
    rationale = "Why this should work, stated before testing it."

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -(data.close / data.close.rolling(10).mean() - 1)
```

Everything downstream works immediately. The `rationale` field is
required by convention — a signal that cannot be justified before testing
is a data-mining result waiting to happen.

### Validating the harness itself

Two tests matter more than the rest:

- `test_recovers_planted_signal` — synthetic data with a known embedded
  effect; the harness must find it
- `test_finds_nothing_in_noise` — same data, effect removed; the harness
  must report ~zero

Together these establish **power** and **size**. Without both, one cannot
distinguish "the harness works and the signal is weak" from "the harness
is broken."

---

## 3. Methodological guards

| Failure mode | Guard implemented |
|---|---|
| Survivorship bias | Point-in-time membership from 36 quarterly Wikipedia revisions; delisted names retained |
| Lookahead bias | `TRADE_LAG = 1` enforced in `forward_returns`; unit-tested |
| Restated fundamentals | Library restricted to price/volume |
| Overlapping-return t-stats | Newey-West HAC standard errors on all IC series |
| Train/test leakage | Purged walk-forward, 21-day embargo |
| Multiple testing | Approximate deflated-Sharpe hurdle scaled by trial count |
| Unmodeled costs | 10bps/side on realized turnover; gross and net both reported |
| Unintended sector bets | Signals demeaned within GICS sector |
| Unshortable names | Liquidity screen: $5M ADV, $5 minimum price |
| Bad vendor prints | Moves >100% masked; empty and thin columns dropped |

---

## 4. Data pipeline as built

**Universe.** Historical revisions of the Wikipedia S&P 500 constituents
page, sampled quarterly, fetched via the MediaWiki revision API. Each
snapshot is a direct observation of index membership on that date rather
than an inference from a change log.

This approach was adopted after the original change-log method failed:
Wikipedia removed the change table from the page entirely. Reading
revisions of the constituent table proved both more robust and more
accurate, and it yields auditable revision IDs — any reader can open
`en.wikipedia.org/w/index.php?oldid=<rev>` and see the exact data used.

**Result:** 2015–2026, 36 snapshots, ~503 constituents each,
**779 distinct companies** across the window. The 276-name gap between
779 and 503 is the survivorship bias being removed.

**Prices.** yfinance, split- and dividend-adjusted, batched with retry.

**After cleaning: 613 usable tickers.** 150 columns were entirely empty
(yfinance has no data for long-delisted names), 16 were too thin, and 8
bad prints were masked.

**Sectors.** GICS sector captured from each historical revision, so
delisted companies retain the sector they had while they were members.
Coverage: 100%.

---

## 5. Results to date

### The three bugs found (and fixed)

The first live run produced a flat league table. Diagnostics showed this
was measuring the pipeline, not the market:

1. **Unmapped sector bucket.** `sector_map()` covered only the ~503
   *current* constituents, so 276 delisted names were pooled into one
   `_UNMAPPED` pseudo-sector — 35% of the cross-section. Demeaning
   unrelated companies against each other injected noise at precisely the
   step meant to remove it. Sector neutralization was destroying 30–55%
   of every signal's IC.
2. **150 all-empty columns** counted as successful downloads, inflating
   apparent universe size and quietly reintroducing survivorship bias.
3. **Unadjusted corporate actions**, including a 350% single-day move in
   STI (SunTrust/Truist merger).

Cleaning moved the gross composite return from −1.18% to −0.38% annually.
**Roughly two-thirds of the apparent loss was artifact.**

### Current results (clean data, 613 tickers, 2015–2026)

| Signal | mean IC | t-stat | monotonicity | net spread | turnover |
|---|---|---|---|---|---|
| ShortTermReversal | 0.0057 | 1.17 | 1.00 | +2.6% | 0.295 |
| VolumeShock | 0.0025 | 1.17 | −0.40 | −2.3% | 0.344 |
| TrendStrength | 0.0063 | 0.97 | −0.20 | −0.2% | 0.109 |
| IdiosyncraticMomentum | 0.0061 | 0.92 | −0.10 | −1.7% | 0.081 |
| Momentum12_1 | 0.0048 | 0.72 | 0.00 | −2.8% | 0.080 |
| LowVolatility | 0.0021 | 0.34 | −0.70 | −4.6% | 0.053 |
| LiquidityTurnover | −0.0073 | **−2.31** | **−1.00** | −4.6% | 0.010 |

Composite (equal-weight): gross Sharpe −0.05, net −0.39. All four
walk-forward folds negative. Observed Sharpe does not clear the
multiple-testing hurdle of 0.41.

### Interpretation

**This is a null result, and a defensible one.** Six of seven signals
have |t| < 1.2 — indistinguishable from zero. The composite is
approximately flat gross and negative net because costs take ~2.6%/yr.
That is a clean story: no detectable edge, and transaction costs do the
rest.

**Two findings that survive scrutiny:**

*Momentum12_1 and IdiosyncraticMomentum correlate at 0.998.* Subtracting
the market return before measuring momentum is redundant once scores are
cross-sectionally demeaned — the demeaning already removes the common
factor. This held on both synthetic and live data. One should be dropped.

*LiquidityTurnover is the only |t| > 2, with perfect −1.00 monotonicity.*
But its turnover is 0.0098 — the book barely changes over eleven years.
It is a nearly static position: long low-volume names, short high-volume
ones, which over 2015–2026 approximates shorting mega-cap technology.
Effective breadth is therefore close to 1, not 613, and a t-stat computed
as though each name were independent is badly overstated. Newey-West
corrects for overlapping returns, not for the positions being the same
trade every day. **Provisional reading: a size/mega-cap factor exposure,
not a liquidity anomaly.**

---

## 6. What remains

### Immediate (next two weeks)

- [ ] **Test whether LiquidityTurnover is size.** Regress its scores
      against log market cap. If highly correlated, it is a known factor,
      not a discovery — still a legitimate result, but a different claim.
- [ ] **Re-run at 21- and 63-day horizons.** The library is
      momentum-heavy and momentum does not live at 5 days.
- [ ] **Drop one of the two redundant momentum signals.**
- [ ] **Walk-forward sign determination** for the composite, using only
      prior-fold data. Setting signs after seeing the full sample is
      exactly the p-hacking the deflated-Sharpe section guards against.

### Core remaining work

- [ ] **Sentiment layer.** Construct the composite index from FRED
      (VIX, high-yield OAS), CBOE put/call, and ETF relative returns.
      Validate the reconstruction against CNN's published Fear & Greed
      values over the overlapping period — a rare ground-truth check.
- [ ] **Wire sentiment into `exposure_scaler` and ablate it.** Run with
      and without; without an ablation, no claim can be made that it
      contributed.
- [ ] **Expand the signal library** beyond momentum-adjacent families —
      seasonality, dispersion, cross-sectional beta.
- [ ] **Beta-neutral construction.** The book is dollar-neutral but can
      carry residual market beta.
- [ ] **Streamlit dashboard** over the cached parquet.
- [ ] **Deploy:** GitHub → Actions weekly refresh → Streamlit Cloud.

### Writeup

- [ ] Methodology chapter (largely written as code documentation)
- [ ] Bug-discovery narrative — the diagnostic trail is strong evidence
      of rigor and should be presented, not hidden
- [ ] Limitations section (see below)
- [ ] Lookahead-bias demonstration: run with `TRADE_LAG = 0` vs `1` and
      report the difference as a direct measurement of the bias avoided

---

## 7. Known limitations

1. **Quarterly membership sampling.** A company joining and leaving
   within one quarter can be missed; changes are dated to within a
   quarter. Immaterial for weekly rebalancing, material for event studies.
2. **150 tickers unavailable from yfinance.** These are disproportionately
   failures and acquisitions, which reintroduces some survivorship bias.
   This is the single largest residual data limitation.
3. **Sectors are first-observed, not point-in-time.** GICS
   reclassifications (notably the 2018 creation of Communication
   Services) are not tracked.
4. **Dollar-neutral ≠ beta-neutral.**
5. **No borrow-cost or market-impact model.** The liquidity screen is a
   proxy for shortability.
6. **Deflated Sharpe is approximate**, indicative rather than a faithful
   implementation of the published method.
7. **Free-data fragility.** Three separate scraper breakages occurred
   during development (HTTP 403, header parsing, table removal). This is
   itself worth reporting as a finding about reproducibility with free
   sources — and an argument for institutional data access (WRDS/CRSP)
   where available.

---

## 8. Reproducibility

- 13/13 tests pass; synthetic runs are byte-identical across machines
  and operating systems
- All universe snapshots carry Wikipedia revision IDs, so the exact input
  data is independently retrievable
- All assumptions centralized in `config.py`
- Data cached as Parquet with a manifest recording write times
