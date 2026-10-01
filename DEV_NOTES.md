# Signal Research Platform

A cross-sectional equity signal research platform: ingest market data,
evaluate arbitrary trading signals under a uniform methodology, and
report honestly whether edge survives transaction costs and
multiple-testing adjustment.

**The deliverable is the platform and the methodology, not a profitable
strategy.** Whether any given signal survives evaluation is a *finding*.
A well-executed null result is a legitimate outcome and is defended as
such throughout.

---

## Architecture

```
signal_platform/
├── config.py              All tunable assumptions in one auditable file
├── data/
│   ├── cache.py           Parquet store + DuckDB SQL surface
│   ├── universe.py        Point-in-time S&P 500 reconstruction
│   ├── prices.py          yfinance ingestion, retry, quality validation
│   └── synthetic.py       Known-signal data for validating the harness
├── signals/
│   ├── base.py            The Signal interface every signal implements
│   └── library.py         Seven price/volume signals across six families
└── research/
    ├── transforms.py      Winsorize → sector-neutralize → z-score
    ├── evaluation.py      IC, IC decay, quantiles, turnover, correlation
    └── backtest.py        Long-short backtest, purged walk-forward
```

Three layers, matching how multi-strategy funds decompose risk:

| Layer | Role | Output |
|---|---|---|
| Sentiment | How much risk to take | Gross exposure scalar |
| Sector | Remove unintended factor bets | Neutralized scores |
| Stock selection | The breadth engine | Cross-sectional ranks |

---

## The central design decision

**Signal evaluation is separated from portfolio backtesting.**

A backtest produces one noisy number that blends signal quality,
portfolio construction, and cost assumptions. When it disappoints you
cannot tell which component failed. The evaluation harness isolates
signal quality and answers four questions about any signal:

1. **Does it predict?** — information coefficient with a Newey-West t-stat
2. **Over what horizon?** — IC decay across 1/5/21 days
3. **Is it monotone?** — quantile spread across buckets
4. **What will it cost?** — turnover, converted to annual drag

And one question about the library as a whole:

5. **Does this add anything?** — pairwise correlation with every incumbent

Question 5 is the one that matters most. Per the Fundamental Law of
Active Management (IR ≈ IC × √breadth), a signal correlated 0.9 with an
incumbent adds almost nothing regardless of its standalone IC.

---

## Adding a signal

Subclass `Signal`, implement `compute`, return higher-is-better scores:

```python
class MyIdea(Signal):
    family = "mean_reversion"
    rationale = "Why this should work, stated before testing it."

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -(data.close / data.close.rolling(10).mean() - 1)

registry.add(MyIdea())
```

Everything else — evaluation, correlation, combination, backtesting —
works immediately. That uniformity is what makes this a platform rather
than a script.

The `rationale` field is required by convention. A signal you cannot
justify *before* testing is a data-mining result waiting to happen.

---

## Guards against the standard failure modes

| Failure mode | Guard |
|---|---|
| Survivorship bias | Point-in-time membership reconstructed from Wikipedia change log; delisted tickers retained |
| Lookahead bias | `TRADE_LAG = 1` enforced in `forward_returns`; unit-tested |
| Fundamental restatement | Library restricted to price/volume — unambiguous timestamps |
| Overlapping-return t-stats | Newey-West HAC standard errors on all IC series |
| Train/test leakage | Purged walk-forward with 21-day embargo |
| Multiple testing | Approximate deflated-Sharpe hurdle scaled by trial count |
| Unmodeled costs | 10bps/side charged on realized turnover, gross and net both reported |
| Unintended sector bets | Signals demeaned within GICS sector before combination |
| Unshortable names | Liquidity screen: min $5M ADV, min $5 price |

---

## Validating the harness itself

`tests/test_harness.py` includes two tests that matter more than the rest:

- `test_recovers_planted_signal` — synthetic data with a known embedded
  effect; the harness must find it
- `test_finds_nothing_in_noise` — identical data with the effect removed;
  the harness must report approximately zero

Together these establish **power** and **size**. Without both, you cannot
distinguish "the harness works and this signal is weak" from "the harness
is broken." A harness that finds signal in noise is worse than no harness.

All 13 tests pass. Run: `PYTHONPATH=. python tests/test_harness.py`

---

## Reading the demo output

`python run_research.py` runs on synthetic data offline. Three things in
that output are worth understanding, because they are exactly the kind of
finding the platform exists to surface:

**Momentum12_1 and IdiosyncraticMomentum correlate at 1.00.** Subtracting
the market return before measuring momentum is redundant once scores are
cross-sectionally demeaned — the demeaning already removes the common
factor. One of the two should be dropped. This is a real methodological
finding produced by the correlation table, and precisely why that table
exists.

**The equal-weighted composite loses money** even though several
components have strong standalone ICs. ShortTermReversal is sharply
wrong-signed in this data and drags the composite down. Equal weighting
is a strong baseline but not an unconditional one — components should
clear a sign and significance screen first.

**The walk-forward folds disagree** (two mildly positive, two negative).
That inconsistency *is* the finding. A signal that works in one fold and
fails in three worked in one regime, which is a very different claim from
a signal that works.

Note that synthetic-data ICs (0.12, t=34) are wildly unrealistic by
construction. Real equity signals live around IC 0.01–0.05.

---

## Deployment

Requirements: multiple users, always-current, no ongoing cost.

```
GitHub repo (source of truth)
    │
    ├── GitHub Actions ── weekly cron, Sat 06:00 UTC
    │       ingest → evaluate → validate → publish parquet
    │
    └── Streamlit Community Cloud ── auto-redeploys on push
            free, public URL, shareable with committee + teammates
```

**Why this stack:** Streamlit Community Cloud is free, connects directly
to a GitHub repo, redeploys on every push, and gives a URL anyone can
open with no install. You have already built Streamlit apps, so there is
no learning curve. GitHub Actions provides free scheduled compute on
public repos, so the data refreshes without anyone remembering to run it.

**Critical detail in the workflow:** the test suite runs *after*
ingestion and *before* publishing. If a yfinance schema change corrupts
the data, the run fails and the previous good artifacts stay live. A
refresh pipeline with no validation gate will eventually publish a week
of NaNs and nobody will notice for a month.

For datasets beyond ~100MB, swap the artifact upload for Cloudflare R2
(free egress) or GitHub Release assets. Point `SIGPLAT_DATA_DIR` at the
mount; no code changes needed.

**Alternatives considered:** Hugging Face Spaces (works, less familiar);
Railway/Render (better background jobs, but paid past the free tier);
a local Postgres (operational overhead with no benefit at this scale).

---

## Known limitations

State these in the writeup rather than hoping nobody asks.

1. **Wikipedia membership is incomplete** before roughly the mid-2010s.
   Survivorship bias is reduced, not eliminated. Quantify the residual by
   comparing against a shorter, cleaner window if possible.
2. **Sector classifications are current, not point-in-time.** The 2018
   creation of Communication Services is not handled, mildly
   contaminating sector-neutral results before that date.
3. **Dollar-neutral is not beta-neutral.** The book nets to zero in
   dollars but can carry residual market beta. Beta-neutralizing is a
   natural extension.
4. **Borrow costs are not modeled.** The liquidity screen is a proxy.
   Hard-to-borrow names would cost more than the flat 10bps assumed.
5. **No market-impact model.** Fine at small size; not scalable.
6. **The deflated Sharpe implementation is approximate.** It is
   indicative of the multiple-testing haircut, not a faithful
   reproduction of the published method. Verify the reference before
   citing it — this was written without literature access.

---

## Next steps

- Wire the sentiment index into `exposure_scaler` and **ablate it** —
  show performance with and without, or you cannot claim it contributed
- Sign-and-significance screen before combining signals
- Drop one of the two redundant momentum signals
- Add signals from genuinely different families (seasonality, dispersion)
  since the current library is momentum-heavy
- Beta-neutral portfolio construction
- Streamlit front end over the cached parquet
