"""Signal Research Platform — interactive dashboard.

Run locally:   streamlit run app.py
Deployed:      Streamlit Community Cloud, connected to the GitHub repo.

Reads the cached Parquet store produced by run_research.py and
run_sentiment.py. If the cache is absent the app explains how to build
it rather than failing.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Signal Research Platform",
                   page_icon="📊", layout="wide")

NAVY, RED, GREEN, GREY = "#1F3864", "#C0392B", "#1E8449", "#7F8C8D"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cached data…")
def load_cache():
    from signal_platform.data import cache
    out = {}
    for key in ("px_close", "px_volume", "universe_sp500",
                "sentiment_index", "sentiment_proxies", "sp500_sectors_hist"):
        try:
            out[key] = cache.read(key)
        except Exception:
            out[key] = None
    return out


@st.cache_data(show_spinner="Computing signals…")
def compute_signals(_close, _volume, _membership, _sectors):
    from signal_platform.data.prices import clean_prices
    from signal_platform.data import universe as uni
    from signal_platform.research import transforms as tf
    from signal_platform.signals import MarketData, default_registry

    close, volume, report = clean_prices(_close, _volume, verbose=False)
    mem = _membership.reindex(index=close.index, columns=close.columns).fillna(False)
    mem = uni.apply_liquidity_filter(mem, close, volume,
                                     min_dollar_volume=5_000_000.0, min_price=5.0)
    data = MarketData(close=close, volume=volume,
                      sectors=_sectors, membership=mem)
    raw = default_registry().compute_all(data)
    prepared = {n: tf.prepare(f, _sectors, neutralize=True) for n, f in raw.items()}
    # Winsorisation clips the extreme 1% of each cross-section, which is
    # correct for research but collapses the top and bottom names to
    # identical values. For a rankings display that produces arbitrary
    # ordering among ties, so display scores skip the clipping step.
    display = {n: tf.zscore(tf.sector_neutralize(f, _sectors))
               for n, f in raw.items()}
    return close, volume, mem, prepared, display, report


@st.cache_data(show_spinner="Evaluating signals…")
def evaluate(_prepared, _close, horizon: int):
    from signal_platform.research import evaluation as ev
    table, details = ev.evaluate_registry(_prepared, _close,
                                          primary_horizon=horizon, cost_bps=10.0)
    corr = ev.signal_correlation(_prepared)
    return table, details, corr


def missing_cache_notice():
    st.error("No cached data found.")
    st.markdown("""
This dashboard reads a local Parquet cache. To build it:

```bash
pip install -r requirements.txt
python run_research.py --live      # universe + prices + signals (~30 min)
python run_sentiment.py --live     # sentiment index (~2 min)
```

Then reload this page.
    """)
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Signal Research Platform")
st.caption("Cross-sectional equity signal research · M.S. Business Analytics capstone · "
           "Tynan Sander")

data = load_cache()
if data["px_close"] is None:
    missing_cache_notice()

sectors = (data["sp500_sectors_hist"]["sector"]
           if data["sp500_sectors_hist"] is not None
           and "sector" in data["sp500_sectors_hist"].columns
           else pd.Series(dtype=object))

close, volume, membership, prepared, display_scores, clean_report = compute_signals(
    data["px_close"], data["px_volume"], data["universe_sp500"], sectors)

sentiment = (data["sentiment_index"]["composite"]
             if data["sentiment_index"] is not None else None)

tab_monitor, tab_research, tab_method = st.tabs(
    ["Market Monitor", "Research Findings", "Methodology"])


# ===========================================================================
# TAB 1 — MARKET MONITOR
# ===========================================================================
with tab_monitor:
    st.warning(
        "**This is a research artifact, not investment advice.** "
        "The underlying research found **no statistically significant predictive "
        "edge** in any of these signals after transaction costs and multiple-testing "
        "correction, and the sentiment index tested as coincident rather than "
        "predictive. The rankings below show what the model *would* select; the "
        "research says you should not expect that to be profitable. "
        "See the Methodology tab.",
        icon="⚠️")

    if sentiment is not None and sentiment.notna().any():
        from signal_platform.sentiment.index import label

        latest = sentiment.dropna()
        val, when = float(latest.iloc[-1]), latest.index[-1]

        st.subheader("Market sentiment")
        c1, c2, c3 = st.columns([1.1, 1, 2.4])

        prev = float(latest.iloc[-22]) if len(latest) > 22 else np.nan
        # The label is a category, not a change. Streamlit's delta slot
        # renders its argument with a direction arrow and colour, which
        # would show "Extreme Fear" as a green upward move.
        c1.metric("Composite", f"{val:.1f}")
        c1.markdown(f"**{label(val)}**")
        c2.metric("One month ago",
                  f"{prev:.1f}" if not np.isnan(prev) else "—",
                  f"{val - prev:+.1f}" if not np.isnan(prev) else None,
                  delta_color="normal")
        c3.caption(f"As of {when.date()} · 0 = extreme fear, 100 = extreme greed · "
                   f"percentile against a trailing 3-year window")

        st.progress(min(max(val / 100.0, 0.0), 1.0))

        comps = [c for c in data["sentiment_index"].columns
                 if c not in ("composite", "n_components")]
        if comps:
            st.markdown("**Components**")
            cols = st.columns(len(comps))
            for col, name in zip(cols, comps):
                series = data["sentiment_index"][name].dropna()
                if len(series):
                    v = float(series.iloc[-1])
                    col.metric(name.replace("_", " ").title(), f"{v:.0f}")

        st.markdown("**Twelve-month history**")
        # Series.last() was removed in pandas 3.0; filter on the index instead.
        cutoff = latest.index[-1] - pd.Timedelta(days=365)
        hist = latest[latest.index >= cutoff]
        st.line_chart(hist.rename("Sentiment"), height=220)

    st.divider()

    st.subheader("Current signal rankings")
    sig_names = sorted(prepared)
    pick = st.selectbox("Signal", sig_names, index=0, key="monitor_signal")

    scores = display_scores[pick]
    last_row = scores.dropna(how="all").iloc[-1].dropna()
    if len(last_row):
        as_of = scores.dropna(how="all").index[-1]
        st.caption(f"As of {as_of.date()} · {len(last_row)} names scored · "
                   f"sector-neutralised z-scores (unclipped, so ranks are distinct; "
                   f"the research pipeline winsorises at the 1st and 99th percentile)")

        top = last_row.sort_values(ascending=False).head(15)
        bot = last_row.sort_values().head(15)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Highest scoring** (model would hold long)")
            st.dataframe(
                pd.DataFrame({"Ticker": top.index, "Score": top.values.round(3),
                              "Sector": [sectors.get(t, "—") for t in top.index]}),
                hide_index=True, width='stretch', height=560)
        with c2:
            st.markdown("**Lowest scoring** (model would hold short)")
            st.dataframe(
                pd.DataFrame({"Ticker": bot.index, "Score": bot.values.round(3),
                              "Sector": [sectors.get(t, "—") for t in bot.index]}),
                hide_index=True, width='stretch', height=560)


# ===========================================================================
# TAB 2 — RESEARCH
# ===========================================================================
with tab_research:
    st.subheader("Universe construction")
    c1, c2, c3, c4 = st.columns(4)

    ever = int(data["universe_sp500"].any(axis=0).sum())
    median_members = int(data["universe_sp500"].sum(axis=1).median())
    c1.metric("Companies ever a member", f"{ever}")
    c2.metric("Members on a typical date", f"{median_members}")
    c3.metric("Survivorship gap", f"{ever - median_members}",
              help="Companies acquired, delisted, or relegated. A naive backtest "
                   "silently excludes these, inflating returns.")
    c4.metric("Usable after cleaning", f"{close.shape[1]}")

    st.caption(
        f"Point-in-time membership reconstructed from quarterly Wikipedia revision "
        f"snapshots. Cleaning removed {len(clean_report['empty_columns'])} empty and "
        f"{len(clean_report['thin_columns'])} thin series and masked "
        f"{clean_report['bad_prints']} suspected bad prints.")

    members = data["universe_sp500"].sum(axis=1)
    st.line_chart(members.rename("Index members"), height=200)

    st.divider()
    st.subheader("Signal evaluation")

    horizon = st.radio("Forward horizon (trading days)", [1, 5, 21, 63],
                       index=1, horizontal=True)
    table, details, corr = evaluate(prepared, close, horizon)

    show = table[["mean_ic", "ic_t_stat", "ic_ir", "monotonicity",
                  "ann_spread_gross", "ann_cost_drag", "ann_spread_net",
                  "avg_turnover"]].copy()
    show.columns = ["Mean IC", "t-stat", "IC-IR", "Monotonicity",
                    "Gross spread", "Cost drag", "Net spread", "Turnover"]
    st.dataframe(show.style.format("{:.4f}").background_gradient(
        subset=["t-stat"], cmap="RdYlGn", vmin=-3, vmax=3),
        width='stretch')

    n_sig = int((table["ic_t_stat"].abs() > 3).sum())
    if n_sig == 0:
        st.info(f"**No signal clears |t| > 3 at the {horizon}-day horizon.** "
                "That threshold reflects the 44 specifications tested across this "
                "project; a t-statistic that would impress at one trial is "
                "unremarkable at forty-four.")
    else:
        st.success(f"{n_sig} signal(s) clear |t| > 3 at this horizon.")

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Cross-signal correlation**")
        st.caption("Below 0.3 means a signal adds breadth. Above 0.8 is redundant.")
        short = {n: n.split("_")[0][:16] for n in corr.columns}
        st.dataframe(corr.rename(index=short, columns=short)
                     .style.format("{:.2f}").background_gradient(
                         cmap="RdBu_r", vmin=-1, vmax=1),
                     width='stretch')

        redundant = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
                     for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.8]
        for a, b, v in redundant:
            st.caption(f"⚠️ {a.split('_')[0]} ↔ {b.split('_')[0]}: {v:.3f} — redundant")

    with c2:
        st.markdown("**IC decay across horizons**")
        st.caption("Reveals each signal's natural holding period.")
        from signal_platform.research import evaluation as ev
        decay_t = {}
        for name, sig in prepared.items():
            d = ev.ic_decay(sig, close, horizons=(1, 5, 21, 63))
            decay_t[name.split("_")[0][:16]] = d["t_stat"]
        st.line_chart(pd.DataFrame(decay_t), height=300)

    st.divider()
    st.subheader("Composite strategy")

    from signal_platform.research import transforms as tf
    from signal_platform.research.backtest import run_backtest, walk_forward_report

    use_sent = st.checkbox(
        "Apply sentiment exposure scaling (contrarian)", value=False,
        help="Ablation testing found this does not improve out-of-sample results.")

    composite = tf.combine(prepared)
    scaler = None
    if use_sent and sentiment is not None:
        from signal_platform.sentiment.index import to_exposure_scaler
        scaler = to_exposure_scaler(sentiment, direction="contrarian")

    res = run_backtest(composite, close, rebalance_freq="W-FRI",
                       cost_bps=10.0, exposure_scaler=scaler)

    c1, c2, c3, c4 = st.columns(4)
    s = res["stats"]
    c1.metric("Net Sharpe", f"{s['sharpe']:.3f}")
    c2.metric("Annual return", f"{s['ann_return']:.2%}")
    c3.metric("Max drawdown", f"{s['max_drawdown']:.1%}")
    c4.metric("Cost drag",
              f"{res['stats_gross']['ann_return'] - s['ann_return']:.2%}")

    eq = pd.DataFrame({
        "Gross": (1 + res["returns_gross"].fillna(0)).cumprod(),
        "Net of costs": (1 + res["returns_net"].fillna(0)).cumprod(),
    })
    st.line_chart(eq, height=280)

    with st.expander("Walk-forward validation (purged, 21-day embargo)"):
        wf = walk_forward_report(composite, close, n_splits=4, embargo_days=21)
        st.dataframe(wf[["test_start", "test_end", "ann_return", "ann_vol",
                         "sharpe", "max_drawdown"]].style.format(
                             {"ann_return": "{:.2%}", "ann_vol": "{:.2%}",
                              "sharpe": "{:.3f}", "max_drawdown": "{:.1%}"}),
                     width='stretch')
        pos = int((wf["sharpe"] > 0).sum())
        st.caption(f"{pos} of {len(wf)} folds positive. Consistency across folds is "
                   "the finding; four of four would be a signal, two of four is a "
                   "coin flip.")


# ===========================================================================
# TAB 3 — METHODOLOGY
# ===========================================================================
with tab_method:
    st.subheader("What this project found")
    st.markdown("""
Across **613 companies**, **eleven years**, **seven signals**, and **four horizons** —
44 specifications in total — **no result survived correction for the number of tests
performed.** The sentiment index tested as coincident with recent market returns rather
than predictive of future ones, and neither contrarian nor momentum exposure scaling
improved out-of-sample performance.

This is the expected outcome. Large-capitalization U.S. equities are the most heavily
arbitraged segment of the market, and the replication literature finds that published
anomalies decay substantially after publication.
    """)

    st.divider()
    st.subheader("How the standard failure modes are guarded")
    st.table(pd.DataFrame([
        ["Survivorship bias", "Point-in-time membership from quarterly Wikipedia "
         "revisions; delisted names retained"],
        ["Lookahead bias", "One-day execution lag enforced in forward returns; "
         "unit-tested against hand-computed values"],
        ["Restated fundamentals", "Signal library restricted to price and volume, "
         "which carry unambiguous timestamps"],
        ["Overlapping-return t-stats", "Newey-West HAC standard errors throughout"],
        ["Train/test leakage", "Purged walk-forward with a 21-day embargo"],
        ["Multiple testing", "Deflated Sharpe hurdle scaled by cumulative trial count"],
        ["Unmodeled costs", "10 bps per side on realised turnover; gross and net "
         "both reported"],
        ["Unintended sector bets", "Signals demeaned within GICS sector"],
        ["Anti-conservative tests", "Block-bootstrap null calibration for the "
         "sentiment regression"],
    ], columns=["Failure mode", "Guard"]))

    st.divider()
    st.subheader("Validating the instrument")
    st.markdown("""
A flat result admits two readings: the signals have no edge, or the harness is broken.
Distinguishing them requires showing the harness has both **power** and **correct size**.

Synthetic data is generated with a planted effect of known strength. The harness must
find it (it does: IC 0.376 at t = 147) and must report approximately zero when the
effect is removed (it does: IC 0.006 at t = 2.15). Thirteen tests cover lookahead
alignment, sector demeaning, dollar neutrality, cost monotonicity, and embargo
enforcement. All pass, and synthetic runs are byte-identical across operating systems.
    """)

    st.divider()
    st.subheader("A test that was wrong, and how it was caught")
    st.markdown("""
The predictive regression for the sentiment index declared a significant relationship
**on synthetic data containing no relationship by construction** (t = −3.04).

Across forty independent synthetic datasets, the conventional test produced |t| > 2 in
**8 to 20 percent of cases** rather than the nominal 5 percent. The cause is Stambaugh
bias: the index is highly persistent and partly derived from the same price series whose
future it is asked to predict. Newey-West corrects for overlapping returns, not for this.

A block-bootstrap null distribution was implemented. It shows the true standard deviation
of the statistic is **1.25, not 1.0**, and the 99th percentile is **3.19, not 2.58**.
Under the calibrated null, the live result (p = 0.073) is not significant — where the
naive test reported p = 0.026.
    """)

    st.divider()
    st.subheader("Limitations")
    st.markdown("""
- **164 companies unavailable** from the price vendor, disproportionately failures and
  acquisitions, which reintroduces some survivorship bias. The largest residual limitation.
- **Quarterly membership sampling** dates index changes only to within a quarter.
- **Sectors are first-observed**, so GICS reclassifications are not tracked through time.
- **Dollar-neutral is not beta-neutral.**
- **No borrow-cost or market-impact model**; the liquidity screen is a proxy.
- **Two sentiment components omitted** (put/call ratio, McClellan volume), and the credit
  component is an ETF proxy rather than the option-adjusted spread.
- **Market timing has little breadth** — roughly 44 independent observations at a 63-day
  horizon — so negative results there have limited power. *Not significant* does not
  establish *no effect*.
    """)

    st.divider()
    st.caption("Source: github.com/ty17sander/UC-Capstone-2026 · "
               "Built with Python, pandas, statsmodels, DuckDB, and Streamlit.")
