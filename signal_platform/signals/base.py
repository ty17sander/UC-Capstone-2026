"""The signal interface.

Every signal in the library implements the same contract:

    compute(data) -> DataFrame (dates x tickers) of raw scores

Higher score means "expected to outperform". Any signal where the
economic logic runs the other way (short-term reversal, high volatility)
must flip its own sign internally, so that the evaluation harness never
has to special-case anything.

This uniformity is what makes the platform a platform. Because every
signal looks identical from the outside, the evaluation harness, the
correlation matrix, and the combiner all work on signals that had not
been written when they were built.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketData:
    """Container passed to every signal. Wide frames, dates x tickers."""
    close: pd.DataFrame
    volume: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    open: pd.DataFrame | None = None
    sectors: pd.Series | None = None
    membership: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.close is None or self.close.empty:
            raise ValueError("MarketData requires a non-empty close frame.")
        self.close = self.close.sort_index()

    @property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

    def require(self, *fields: str) -> None:
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(f"Signal requires fields not supplied: {missing}")


class Signal(ABC):
    """Base class for all signals."""

    #: Signals in the same family tend to correlate; used when reporting.
    family: str = "unclassified"
    #: Free-text economic rationale. Required -- a signal you cannot
    #: justify beforehand is a data-mining result waiting to happen.
    rationale: str = ""

    def __init__(self, **params) -> None:
        self.params = params

    @property
    def name(self) -> str:
        if not self.params:
            return type(self).__name__
        bits = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return f"{type(self).__name__}_{bits}"

    @abstractmethod
    def compute(self, data: MarketData) -> pd.DataFrame:
        """Raw cross-sectional scores. Higher = more attractive long."""
        raise NotImplementedError

    def __call__(self, data: MarketData) -> pd.DataFrame:
        scores = self.compute(data)
        if not isinstance(scores, pd.DataFrame):
            raise TypeError(f"{self.name}.compute must return a DataFrame")
        # Mask to the investable universe so signals never score names
        # that were not in the index on that date.
        if data.membership is not None:
            mask = data.membership.reindex(
                index=scores.index, columns=scores.columns
            ).fillna(False)
            scores = scores.where(mask)
        return scores

    def __repr__(self) -> str:
        return f"<Signal {self.name}>"


@dataclass
class SignalRegistry:
    """Named collection of signals, evaluated together."""
    signals: list[Signal] = field(default_factory=list)

    def add(self, sig: Signal) -> "SignalRegistry":
        if any(s.name == sig.name for s in self.signals):
            raise ValueError(f"Duplicate signal name: {sig.name}")
        self.signals.append(sig)
        return self

    def compute_all(self, data: MarketData) -> dict[str, pd.DataFrame]:
        return {s.name: s(data) for s in self.signals}

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"name": s.name, "family": s.family, "rationale": s.rationale}
            for s in self.signals
        ])

    def __len__(self) -> int:
        return len(self.signals)
