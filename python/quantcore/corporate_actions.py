from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class SplitEvent:
    """A stock split or reverse split."""
    ex_date_ns: int    # nanosecond timestamp of the ex-date
    ratio: float       # shares_after / shares_before
    # 2.0 = 2-for-1 split, 0.5 = 1-for-2 reverse split


@dataclass
class DividendEvent:
    """A cash dividend payment."""
    ex_date_ns: int    # nanosecond timestamp of the ex-date
    amount: float      # dividend per share in the same currency as prices


class CorporateActionsAdjuster:
    """
    Adjust a raw OHLCV bar series for splits and dividends.

    Applies backward adjustment: all prices before each event are scaled
    so that the series is continuous as of the most recent date. This is
    the industry-standard method (CRSP methodology) and is what Yahoo
    Finance's Adj Close and most data vendors produce.

    For return-based strategies this is equivalent to forward adjustment.
    The only scenario where it matters is absolute-price-level strategies,
    which are unusual.

    Usage:
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=..., ratio=2.0)          # 2-for-1
        adjuster.add_dividend(ex_date_ns=..., amount=0.23)

        adjusted_bars = adjuster.adjust(raw_bars)

    Loading from CSV:
        adjuster = CorporateActionsAdjuster.from_csv(
            splits_csv='data/aapl_splits.csv',
            dividends_csv='data/aapl_dividends.csv',
        )
    """

    def __init__(self):
        self._splits:    List[SplitEvent]   = []
        self._dividends: List[DividendEvent] = []

    def add_split(self, ex_date_ns: int, ratio: float) -> "CorporateActionsAdjuster":
        """
        Register a split event.

        ratio = shares_after / shares_before
          2.0 for a 2-for-1 forward split
          0.5 for a 1-for-2 reverse split
        """
        if ratio <= 0.0:
            raise ValueError("Split ratio must be positive")
        self._splits.append(SplitEvent(ex_date_ns=ex_date_ns, ratio=ratio))
        return self

    def add_dividend(self, ex_date_ns: int, amount: float) -> "CorporateActionsAdjuster":
        """Register a cash dividend event."""
        if amount < 0.0:
            raise ValueError("Dividend amount must be non-negative")
        self._dividends.append(DividendEvent(ex_date_ns=ex_date_ns, amount=amount))
        return self

    def adjust(self, bars: List, symbol: str = "") -> List:
        # Dividend factors are computed from the raw (unadjusted) close of the bar
        # immediately before each ex-date. This avoids order-dependency between
        # multiple dividend events and matches the behaviour of most data vendors.
        """
        Return a new list of BarData with prices adjusted for all registered
        corporate actions.

        Volumes are inverse-adjusted for splits (a 2-for-1 split doubles
        share count, so pre-split volumes are doubled to reflect equivalent
        liquidity). Volumes are not adjusted for dividends.

        Args:
            bars:   List[BarData] sorted by timestamp ascending.
            symbol: Used only for error messages.

        Returns:
            New List[BarData] with adjusted prices. Original bars are not modified.
        """
        if not bars:
            return []

        if not self._splits and not self._dividends:
            return bars

        # Work with numpy for speed; reconstruct BarData at the end.
        n = len(bars)
        timestamps = np.array([b.timestamp_ns for b in bars], dtype=np.int64)
        opens      = np.array([b.open   for b in bars], dtype=np.float64)
        highs      = np.array([b.high   for b in bars], dtype=np.float64)
        lows       = np.array([b.low    for b in bars], dtype=np.float64)
        closes     = np.array([b.close  for b in bars], dtype=np.float64)
        volumes    = np.array([b.volume for b in bars], dtype=np.float64)

        # Cumulative price multiplier applied to each bar, starting at 1.0.
        # We traverse from newest to oldest, accumulating factors as we
        # pass each corporate action going backward in time.
        price_factor  = np.ones(n, dtype=np.float64)
        volume_factor = np.ones(n, dtype=np.float64)

        # Build a unified timeline of events sorted newest-first.
        events: list[tuple[int, str, float]] = []
        for s in self._splits:
            events.append((s.ex_date_ns, "split", s.ratio))
        for d in self._dividends:
            events.append((d.ex_date_ns, "dividend", d.amount))
        events.sort(key=lambda e: e[0], reverse=True)

        cumulative_price_mult  = 1.0
        cumulative_volume_mult = 1.0

        event_idx = 0
        # Walk bars from newest to oldest.
        for bar_i in range(n - 1, -1, -1):
            # Apply all events whose ex_date is after this bar's timestamp.
            while event_idx < len(events) and events[event_idx][0] > timestamps[bar_i]:
                _, kind, value = events[event_idx]
                if kind == "split":
                    # Pre-split prices are divided by ratio; volumes are multiplied.
                    cumulative_price_mult  /= value
                    cumulative_volume_mult *= value
                elif kind == "dividend":
                    # Proportional dividend adjustment (CRSP method):
                    # factor = (close_on_ex_date - dividend) / close_on_ex_date
                    # We use the close of the bar immediately before ex_date.
                    # At this point bar_i is the bar before the ex-date.
                    ex_close = closes[bar_i]
                    if ex_close > 0:
                        div_factor = (ex_close - value) / ex_close
                        if div_factor > 0:
                            cumulative_price_mult *= div_factor
                event_idx += 1

            price_factor[bar_i]  = cumulative_price_mult
            volume_factor[bar_i] = cumulative_volume_mult

        adj_opens   = opens   * price_factor
        adj_highs   = highs   * price_factor
        adj_lows    = lows    * price_factor
        adj_closes  = closes  * price_factor
        adj_volumes = volumes * volume_factor

        # Reconstruct BarData objects.
        from quantcore._core import BarData
        sym = bars[0].symbol if not symbol else symbol
        result = []
        for i in range(n):
            result.append(BarData(
                sym,
                int(timestamps[i]),
                float(adj_opens[i]),
                float(adj_highs[i]),
                float(adj_lows[i]),
                float(adj_closes[i]),
                float(adj_volumes[i]),
            ))
        return result

    @classmethod
    def from_csv(
            cls,
            splits_csv:    Optional[str] = None,
            dividends_csv: Optional[str] = None,
    ) -> "CorporateActionsAdjuster":
        """
        Load corporate actions from CSV files.

        splits_csv format:    ex_date, ratio
        dividends_csv format: ex_date, amount

        ex_date may be an integer epoch (s/ms/us/ns) or ISO 8601 date string.
        """
        adjuster = cls()

        if splits_csv:
            df = pd.read_csv(splits_csv)
            for _, row in df.iterrows():
                adjuster.add_split(
                    ex_date_ns=_parse_date_to_ns(row["ex_date"]),
                    ratio=float(row["ratio"]),
                )

        if dividends_csv:
            df = pd.read_csv(dividends_csv)
            for _, row in df.iterrows():
                adjuster.add_dividend(
                    ex_date_ns=_parse_date_to_ns(row["ex_date"]),
                    amount=float(row["amount"]),
                )

        return adjuster


def _parse_date_to_ns(value) -> int:
    """Convert a date string or integer epoch to nanoseconds."""
    if isinstance(value, (int, float)):
        v = int(value)
        if v < 4_000_000_000:
            return v * 1_000_000_000
        elif v < 4_000_000_000_000:
            return v * 1_000_000
        elif v < 4_000_000_000_000_000:
            return v * 1_000
        return v
    return int(pd.Timestamp(str(value)).value)