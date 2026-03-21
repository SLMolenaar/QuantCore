"""
python/quantcore/calendar.py
==============================
Trading calendar filter for bar data.

Filters a BarSeries to remove bars that fall on non-trading days
(weekends, exchange holidays, early closes). Uses pandas_market_calendars
as the authoritative source of exchange schedules.

Usage:

    from quantcore.calendar import TradingCalendar

    cal   = TradingCalendar("NYSE")
    bars  = qc.load_csv_data("data/aapl.csv", "AAPL")
    bars  = cal.filter_bars(bars)

Or via the convenience parameter on load_csv_data:

    bars  = qc.load_csv_data("data/aapl.csv", "AAPL", calendar="NYSE")

Supported exchange names are those accepted by pandas_market_calendars,
e.g. "NYSE", "NASDAQ", "LSE", "TSX", "EUREX". Run
TradingCalendar.available_calendars() for the full list.
"""

from __future__ import annotations

import datetime
import warnings
from typing import TYPE_CHECKING, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from quantcore._core import BarData


# Seconds per day, used to convert nanosecond timestamps to date integers.
_NS_PER_DAY = 86_400 * 1_000_000_000


def _ns_to_date_int(timestamp_ns: int) -> int:
    """Convert a nanosecond UNIX timestamp to an integer YYYYMMDD date."""
    dt = datetime.datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000, tz=datetime.timezone.utc
    )
    return dt.year * 10_000 + dt.month * 100 + dt.day


class TradingCalendar:
    """
    Exchange trading calendar backed by pandas_market_calendars.

    Filters bar series to include only bars whose date falls on a valid
    trading session for the given exchange. Weekends and all exchange
    holidays (including Good Friday, Juneteenth, ad hoc closures) are
    excluded.

    Parameters
    ----------
    exchange : str
        Exchange identifier accepted by pandas_market_calendars, e.g.
        "NYSE", "NASDAQ", "LSE", "TSX". Case-insensitive. Pass None or
        an empty string to construct a no-op calendar that passes all bars.
    tz : str
        Timezone used when converting nanosecond timestamps to dates.
        Defaults to "UTC". For US equity daily bars "America/New_York"
        is more accurate, but "UTC" produces identical results for
        whole-day bars whose timestamps fall at midnight UTC.
    """

    def __init__(self, exchange: Optional[str] = None, tz: str = "UTC"):
        self._exchange = exchange
        self._tz       = tz
        self._cache: Optional[Set[int]] = None  # set of valid YYYYMMDD ints
        self._cache_range: Optional[tuple[int, int]] = None  # (start_ns, end_ns)

        if exchange:
            try:
                import pandas_market_calendars  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "pandas_market_calendars is required for calendar filtering:\n"
                    "  pip install pandas_market_calendars"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def available_calendars() -> List[str]:
        """Return a sorted list of supported exchange names."""
        try:
            import pandas_market_calendars as mcal
            return sorted(mcal.get_calendar_names())
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is required: pip install pandas_market_calendars"
            )

    def is_trading_day(self, timestamp_ns: int) -> bool:
        """Return True if the given nanosecond timestamp falls on a trading day."""
        if not self._exchange:
            return True
        date_int = _ns_to_date_int(timestamp_ns)
        trading_days = self._get_trading_days(timestamp_ns, timestamp_ns)
        return date_int in trading_days

    def filter_bars(
            self,
            bars:          "List[BarData]",
            strict:        bool  = False,
            max_skip_pct:  float = 0.20,
    ) -> "List[BarData]":
        """
        Remove bars that fall on non-trading days.

        Parameters
        ----------
        bars :
            Input bar series, sorted by timestamp ascending.
        strict :
            When True, raises RuntimeError if more than max_skip_pct of
            bars are filtered out. When False (default), emits a warning.
        max_skip_pct :
            Threshold fraction for the strict / warn check.

        Returns
        -------
        Filtered list of BarData, preserving the original sort order.
        """
        if not bars or not self._exchange:
            return bars

        start_ns = bars[0].timestamp_ns
        end_ns   = bars[-1].timestamp_ns
        trading_days = self._get_trading_days(start_ns, end_ns)

        filtered = [b for b in bars if _ns_to_date_int(b.timestamp_ns) in trading_days]

        n_removed = len(bars) - len(filtered)
        if n_removed > 0:
            skip_pct = n_removed / len(bars)
            msg = (
                f"TradingCalendar ({self._exchange}): removed {n_removed} of "
                f"{len(bars)} bars ({skip_pct * 100:.1f}%) that fell on "
                f"non-trading days."
            )
            if strict and skip_pct > max_skip_pct:
                raise RuntimeError(msg)
            warnings.warn(msg, stacklevel=3)

        if not filtered:
            raise RuntimeError(
                f"TradingCalendar ({self._exchange}): all {len(bars)} bars were "
                f"removed. Check that the exchange name is correct and that the "
                f"data covers a period when the exchange was open."
            )

        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trading_days(self, start_ns: int, end_ns: int) -> Set[int]:
        """
        Return the set of valid trading day integers (YYYYMMDD) covering
        [start_ns, end_ns], with a small buffer on each side.

        Results are cached per instance. The cache is invalidated when the
        requested range extends beyond the previously cached range.
        """
        # Expand the range by one week on each side to avoid edge effects
        # when the first or last bar falls near a holiday boundary.
        buffer_ns = 7 * _NS_PER_DAY
        req_start = start_ns - buffer_ns
        req_end   = end_ns   + buffer_ns

        if self._cache is not None and self._cache_range is not None:
            cached_start, cached_end = self._cache_range
            if req_start >= cached_start and req_end <= cached_end:
                return self._cache

        import pandas as pd
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar(self._exchange)

        # Convert nanoseconds to date strings for the API.
        start_dt = pd.Timestamp(req_start, unit="ns", tz="UTC").strftime("%Y-%m-%d")
        end_dt   = pd.Timestamp(req_end,   unit="ns", tz="UTC").strftime("%Y-%m-%d")

        schedule = cal.schedule(start_date=start_dt, end_date=end_dt)

        # Extract YYYYMMDD integers from the session index.
        trading_days: Set[int] = set()
        for ts in schedule.index:
            d = ts.date()
            trading_days.add(d.year * 10_000 + d.month * 100 + d.day)

        self._cache       = trading_days
        self._cache_range = (req_start, req_end)
        return trading_days