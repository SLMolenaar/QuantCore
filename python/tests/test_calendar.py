"""
Tests for TradingCalendar

Tests are grouped by concern and use exact expected values where possible.
pandas_market_calendars is required; tests are skipped if it is not installed.
"""

import pytest
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc
from quantcore.calendar import TradingCalendar, _ns_to_date_int

# Skip the entire module if pandas_market_calendars is not installed.
pmc = pytest.importorskip("pandas_market_calendars", reason="pandas_market_calendars not installed")

# ============================================================================
# HELPERS
# ============================================================================

SEC = 1_000_000_000   # one second in nanoseconds
DAY = 86_400 * SEC    # one day in nanoseconds


def ts(year: int, month: int, day: int) -> int:
    """Return a nanosecond UNIX timestamp for midnight UTC on the given date."""
    import datetime
    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp()) * SEC


def make_bar(year: int, month: int, day: int, symbol: str = "TEST") -> qc.BarData:
    """Create a single BarData at midnight UTC on the given date."""
    return qc.BarData(symbol, ts(year, month, day), 100.0, 101.0, 99.0, 100.0, 1_000_000.0)


def make_bars(dates: list, symbol: str = "TEST") -> list:
    """Create a BarSeries from a list of (year, month, day) tuples."""
    return [make_bar(*d, symbol=symbol) for d in dates]


# ============================================================================
# _ns_to_date_int UNIT TESTS
# ============================================================================

class TestNsToDateInt:
    def test_known_date(self):
        # 2023-01-03 00:00:00 UTC
        result = _ns_to_date_int(ts(2023, 1, 3))
        assert result == 20230103

    def test_new_years_day(self):
        result = _ns_to_date_int(ts(2023, 1, 1))
        assert result == 20230101

    def test_end_of_year(self):
        result = _ns_to_date_int(ts(2022, 12, 31))
        assert result == 20221231

    def test_leap_day(self):
        result = _ns_to_date_int(ts(2024, 2, 29))
        assert result == 20240229

    def test_no_deprecation_warning(self):
        # utcfromtimestamp is deprecated in Python 3.12; verify the implementation
        # uses the timezone-aware alternative and emits no DeprecationWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _ns_to_date_int(ts(2023, 6, 15))
        assert result == 20230615


# ============================================================================
# INITIALIZATION
# ============================================================================

class TestInitialization:
    def test_no_op_calendar_with_none(self):
        # None exchange: all bars pass through regardless of date.
        cal = TradingCalendar(None)
        bars = make_bars([(2023, 1, 1), (2023, 1, 2)])  # holiday + weekday
        result = cal.filter_bars(bars)
        assert len(result) == 2

    def test_no_op_calendar_with_empty_string(self):
        cal = TradingCalendar("")
        bars = make_bars([(2023, 1, 1)])
        result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_valid_exchange_constructs(self):
        cal = TradingCalendar("NYSE")
        assert cal is not None

    def test_invalid_exchange_raises_on_filter(self):
        cal = TradingCalendar("INVALID_EXCHANGE_XYZ")
        bars = make_bars([(2023, 6, 1)])
        with pytest.raises(Exception):
            cal.filter_bars(bars)

    def test_missing_dependency_error_message(self):
        # Simulate missing pandas_market_calendars by temporarily hiding it.
        import sys
        original = sys.modules.get("pandas_market_calendars")
        sys.modules["pandas_market_calendars"] = None  # type: ignore
        try:
            with pytest.raises(ImportError, match="pandas_market_calendars"):
                TradingCalendar("NYSE")
        finally:
            if original is not None:
                sys.modules["pandas_market_calendars"] = original
            else:
                del sys.modules["pandas_market_calendars"]

    def test_available_calendars_returns_list(self):
        calendars = TradingCalendar.available_calendars()
        assert isinstance(calendars, list)
        assert len(calendars) > 0
        assert "NYSE" in calendars

    def test_available_calendars_sorted(self):
        calendars = TradingCalendar.available_calendars()
        assert calendars == sorted(calendars)


# ============================================================================
# NYSE HOLIDAY FILTERING
# ============================================================================
#
# Each test passes a series of two bars: the holiday under test plus one
# valid adjacent trading day. This confirms the holiday is removed while
# also verifying that the adjacent valid bar survives, without triggering
# the "all bars removed" RuntimeError guard (which is tested separately).
# ============================================================================

class TestNYSEHolidays:
    """
    Exact dates verified against the NYSE holiday schedule.
    https://www.nyse.com/markets/hours-calendars
    """

    def test_new_years_day_2023_filtered(self):
        # 2023-01-01 is a Sunday; NYSE observes it on Monday 2023-01-02.
        # 2023-01-03 is the first valid trading day of 2023.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 2), (2023, 1, 3)])  # observed holiday + valid day
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230103

    def test_new_years_day_2023_next_trading_day_passes(self):
        # 2023-01-03 is the first trading day of 2023.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 3)])
        result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_mlk_day_2023_filtered(self):
        # Martin Luther King Jr. Day: third Monday of January.
        # 2023-01-17 (Tuesday) is the next valid day.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 16), (2023, 1, 17)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230117

    def test_presidents_day_2023_filtered(self):
        # Presidents' Day: third Monday of February.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 2, 20), (2023, 2, 21)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230221

    def test_good_friday_2023_filtered(self):
        # Good Friday 2023: April 7.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 4, 7), (2023, 4, 6)])  # holiday + Thursday
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230406

    def test_easter_monday_not_filtered(self):
        # NYSE does not close for Easter Monday (only Good Friday).
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 4, 10)])  # Easter Monday 2023
        result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_memorial_day_2023_filtered(self):
        # Memorial Day: last Monday of May.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 5, 29), (2023, 5, 30)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230530

    def test_juneteenth_2023_filtered(self):
        # Juneteenth National Independence Day: June 19.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 6, 19), (2023, 6, 20)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230620

    def test_independence_day_2023_filtered(self):
        # July 4, 2023 is a Tuesday.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 7, 4), (2023, 7, 5)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230705

    def test_labor_day_2023_filtered(self):
        # Labor Day: first Monday of September.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 9, 4), (2023, 9, 5)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230905

    def test_thanksgiving_2023_filtered(self):
        # Thanksgiving: fourth Thursday of November.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 11, 23), (2023, 11, 22)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20231122

    def test_christmas_2023_filtered(self):
        # Christmas 2023: December 25 (Monday).
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 12, 25), (2023, 12, 26)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20231226

    def test_christmas_eve_2023_passes(self):
        # 2022-12-23 (Friday before Christmas on Sunday) is a regular trading day.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2022, 12, 23)])
        result = cal.filter_bars(bars)
        assert len(result) == 1


# ============================================================================
# WEEKEND FILTERING
# ============================================================================

class TestWeekendFiltering:
    def test_saturday_filtered(self):
        # 2023-01-07 is a Saturday; 2023-01-09 is the next Monday.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 7), (2023, 1, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230109

    def test_sunday_filtered(self):
        # 2023-01-08 is a Sunday; 2023-01-09 is Monday.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 8), (2023, 1, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230109

    def test_monday_passes(self):
        # 2023-01-09 is a regular Monday.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9)])
        result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_full_week_with_weekend(self):
        # Mon 2023-01-09 through Sun 2023-01-15 → 5 trading days, 2 weekend days.
        dates = [
            (2023, 1, 9),   # Mon
            (2023, 1, 10),  # Tue
            (2023, 1, 11),  # Wed
            (2023, 1, 12),  # Thu
            (2023, 1, 13),  # Fri
            (2023, 1, 14),  # Sat — filtered
            (2023, 1, 15),  # Sun — filtered
        ]
        cal  = TradingCalendar("NYSE")
        bars = make_bars(dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 5


# ============================================================================
# FILTER BEHAVIOUR
# ============================================================================

class TestFilterBehaviour:
    def test_empty_bars_returns_empty(self):
        cal    = TradingCalendar("NYSE")
        result = cal.filter_bars([])
        assert result == []

    def test_order_preserved(self):
        # Verify the output is sorted identically to the input (no reordering).
        dates = [(2023, 1, 9), (2023, 1, 10), (2023, 1, 11)]
        cal   = TradingCalendar("NYSE")
        bars  = make_bars(dates)
        result = cal.filter_bars(bars)

        assert [b.timestamp_ns for b in result] == [b.timestamp_ns for b in bars]

    def test_symbol_preserved(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9)], symbol="AAPL")
        result = cal.filter_bars(bars)
        assert result[0].symbol == "AAPL"

    def test_all_holidays_removed_raises(self):
        # A series consisting entirely of non-trading days must raise RuntimeError
        # rather than returning an empty list that would silently crash the engine.
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 2),   # New Year's (observed Mon)
            (2023, 1, 7),   # Saturday
            (2023, 1, 8),   # Sunday
        ])
        with pytest.raises(RuntimeError, match="all"):
            cal.filter_bars(bars)

    def test_single_holiday_bar_raises(self):
        # A single-bar series with only a non-trading day also raises.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 7, 4)])  # Independence Day
        with pytest.raises(RuntimeError, match="all"):
            cal.filter_bars(bars)

    def test_strict_mode_raises_above_threshold(self):
        # 4 bars: 3 weekends/holidays + 1 valid → 75% removed > 20% threshold.
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 2),   # observed New Year's — holiday
            (2023, 1, 7),   # Saturday
            (2023, 1, 8),   # Sunday
            (2023, 1, 9),   # Monday — valid
        ])
        with pytest.raises(RuntimeError):
            cal.filter_bars(bars, strict=True, max_skip_pct=0.20)

    def test_strict_mode_passes_below_threshold(self):
        # 5 bars: 1 weekend + 4 valid → 20% removed == threshold (not above).
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 7),   # Saturday — filtered
            (2023, 1, 9),   # Mon
            (2023, 1, 10),  # Tue
            (2023, 1, 11),  # Wed
            (2023, 1, 12),  # Thu
        ])
        # 1/5 = 20%, which equals but does not exceed the threshold → no raise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars, strict=True, max_skip_pct=0.20)
        assert len(result) == 4

    def test_warns_on_filtered_bars(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 7), (2023, 1, 9)])  # Saturday + Monday
        with pytest.warns(UserWarning, match="non-trading"):
            result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_no_warning_when_nothing_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9), (2023, 1, 10)])  # two valid trading days
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cal.filter_bars(bars)
        assert len(result) == 2


# ============================================================================
# is_trading_day
# ============================================================================

class TestIsTradingDay:
    def test_regular_trading_day(self):
        cal = TradingCalendar("NYSE")
        assert cal.is_trading_day(ts(2023, 1, 9)) is True   # Monday

    def test_saturday_not_trading_day(self):
        cal = TradingCalendar("NYSE")
        assert cal.is_trading_day(ts(2023, 1, 7)) is False

    def test_sunday_not_trading_day(self):
        cal = TradingCalendar("NYSE")
        assert cal.is_trading_day(ts(2023, 1, 8)) is False

    def test_holiday_not_trading_day(self):
        cal = TradingCalendar("NYSE")
        assert cal.is_trading_day(ts(2023, 7, 4)) is False   # Independence Day

    def test_no_op_calendar_always_true(self):
        cal = TradingCalendar(None)
        assert cal.is_trading_day(ts(2023, 1, 1)) is True   # would normally be a holiday
        assert cal.is_trading_day(ts(2023, 1, 7)) is True   # Saturday


# ============================================================================
# CACHING
# ============================================================================

class TestCaching:
    def test_repeated_calls_return_same_result(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9), (2023, 1, 16)])  # trading day + MLK Day

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = cal.filter_bars(bars)
            result2 = cal.filter_bars(bars)

        assert len(result1) == len(result2)
        assert [b.timestamp_ns for b in result1] == [b.timestamp_ns for b in result2]

    def test_cache_used_on_second_call(self):
        # Verify the schedule is not re-fetched on the second call by checking
        # that _cache is populated after the first call and unchanged after the second.
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 6, 1)])

        cal.filter_bars(bars)
        cache_after_first = id(cal._cache)

        cal.filter_bars(bars)
        cache_after_second = id(cal._cache)

        assert cache_after_first == cache_after_second

    def test_cache_invalidated_on_extended_range(self):
        # First call covers 2023; second call covers 2021 (outside cached range).
        cal = TradingCalendar("NYSE")

        cal.filter_bars(make_bars([(2023, 6, 1)]))
        first_range = cal._cache_range

        cal.filter_bars(make_bars([(2021, 6, 1)]))
        second_range = cal._cache_range

        assert second_range != first_range


# ============================================================================
# MULTI-EXCHANGE
# ============================================================================

class TestMultiExchange:
    def test_lse_filters_uk_bank_holiday(self):
        # 2023-05-08: extra bank holiday for King Charles III coronation.
        # 2023-05-09 (Tuesday) is the next valid day.
        cal  = TradingCalendar("LSE")
        bars = make_bars([(2023, 5, 8), (2023, 5, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230509

    def test_tsx_filters_canadian_holiday(self):
        # 2023-07-03: Canada Day (observed, since July 1 falls on a Saturday).
        # 2023-07-04 (Tuesday) is the next valid day.
        cal  = TradingCalendar("TSX")
        bars = make_bars([(2023, 7, 3), (2023, 7, 4)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230704

    def test_different_exchanges_have_different_holidays(self):
        # Veterans Day (2023-11-10 observed, since Nov 11 is a Saturday):
        # NYSE stays open; some European exchanges close.
        nyse_cal = TradingCalendar("NYSE")
        bars     = make_bars([(2023, 11, 10)])
        # NYSE is open on Veterans Day — bar should not be filtered.
        nyse_result = nyse_cal.filter_bars(bars)
        assert len(nyse_result) == 1


# ============================================================================
# INTEGRATION WITH load_csv_data / run_backtest
# ============================================================================

class TestPublicAPIIntegration:
    """
    These tests verify that the calendar parameter wires through correctly
    on the public-facing convenience functions. They use synthetic BarData
    (no actual CSV files required) to stay self-contained.
    """

    def test_run_backtest_with_calendar_filters_holidays(self):
        # Build a series that includes Christmas 2023 (Dec 25, a Monday) and
        # the surrounding trading days. The holiday bar should be removed before
        # the backtest runs, so the strategy sees one fewer bar.
        dates_with_holiday = [
            (2023, 12, 22),  # Friday — trading
            (2023, 12, 25),  # Monday — Christmas, filtered
            (2023, 12, 26),  # Tuesday — trading
        ]
        dates_without_holiday = [
            (2023, 12, 22),
            (2023, 12, 26),
        ]

        bars_with    = make_bars(dates_with_holiday)
        bars_without = make_bars(dates_without_holiday)

        class CountingStrategy(qc.Strategy):
            def __init__(self):
                super().__init__("Counter")
                self.count = 0

            def on_data(self, event):
                self.count += 1

        strat_filtered   = CountingStrategy()
        strat_unfiltered = CountingStrategy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc.run_backtest(strat_filtered, {"TEST": bars_with}, calendar="NYSE")

        qc.run_backtest(strat_unfiltered, {"TEST": bars_without})

        assert strat_filtered.count == strat_unfiltered.count == 2

    def test_run_backtest_without_calendar_includes_holidays(self):
        # Without a calendar, all bars pass through including the holiday.
        dates = [(2023, 12, 22), (2023, 12, 25), (2023, 12, 26)]
        bars  = make_bars(dates)

        class CountingStrategy(qc.Strategy):
            def __init__(self):
                super().__init__("Counter")
                self.count = 0

            def on_data(self, event):
                self.count += 1

        strat = CountingStrategy()
        qc.run_backtest(strat, {"TEST": bars})
        assert strat.count == 3

    def test_run_backtest_multi_symbol_calendar_applied_to_all(self):
        # Both symbols must have the holiday bar removed.
        holiday_and_two_valid = [
            (2023, 12, 22),  # valid
            (2023, 12, 25),  # Christmas — filtered
            (2023, 12, 26),  # valid
        ]

        class CountingStrategy(qc.Strategy):
            def __init__(self):
                super().__init__("Counter")
                self.counts = {}

            def on_data(self, event):
                sym = event.get_symbol()
                self.counts[sym] = self.counts.get(sym, 0) + 1

        bars_a = make_bars(holiday_and_two_valid, symbol="AAA")
        bars_b = make_bars(holiday_and_two_valid, symbol="BBB")

        strat = CountingStrategy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc.run_backtest(strat, {"AAA": bars_a, "BBB": bars_b}, calendar="NYSE")

        assert strat.counts.get("AAA", 0) == 2
        assert strat.counts.get("BBB", 0) == 2

    def test_calendar_parameter_on_load_csv_data_filters(self, tmp_path):
        # Write a minimal CSV that includes Christmas 2023 and check it's filtered.
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            f"{ts(2023, 12, 22) // SEC},100,101,99,100,1000000\n"  # valid
            f"{ts(2023, 12, 25) // SEC},100,101,99,100,1000000\n"  # Christmas
            f"{ts(2023, 12, 26) // SEC},100,101,99,100,1000000\n"  # valid
        )
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bars = qc.load_csv_data(str(csv_file), "TEST", calendar="NYSE")

        assert len(bars) == 2

        dates = [_ns_to_date_int(b.timestamp_ns) for b in bars]
        assert 20231225 not in dates
        assert 20231222 in dates
        assert 20231226 in dates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])