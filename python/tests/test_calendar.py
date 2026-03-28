"""
Tests for TradingCalendar.
pandas_market_calendars is required; tests are skipped if not installed.
"""

import pytest
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc
from quantcore.calendar import TradingCalendar, _ns_to_date_int

pmc = pytest.importorskip("pandas_market_calendars", reason="pandas_market_calendars not installed")

SEC = 1_000_000_000
DAY = 86_400 * SEC


def ts(year: int, month: int, day: int) -> int:
    import datetime
    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp()) * SEC


def make_bar(year: int, month: int, day: int, symbol: str = "TEST") -> qc.BarData:
    return qc.BarData(symbol, ts(year, month, day), 100.0, 101.0, 99.0, 100.0, 1_000_000.0)


def make_bars(dates: list, symbol: str = "TEST") -> list:
    return [make_bar(*d, symbol=symbol) for d in dates]


class TestNsToDateInt:
    def test_known_date(self):
        assert _ns_to_date_int(ts(2023, 1, 3)) == 20230103

    def test_new_years_day(self):
        assert _ns_to_date_int(ts(2023, 1, 1)) == 20230101

    def test_end_of_year(self):
        assert _ns_to_date_int(ts(2022, 12, 31)) == 20221231

    def test_leap_day(self):
        assert _ns_to_date_int(ts(2024, 2, 29)) == 20240229

    def test_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _ns_to_date_int(ts(2023, 6, 15))
        assert result == 20230615


class TestInitialization:
    def test_no_op_calendar_with_none(self):
        cal = TradingCalendar(None)
        assert len(cal.filter_bars(make_bars([(2023, 1, 1), (2023, 1, 2)]))) == 2

    def test_no_op_calendar_with_empty_string(self):
        cal = TradingCalendar("")
        assert len(cal.filter_bars(make_bars([(2023, 1, 1)]))) == 1

    def test_valid_exchange_constructs(self):
        assert TradingCalendar("NYSE") is not None

    def test_invalid_exchange_raises_on_filter(self):
        cal = TradingCalendar("INVALID_EXCHANGE_XYZ")
        with pytest.raises(Exception):
            cal.filter_bars(make_bars([(2023, 6, 1)]))

    def test_missing_dependency_error_message(self):
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


class TestNYSEHolidays:
    """Exact dates verified against the NYSE holiday schedule."""

    def test_new_years_day_2023_filtered(self):
        # 2023-01-01 is Sunday; NYSE observes on Monday 2023-01-02
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 2), (2023, 1, 3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230103

    def test_new_years_day_2023_next_trading_day_passes(self):
        cal  = TradingCalendar("NYSE")
        assert len(cal.filter_bars(make_bars([(2023, 1, 3)]))) == 1

    def test_mlk_day_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 16), (2023, 1, 17)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230117

    def test_presidents_day_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 2, 20), (2023, 2, 21)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230221

    def test_good_friday_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 4, 7), (2023, 4, 6)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230406

    def test_easter_monday_not_filtered(self):
        # NYSE does not close for Easter Monday
        cal  = TradingCalendar("NYSE")
        assert len(cal.filter_bars(make_bars([(2023, 4, 10)]))) == 1

    def test_memorial_day_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 5, 29), (2023, 5, 30)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230530

    def test_juneteenth_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 6, 19), (2023, 6, 20)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230620

    def test_independence_day_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 7, 4), (2023, 7, 5)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230705

    def test_labor_day_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 9, 4), (2023, 9, 5)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230905

    def test_thanksgiving_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 11, 23), (2023, 11, 22)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20231122

    def test_christmas_2023_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 12, 25), (2023, 12, 26)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20231226

    def test_christmas_eve_2023_passes(self):
        cal  = TradingCalendar("NYSE")
        assert len(cal.filter_bars(make_bars([(2022, 12, 23)]))) == 1


class TestWeekendFiltering:
    def test_saturday_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 7), (2023, 1, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230109

    def test_sunday_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 8), (2023, 1, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230109

    def test_monday_passes(self):
        cal  = TradingCalendar("NYSE")
        assert len(cal.filter_bars(make_bars([(2023, 1, 9)]))) == 1

    def test_full_week_with_weekend(self):
        dates = [
            (2023, 1, 9),   # Mon
            (2023, 1, 10),  # Tue
            (2023, 1, 11),  # Wed
            (2023, 1, 12),  # Thu
            (2023, 1, 13),  # Fri
            (2023, 1, 14),  # Sat, filtered
            (2023, 1, 15),  # Sun, filtered
        ]
        cal  = TradingCalendar("NYSE")
        bars = make_bars(dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 5


class TestFilterBehaviour:
    def test_empty_bars_returns_empty(self):
        assert TradingCalendar("NYSE").filter_bars([]) == []

    def test_order_preserved(self):
        dates = [(2023, 1, 9), (2023, 1, 10), (2023, 1, 11)]
        cal   = TradingCalendar("NYSE")
        bars  = make_bars(dates)
        result = cal.filter_bars(bars)

        assert [b.timestamp_ns for b in result] == [b.timestamp_ns for b in bars]

    def test_symbol_preserved(self):
        cal  = TradingCalendar("NYSE")
        result = cal.filter_bars(make_bars([(2023, 1, 9)], symbol="AAPL"))
        assert result[0].symbol == "AAPL"

    def test_all_holidays_removed_raises(self):
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 2),  # New Year's observed
            (2023, 1, 7),  # Saturday
            (2023, 1, 8),  # Sunday
        ])
        with pytest.raises(RuntimeError, match="all"):
            cal.filter_bars(bars)

    def test_single_holiday_bar_raises(self):
        cal  = TradingCalendar("NYSE")
        with pytest.raises(RuntimeError, match="all"):
            cal.filter_bars(make_bars([(2023, 7, 4)]))

    def test_strict_mode_raises_above_threshold(self):
        # 3 non-trading + 1 valid = 75% removed > 20% threshold
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 2),
            (2023, 1, 7),
            (2023, 1, 8),
            (2023, 1, 9),
        ])
        with pytest.raises(RuntimeError):
            cal.filter_bars(bars, strict=True, max_skip_pct=0.20)

    def test_strict_mode_passes_below_threshold(self):
        # 1/5 = 20% == threshold (not above)
        cal = TradingCalendar("NYSE")
        bars = make_bars([
            (2023, 1, 7),   # Saturday
            (2023, 1, 9),
            (2023, 1, 10),
            (2023, 1, 11),
            (2023, 1, 12),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars, strict=True, max_skip_pct=0.20)
        assert len(result) == 4

    def test_warns_on_filtered_bars(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 7), (2023, 1, 9)])
        with pytest.warns(UserWarning, match="non-trading"):
            result = cal.filter_bars(bars)
        assert len(result) == 1

    def test_no_warning_when_nothing_filtered(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9), (2023, 1, 10)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cal.filter_bars(bars)
        assert len(result) == 2


class TestIsTradingDay:
    def test_regular_trading_day(self):
        assert TradingCalendar("NYSE").is_trading_day(ts(2023, 1, 9)) is True

    def test_saturday_not_trading_day(self):
        assert TradingCalendar("NYSE").is_trading_day(ts(2023, 1, 7)) is False

    def test_sunday_not_trading_day(self):
        assert TradingCalendar("NYSE").is_trading_day(ts(2023, 1, 8)) is False

    def test_holiday_not_trading_day(self):
        assert TradingCalendar("NYSE").is_trading_day(ts(2023, 7, 4)) is False

    def test_no_op_calendar_always_true(self):
        cal = TradingCalendar(None)
        assert cal.is_trading_day(ts(2023, 1, 1)) is True
        assert cal.is_trading_day(ts(2023, 1, 7)) is True


class TestCaching:
    def test_repeated_calls_return_same_result(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 1, 9), (2023, 1, 16)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = cal.filter_bars(bars)
            result2 = cal.filter_bars(bars)

        assert len(result1) == len(result2)
        assert [b.timestamp_ns for b in result1] == [b.timestamp_ns for b in result2]

    def test_cache_used_on_second_call(self):
        cal  = TradingCalendar("NYSE")
        bars = make_bars([(2023, 6, 1)])

        cal.filter_bars(bars)
        cache_after_first = id(cal._cache)

        cal.filter_bars(bars)
        cache_after_second = id(cal._cache)

        assert cache_after_first == cache_after_second

    def test_cache_invalidated_on_extended_range(self):
        cal = TradingCalendar("NYSE")

        cal.filter_bars(make_bars([(2023, 6, 1)]))
        first_range = cal._cache_range

        cal.filter_bars(make_bars([(2021, 6, 1)]))
        second_range = cal._cache_range

        assert second_range != first_range


class TestMultiExchange:
    def test_lse_filters_uk_bank_holiday(self):
        # 2023-05-08: extra bank holiday for King Charles III coronation
        cal  = TradingCalendar("LSE")
        bars = make_bars([(2023, 5, 8), (2023, 5, 9)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230509

    def test_tsx_filters_canadian_holiday(self):
        # 2023-07-03: Canada Day observed
        cal  = TradingCalendar("TSX")
        bars = make_bars([(2023, 7, 3), (2023, 7, 4)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cal.filter_bars(bars)
        assert len(result) == 1
        assert _ns_to_date_int(result[0].timestamp_ns) == 20230704

    def test_different_exchanges_have_different_holidays(self):
        # NYSE stays open on Veterans Day
        nyse_cal = TradingCalendar("NYSE")
        assert len(nyse_cal.filter_bars(make_bars([(2023, 11, 10)]))) == 1


class TestPublicAPIIntegration:
    def test_run_backtest_with_calendar_filters_holidays(self):
        dates_with_holiday = [
            (2023, 12, 22),
            (2023, 12, 25),  # Christmas, filtered
            (2023, 12, 26),
        ]
        dates_without_holiday = [(2023, 12, 22), (2023, 12, 26)]

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
            qc.run_backtest(strat_filtered, {"TEST": make_bars(dates_with_holiday)}, calendar="NYSE")

        qc.run_backtest(strat_unfiltered, {"TEST": make_bars(dates_without_holiday)})

        assert strat_filtered.count == strat_unfiltered.count == 2

    def test_run_backtest_without_calendar_includes_holidays(self):
        dates = [(2023, 12, 22), (2023, 12, 25), (2023, 12, 26)]

        class CountingStrategy(qc.Strategy):
            def __init__(self):
                super().__init__("Counter")
                self.count = 0

            def on_data(self, event):
                self.count += 1

        strat = CountingStrategy()
        qc.run_backtest(strat, {"TEST": make_bars(dates)})
        assert strat.count == 3

    def test_run_backtest_multi_symbol_calendar_applied_to_all(self):
        holiday_and_two_valid = [
            (2023, 12, 22),
            (2023, 12, 25),  # Christmas, filtered
            (2023, 12, 26),
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
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            f"{ts(2023, 12, 22) // SEC},100,101,99,100,1000000\n"
            f"{ts(2023, 12, 25) // SEC},100,101,99,100,1000000\n"  # Christmas
            f"{ts(2023, 12, 26) // SEC},100,101,99,100,1000000\n"
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
