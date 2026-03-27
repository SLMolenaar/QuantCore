"""
Tests for CorporateActionsAdjuster

All tests use exact expected values derived from manual calculations.

Adjustment math (backward adjustment, CRSP methodology):
  Split  : pre-split prices are divided by ratio, pre-split volumes multiplied by ratio
  Dividend: factor = (close_on_ex_date - dividend) / close_on_ex_date
             pre-dividend prices are multiplied by factor

All adjustments compound from newest to oldest when multiple events exist.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc
from quantcore.corporate_actions import CorporateActionsAdjuster, SplitEvent, DividendEvent

# ============================================================================
# HELPERS
# ============================================================================

SEC = 1_000_000_000  # one second in nanoseconds


def make_bars(prices, volume=1_000_000.0, symbol="TEST", base_ts=0):
    """Create a BarSeries from a list of close prices. open=high=low=close."""
    return [
        qc.BarData(symbol, base_ts + i * SEC, p, p, p, p, volume)
        for i, p in enumerate(prices)
    ]


def make_bars_with_volume(price_volume_pairs, symbol="TEST", base_ts=0):
    """Create a BarSeries from (price, volume) pairs."""
    return [
        qc.BarData(symbol, base_ts + i * SEC, p, p, p, p, v)
        for i, (p, v) in enumerate(price_volume_pairs)
    ]


def closes(bars):
    return [b.close for b in bars]


def volumes(bars):
    return [b.volume for b in bars]


# ============================================================================
# INITIALIZATION & VALIDATION
# ============================================================================

class TestInitialization:
    def test_empty_adjuster_returns_bars_unchanged(self):
        adjuster = CorporateActionsAdjuster()
        bars = make_bars([100.0, 101.0, 102.0])
        result = adjuster.adjust(bars)

        assert len(result) == 3
        assert closes(result) == [100.0, 101.0, 102.0]

    def test_empty_input_returns_empty(self):
        adjuster = CorporateActionsAdjuster()
        result = adjuster.adjust([])
        assert result == []

    def test_add_split_invalid_ratio_zero(self):
        adjuster = CorporateActionsAdjuster()
        with pytest.raises(ValueError):
            adjuster.add_split(ex_date_ns=1 * SEC, ratio=0.0)

    def test_add_split_invalid_ratio_negative(self):
        adjuster = CorporateActionsAdjuster()
        with pytest.raises(ValueError):
            adjuster.add_split(ex_date_ns=1 * SEC, ratio=-2.0)

    def test_add_dividend_invalid_negative_amount(self):
        adjuster = CorporateActionsAdjuster()
        with pytest.raises(ValueError):
            adjuster.add_dividend(ex_date_ns=1 * SEC, amount=-0.01)

    def test_add_dividend_zero_amount_accepted(self):
        # Zero dividend is a no-op but must not raise
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=1 * SEC, amount=0.0)

    def test_method_chaining_returns_self(self):
        adjuster = CorporateActionsAdjuster()
        result = (
            adjuster
            .add_split(ex_date_ns=2 * SEC, ratio=2.0)
            .add_dividend(ex_date_ns=1 * SEC, amount=1.0)
        )
        assert result is adjuster


# ============================================================================
# SPLIT ADJUSTMENT; PRICE
# ============================================================================

class TestSplitPrice:
    def test_two_for_one_split_halves_pre_split_prices(self):
        # 5 bars: bars 0-2 are pre-split, ex_date falls between bar 2 and bar 3
        # ex_date_ns = 3*SEC means "on or after bar at ts=3s" → bars 0,1,2 are adjusted
        bars = make_bars([200.0, 210.0, 220.0, 110.0, 115.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        # ex_date is at ts=3s; bars at ts=0,1,2 are before it → adjusted
        adjuster.add_split(ex_date_ns=3 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        # pre-split prices divided by 2.0
        assert result[0].close == pytest.approx(100.0)
        assert result[1].close == pytest.approx(105.0)
        assert result[2].close == pytest.approx(110.0)
        # post-split prices unchanged
        assert result[3].close == pytest.approx(110.0)
        assert result[4].close == pytest.approx(115.0)

    def test_three_for_one_split(self):
        bars = make_bars([300.0, 310.0, 100.0, 105.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=3.0)
        result = adjuster.adjust(bars)

        assert result[0].close == pytest.approx(100.0)
        assert result[1].close == pytest.approx(310.0 / 3.0)
        assert result[2].close == pytest.approx(100.0)
        assert result[3].close == pytest.approx(105.0)

    def test_three_for_two_split(self):
        # ratio = 3/2 = 1.5
        bars = make_bars([150.0, 100.0, 102.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=1.5)
        result = adjuster.adjust(bars)

        assert result[0].close == pytest.approx(150.0 / 1.5)
        assert result[1].close == pytest.approx(100.0)
        assert result[2].close == pytest.approx(102.0)

    def test_reverse_split_multiplies_pre_split_prices(self):
        # 1-for-2 reverse split: ratio = 0.5
        # pre-split shares halved → price doubles
        bars = make_bars([5.0, 6.0, 10.0, 11.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=0.5)
        result = adjuster.adjust(bars)

        # price divided by 0.5 = multiplied by 2
        assert result[0].close == pytest.approx(10.0)
        assert result[1].close == pytest.approx(12.0)
        assert result[2].close == pytest.approx(10.0)
        assert result[3].close == pytest.approx(11.0)

    def test_split_adjusts_all_ohlc_fields(self):
        bars = [qc.BarData("TEST", 0, 100.0, 110.0, 90.0, 105.0, 1_000_000.0)]
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert result[0].open  == pytest.approx(50.0)
        assert result[0].high  == pytest.approx(55.0)
        assert result[0].low   == pytest.approx(45.0)
        assert result[0].close == pytest.approx(52.5)

    def test_split_on_last_bar_adjusts_all_bars(self):
        # ex_date is after the last bar, all bars are pre-split
        bars = make_bars([100.0, 110.0, 120.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=100 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert result[0].close == pytest.approx(50.0)
        assert result[1].close == pytest.approx(55.0)
        assert result[2].close == pytest.approx(60.0)

    def test_split_before_first_bar_adjusts_nothing(self):
        # ex_date is before or at the first bar, no bars are pre-split
        bars = make_bars([100.0, 110.0, 120.0], base_ts=10 * SEC)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=5 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert result[0].close == pytest.approx(100.0)
        assert result[1].close == pytest.approx(110.0)
        assert result[2].close == pytest.approx(120.0)


# ============================================================================
# SPLIT ADJUSTMENT; VOLUME
# ============================================================================

class TestSplitVolume:
    def test_two_for_one_split_doubles_pre_split_volume(self):
        bars = make_bars_with_volume(
            [(200.0, 500_000), (210.0, 600_000), (105.0, 1_100_000), (110.0, 1_200_000)],
        )
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        # pre-split volumes multiplied by 2
        assert result[0].volume == pytest.approx(1_000_000.0)
        assert result[1].volume == pytest.approx(1_200_000.0)
        # post-split volumes unchanged
        assert result[2].volume == pytest.approx(1_100_000.0)
        assert result[3].volume == pytest.approx(1_200_000.0)

    def test_reverse_split_halves_pre_split_volume(self):
        # 1-for-2 reverse split: ratio = 0.5
        bars = make_bars_with_volume([(10.0, 2_000_000), (5.0, 1_000_000), (10.5, 500_000)])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=0.5)
        result = adjuster.adjust(bars)

        # pre-split volume multiplied by 0.5
        assert result[0].volume == pytest.approx(1_000_000.0)
        assert result[1].volume == pytest.approx(1_000_000.0)
        assert result[2].volume == pytest.approx(500_000.0)


# ============================================================================
# DIVIDEND ADJUSTMENT
# ============================================================================

class TestDividendAdjustment:
    def test_cash_dividend_reduces_pre_dividend_prices(self):
        # Bars: ts=0,1,2. ex_date at ts=2 → bars 0,1 are pre-dividend.
        # Bar at ts=1 (close=100) is used for the factor calculation.
        # factor = (100 - 2) / 100 = 0.98
        bars = make_bars([98.0, 100.0, 98.0, 99.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=2 * SEC, amount=2.0)
        result = adjuster.adjust(bars)

        # factor = (100 - 2) / 100 = 0.98
        assert result[0].close == pytest.approx(98.0 * 0.98)
        assert result[1].close == pytest.approx(100.0 * 0.98)
        # post-dividend bars unchanged
        assert result[2].close == pytest.approx(98.0)
        assert result[3].close == pytest.approx(99.0)

    def test_dividend_does_not_adjust_volume(self):
        bars = make_bars_with_volume([(100.0, 1_000_000), (102.0, 1_200_000), (100.0, 900_000)])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=2 * SEC, amount=2.0)
        result = adjuster.adjust(bars)

        # volumes must be untouched for dividends
        assert result[0].volume == pytest.approx(1_000_000.0)
        assert result[1].volume == pytest.approx(1_200_000.0)
        assert result[2].volume == pytest.approx(900_000.0)

    def test_dividend_larger_than_close_produces_no_adjustment(self):
        # If dividend >= close, factor would be <= 0 which is nonsensical.
        # The adjuster skips the adjustment rather than producing negative prices.
        bars = make_bars([50.0, 100.0, 98.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=1 * SEC, amount=150.0)
        result = adjuster.adjust(bars)

        # bar 0 should be unchanged (factor skipped)
        assert result[0].close == pytest.approx(50.0)
        assert result[1].close == pytest.approx(100.0)
        assert result[2].close == pytest.approx(98.0)

    def test_zero_dividend_leaves_prices_unchanged(self):
        bars = make_bars([100.0, 102.0, 104.0])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=1 * SEC, amount=0.0)
        result = adjuster.adjust(bars)

        assert closes(result) == pytest.approx([100.0, 102.0, 104.0])

    def test_small_dividend_precision(self):
        # $0.23 dividend on a $150 stock
        # factor = (150 - 0.23) / 150 = 149.77 / 150
        bars = make_bars([148.0, 150.0, 149.77, 151.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=2 * SEC, amount=0.23)
        result = adjuster.adjust(bars)

        factor = (150.0 - 0.23) / 150.0
        assert result[0].close == pytest.approx(148.0 * factor)
        assert result[1].close == pytest.approx(150.0 * factor)
        assert result[2].close == pytest.approx(149.77)
        assert result[3].close == pytest.approx(151.0)


# ============================================================================
# MULTIPLE EVENTS
# ============================================================================

class TestMultipleEvents:
    def test_two_splits_compound_correctly(self):
        # Split 1: 2:1 at ts=2 → bars 0,1 divided by 2.0
        # Split 2: 2:1 at ts=4 → bars 0,1,2,3 divided by 2.0
        # Combined: bars 0,1 divided by 4.0; bars 2,3 divided by 2.0
        bars = make_bars([400.0, 420.0, 200.0, 210.0, 100.0, 105.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=2.0)
        adjuster.add_split(ex_date_ns=4 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        # bars 0,1: divided by 2 (split at ts=2) then by 2 again (split at ts=4) = /4
        assert result[0].close == pytest.approx(100.0)
        assert result[1].close == pytest.approx(105.0)
        # bars 2,3: divided by 2 (split at ts=4 only)
        assert result[2].close == pytest.approx(100.0)
        assert result[3].close == pytest.approx(105.0)
        # bars 4,5: post both splits, unchanged
        assert result[4].close == pytest.approx(100.0)
        assert result[5].close == pytest.approx(105.0)

    def test_split_then_dividend_compound_correctly(self):
        # Prices: [200, 210, 105, 110, 108, 112]
        # Split 2:1 at ts=2 → bars 0,1 divided by 2
        # Dividend $1.05 at ts=4 (bar before = bar at ts=3, close=110)
        # factor = (110 - 1.05) / 110 = 108.95 / 110
        # bars 0,1,2,3 get dividend factor; bars 0,1 also get split factor
        prices = [200.0, 210.0, 105.0, 110.0, 108.95, 112.0]
        bars = make_bars(prices, base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=2.0)
        adjuster.add_dividend(ex_date_ns=4 * SEC, amount=1.05)
        result = adjuster.adjust(bars)

        div_factor = (110.0 - 1.05) / 110.0

        # bar 0: split (/2) then dividend factor
        assert result[0].close == pytest.approx(200.0 / 2.0 * div_factor)
        assert result[1].close == pytest.approx(210.0 / 2.0 * div_factor)
        # bar 2,3: only dividend factor
        assert result[2].close == pytest.approx(105.0 * div_factor)
        assert result[3].close == pytest.approx(110.0 * div_factor)
        # post-dividend bars: unchanged
        assert result[4].close == pytest.approx(108.95)
        assert result[5].close == pytest.approx(112.0)

    def test_events_applied_regardless_of_registration_order(self):
        # Register events in reverse chronological order, result must be the same.
        bars = make_bars([400.0, 200.0, 100.0, 105.0], base_ts=0)

        adjuster_forward = CorporateActionsAdjuster()
        adjuster_forward.add_split(ex_date_ns=1 * SEC, ratio=2.0)
        adjuster_forward.add_split(ex_date_ns=2 * SEC, ratio=2.0)

        adjuster_reverse = CorporateActionsAdjuster()
        adjuster_reverse.add_split(ex_date_ns=2 * SEC, ratio=2.0)
        adjuster_reverse.add_split(ex_date_ns=1 * SEC, ratio=2.0)

        result_f = adjuster_forward.adjust(bars)
        result_r = adjuster_reverse.adjust(bars)

        assert closes(result_f) == pytest.approx(closes(result_r))

    def test_two_dividends_compound_correctly(self):
        # Dividend $1 at ts=2 (bar before = bar at ts=1, close=100), factor1 = 0.99
        # Dividend $2 at ts=4 (bar before = bar at ts=3, close=99.0, unadjusted)
        #   factor2 = (99 - 2) / 99
        # bars 0,1 receive both factors; bars 2,3 receive only factor2; bars 4,5 unchanged
        bars = make_bars([99.0, 100.0, 99.0, 99.0, 97.0, 98.0], base_ts=0)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_dividend(ex_date_ns=2 * SEC, amount=1.0)
        adjuster.add_dividend(ex_date_ns=4 * SEC, amount=2.0)
        result = adjuster.adjust(bars)

        # factor1 uses unadjusted close of bar at ts=1 = 100.0
        factor1 = (100.0 - 1.0) / 100.0

        # factor2 uses unadjusted close of bar at ts=3 = 99.0
        factor2 = (99.0 - 2.0) / 99.0

        assert result[0].close == pytest.approx(99.0  * factor1 * factor2)
        assert result[1].close == pytest.approx(100.0 * factor1 * factor2)
        assert result[2].close == pytest.approx(99.0  * factor2)
        assert result[3].close == pytest.approx(99.0  * factor2)
        assert result[4].close == pytest.approx(97.0)
        assert result[5].close == pytest.approx(98.0)

# ============================================================================
# REAL-WORLD SCENARIO; AAPL 7-FOR-1 SPLIT
# ============================================================================

class TestAppleSplitScenario:
    def test_aapl_7for1_split_june_2014(self):
        pre_split_price  = 645.57
        post_split_price = 92.22
        split_ts = 3 * SEC  # ex_date between bar[2] and bar[3]

        bars = make_bars(
            [640.0, 643.0, pre_split_price, post_split_price, 93.0],
            base_ts=0,
        )

        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=split_ts, ratio=7.0)
        result = adjuster.adjust(bars)

        # bars at ts=0,1,2 are pre-split → divided by 7
        assert result[0].close == pytest.approx(640.0 / 7.0)
        assert result[1].close == pytest.approx(643.0 / 7.0)
        assert result[2].close == pytest.approx(pre_split_price / 7.0)
        # bars at ts=3,4 are post-split → unchanged
        assert result[3].close == pytest.approx(post_split_price)
        assert result[4].close == pytest.approx(93.0)

    def test_adjusted_series_is_continuous_across_split(self):
        # The adjusted close on the last pre-split bar should equal the
        # post-split open, within floating-point tolerance.
        pre_close = 700.0
        post_open = 100.0  # 700 / 7 = 100 exactly
        split_ts  = 2 * SEC

        bars = [
            qc.BarData("AAPL", 0,       pre_close, pre_close, pre_close, pre_close, 1_000_000.0),
            qc.BarData("AAPL", 1 * SEC, pre_close, pre_close, pre_close, pre_close, 1_000_000.0),
            qc.BarData("AAPL", 2 * SEC, post_open, post_open, post_open, post_open, 7_000_000.0),
        ]

        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=split_ts, ratio=7.0)
        result = adjuster.adjust(bars)

        assert result[1].close == pytest.approx(result[2].open)


# ============================================================================
# DATA INTEGRITY
# ============================================================================

class TestDataIntegrity:
    def test_original_bars_not_modified(self):
        bars = make_bars([100.0, 110.0, 120.0])
        original_closes = [b.close for b in bars]

        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=2.0)
        adjuster.adjust(bars)

        # Original objects must be untouched
        assert [b.close for b in bars] == original_closes

    def test_adjusted_bars_have_correct_symbol(self):
        bars = make_bars([100.0, 110.0], symbol="AAPL")
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert all(b.symbol == "AAPL" for b in result)

    def test_adjusted_bars_have_correct_timestamps(self):
        bars = make_bars([100.0, 110.0, 120.0], base_ts=1_000 * SEC)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1_001 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert [b.timestamp_ns for b in result] == [b.timestamp_ns for b in bars]

    def test_adjusted_bars_satisfy_ohlc_constraints(self):
        # high >= open, close, low >= 0 must hold after adjustment
        bars = [qc.BarData("TEST", i * SEC, 100.0, 110.0, 90.0, 105.0, 1_000_000.0)
                for i in range(5)]

        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=3 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        for b in result:
            assert b.high  >= b.open
            assert b.high  >= b.close
            assert b.low   <= b.open
            assert b.low   <= b.close
            assert b.low   >= 0.0

    def test_result_length_equals_input_length(self):
        bars = make_bars([100.0] * 50)
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=25 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert len(result) == 50

    def test_all_adjusted_prices_are_positive(self):
        bars = make_bars([100.0, 200.0, 300.0, 150.0, 160.0])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=3 * SEC, ratio=2.0)
        adjuster.add_dividend(ex_date_ns=4 * SEC, amount=1.0)
        result = adjuster.adjust(bars)

        for b in result:
            assert b.open  > 0.0
            assert b.high  > 0.0
            assert b.low   > 0.0
            assert b.close > 0.0

    def test_single_bar_adjusted_correctly(self):
        bars = make_bars([200.0])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=2.0)
        result = adjuster.adjust(bars)

        assert len(result) == 1
        assert result[0].close == pytest.approx(100.0)


# ============================================================================
# ROUND-TRIP CORRECTNESS
# ============================================================================

class TestRoundTrip:
    def test_split_and_reverse_split_cancel_out(self):
        # Applying a 2:1 split then a 1:2 reverse split on the same bar
        # should return the original price.
        bars = make_bars([100.0, 50.0, 50.0], base_ts=0)

        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=1 * SEC, ratio=2.0)   # forward split at ts=1
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=0.5)   # reverse split at ts=2
        result = adjuster.adjust(bars)

        # bar 0 is pre-both: divided by 2 then by 0.5 = divided by 1 → unchanged
        assert result[0].close == pytest.approx(100.0)

    def test_returns_identical_with_and_without_post_series_events(self):
        # An event that falls after all bars should not affect any bar.
        bars = make_bars([100.0, 110.0, 120.0], base_ts=0)

        adjuster_no_event = CorporateActionsAdjuster()
        adjuster_with_event = CorporateActionsAdjuster()
        adjuster_with_event.add_split(ex_date_ns=1000 * SEC, ratio=2.0)

        result_no   = adjuster_no_event.adjust(bars)
        result_with = adjuster_with_event.adjust(bars)

        # Event is after all bars, so all bars are adjusted by the same factor.
        # Both should be divided by 2 since all bars precede the event.
        assert closes(result_no) == pytest.approx([100.0, 110.0, 120.0])
        assert closes(result_with) == pytest.approx([50.0, 55.0, 60.0])


# ============================================================================
# BACKTEST INTEGRATION
# ============================================================================

class TestBacktestIntegration:
    def test_adjusted_bars_produce_correct_returns(self):
        # Without adjustment, a 2:1 split looks like a 50% crash → BuyAndHold loses money.
        # With adjustment, the series is continuous → BuyAndHold breaks even (flat market).
        pre_split   = 200.0
        post_split  = 100.0   # unadjusted post-split price
        n_bars      = 10
        split_ts    = n_bars * SEC

        unadjusted_bars = (
            make_bars([pre_split] * n_bars, base_ts=0) +
            make_bars([post_split] * n_bars, base_ts=split_ts)
        )

        # Adjusted: all pre-split bars divided by 2 → 100.0, continuity restored
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=split_ts, ratio=2.0)
        adjusted_bars = adjuster.adjust(unadjusted_bars)

        # Run backtest on unadjusted data
        engine_raw = qc.BacktestEngine(100_000.0)
        limits = qc.RiskLimits()
        limits.enabled = False
        engine_raw.set_risk_limits(limits)
        engine_raw.add_data("TEST", unadjusted_bars)
        engine_raw.set_strategy(qc.BuyAndHold())
        pnl_raw = engine_raw.get_total_pnl()
        engine_raw.run()
        pnl_raw = engine_raw.get_total_pnl()

        # Run backtest on adjusted data
        engine_adj = qc.BacktestEngine(100_000.0)
        engine_adj.set_risk_limits(limits)
        engine_adj.add_data("TEST", adjusted_bars)
        engine_adj.set_strategy(qc.BuyAndHold())
        engine_adj.run()
        pnl_adj = engine_adj.get_total_pnl()

        # Unadjusted: strategy opens at $200, "price crashes" to $100 → large loss
        # Adjusted: flat market → small negative (just fees)
        assert pnl_raw < pnl_adj, (
            "Unadjusted split should produce worse PnL than adjusted series"
        )
        # Adjusted series is flat → PnL should be just fees (small negative)
        assert abs(pnl_adj) < 1000.0, (
            "Adjusted flat series should produce near-zero PnL (fees only)"
        )

    def test_adjusted_data_is_valid_bardata(self):
        # Ensure the adjuster output can be loaded into the engine without errors.
        bars = make_bars([100.0, 200.0, 150.0, 160.0, 155.0])
        adjuster = CorporateActionsAdjuster()
        adjuster.add_split(ex_date_ns=2 * SEC, ratio=2.0)
        adjusted = adjuster.adjust(bars)

        engine = qc.BacktestEngine(100_000.0)
        # Should not raise
        engine.add_data("TEST", adjusted)
        engine.set_strategy(qc.BuyAndHold())
        engine.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
