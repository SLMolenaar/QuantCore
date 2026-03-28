"""
Tests for position_sizing module (PositionCalculator and PortfolioPositionSizer).
All tests use exact expected values derived from manual calculations.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantcore.position_sizing import PositionCalculator, PortfolioPositionSizer, PositionSizeResult


class TestPositionCalculatorInit:
    def test_positive_capital_accepted(self):
        calc = PositionCalculator(100000.0)
        assert calc.capital == 100000.0

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            PositionCalculator(0.0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            PositionCalculator(-1000.0)

    def test_default_max_position_pct(self):
        calc = PositionCalculator(100000.0)
        assert calc.max_position_pct == 0.2

    def test_update_capital(self):
        calc = PositionCalculator(100000.0)
        calc.update_capital(200000.0)
        assert calc.capital == 200000.0

    def test_update_capital_zero_raises(self):
        calc = PositionCalculator(100000.0)
        with pytest.raises(ValueError):
            calc.update_capital(0.0)

    def test_result_is_position_size_result(self):
        calc   = PositionCalculator(100000.0)
        result = calc.fixed_percentage(100.0, 0.1)
        assert isinstance(result, PositionSizeResult)


class TestFixedPercentage:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.2)

    def test_basic(self):
        # 10% of $100k = $10k / $100 = 100 shares
        result = self.calc.fixed_percentage(price=100.0, percentage=0.1)
        assert result.quantity           == pytest.approx(100.0)
        assert result.notional_value     == pytest.approx(10000.0)
        assert result.percent_of_capital == pytest.approx(0.1)

    def test_capped_by_max_position_pct(self):
        # 50% requested but max_position_pct=0.2 -> capped to 20%
        # $100k * 0.2 = $20k / $100 = 200 shares
        result = self.calc.fixed_percentage(price=100.0, percentage=0.5)
        assert result.quantity == pytest.approx(200.0)

    def test_high_price(self):
        # 10% of $100k = $10k / $500 = 20 shares
        result = self.calc.fixed_percentage(price=500.0, percentage=0.1)
        assert result.quantity       == pytest.approx(20.0)
        assert result.notional_value == pytest.approx(10000.0)

    def test_low_price(self):
        # 10% of $100k = $10k / $1 = 10000 shares
        result = self.calc.fixed_percentage(price=1.0, percentage=0.1)
        assert result.quantity == pytest.approx(10000.0)

    def test_min_quantity_enforced(self):
        # Very small allocation relative to price -> raw qty < min_quantity
        calc   = PositionCalculator(capital=10.0, max_position_pct=1.0)
        result = calc.fixed_percentage(price=1000.0, percentage=0.01, min_quantity=5)
        assert result.quantity >= 5

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            self.calc.fixed_percentage(price=0.0, percentage=0.1)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            self.calc.fixed_percentage(price=-100.0, percentage=0.1)

    def test_percent_of_capital_consistent(self):
        result = self.calc.fixed_percentage(price=150.0, percentage=0.15)
        assert result.percent_of_capital == pytest.approx(
            result.notional_value / self.calc.capital, rel=1e-9
        )

    def test_capital_update_affects_sizing(self):
        result_before = self.calc.fixed_percentage(price=100.0, percentage=0.1)
        self.calc.update_capital(200000.0)
        result_after  = self.calc.fixed_percentage(price=100.0, percentage=0.1)
        assert result_after.quantity == pytest.approx(result_before.quantity * 2)


class TestRiskBased:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.2)

    def test_basic(self):
        # risk_amount = $100k * 0.01 = $1000
        # risk_per_unit = |100 - 90| = $10
        # qty = 1000 / 10 = 100; notional = $10k < max $20k
        result = self.calc.risk_based(price=100.0, stop_loss_price=90.0, risk_per_trade=0.01)
        assert result.quantity       == pytest.approx(100.0)
        assert result.notional_value == pytest.approx(10000.0)

    def test_tight_stop_capped_by_max(self):
        # stop at 99: risk_per_unit=$1, qty_raw=1000 -> notional=$100k > max $20k -> capped at 200
        result = self.calc.risk_based(price=100.0, stop_loss_price=99.0, risk_per_trade=0.01)
        assert result.quantity == pytest.approx(200.0)

    def test_wide_stop_produces_smaller_position(self):
        # stop at 50: risk_per_unit=$50, qty=1000/50=20
        result = self.calc.risk_based(price=100.0, stop_loss_price=50.0, risk_per_trade=0.01)
        assert result.quantity == pytest.approx(20.0)

    def test_capped_notional_does_not_exceed_max(self):
        result = self.calc.risk_based(price=100.0, stop_loss_price=99.0, risk_per_trade=0.01)
        assert result.notional_value <= self.calc.capital * self.calc.max_position_pct + 1e-9

    def test_stop_above_entry_for_short(self):
        # stop above entry (short): risk_per_unit = |100 - 110| = 10, qty=100
        result = self.calc.risk_based(price=100.0, stop_loss_price=110.0, risk_per_trade=0.01)
        assert result.quantity == pytest.approx(100.0)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            self.calc.risk_based(price=0.0, stop_loss_price=90.0)

    def test_zero_stop_raises(self):
        with pytest.raises(ValueError):
            self.calc.risk_based(price=100.0, stop_loss_price=0.0)

    def test_entry_equals_stop_raises(self):
        with pytest.raises(ValueError):
            self.calc.risk_based(price=100.0, stop_loss_price=100.0)

    def test_higher_risk_per_trade_produces_larger_position(self):
        small = self.calc.risk_based(price=100.0, stop_loss_price=90.0, risk_per_trade=0.01)
        large = self.calc.risk_based(price=100.0, stop_loss_price=90.0, risk_per_trade=0.02)
        assert large.quantity > small.quantity


class TestKellyCriterion:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.5)

    def test_basic_full_kelly(self):
        # wl_ratio=2.0, kelly_pct=(0.6*2.0 - 0.4)/2.0 = 0.4 -> qty = $40k/$100 = 400
        result = self.calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=1.0
        )
        assert result.quantity == pytest.approx(400.0)

    def test_half_kelly(self):
        # 0.4 * 0.5 = 0.2 -> qty = 200
        result = self.calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=0.5
        )
        assert result.quantity == pytest.approx(200.0)

    def test_quarter_kelly(self):
        # 0.4 * 0.25 = 0.1 -> qty = 100
        result = self.calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=0.25
        )
        assert result.quantity == pytest.approx(100.0)

    def test_losing_strategy_returns_min_quantity(self):
        # kelly_pct = max(0, -0.4) = 0 -> notional=0 -> min_quantity=1
        result = self.calc.kelly_criterion(
            price=100.0, win_rate=0.3, avg_win=1.0, avg_loss=1.0, kelly_fraction=1.0
        )
        assert result.quantity == pytest.approx(1.0)

    def test_breakeven_returns_min_quantity(self):
        # win_rate=0.5, wl=1.0: kelly_pct=0 -> min_quantity
        result = self.calc.kelly_criterion(
            price=100.0, win_rate=0.5, avg_win=1.0, avg_loss=1.0, kelly_fraction=1.0
        )
        assert result.quantity == pytest.approx(1.0)

    def test_capped_by_max_position_pct(self):
        # kelly_pct=0.4 but max=0.2 -> notional=$20k -> qty=200
        calc   = PositionCalculator(capital=100000.0, max_position_pct=0.2)
        result = calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=1.0
        )
        assert result.notional_value <= 100000.0 * 0.2 + 1e-9

    def test_zero_avg_loss_raises(self):
        with pytest.raises(ValueError):
            self.calc.kelly_criterion(price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=0.0)

    def test_invalid_win_rate_raises(self):
        with pytest.raises(ValueError):
            self.calc.kelly_criterion(price=100.0, win_rate=1.1, avg_win=2.0, avg_loss=1.0)
        with pytest.raises(ValueError):
            self.calc.kelly_criterion(price=100.0, win_rate=-0.1, avg_win=2.0, avg_loss=1.0)

    def test_fraction_scales_linearly(self):
        full = self.calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=1.0
        )
        half = self.calc.kelly_criterion(
            price=100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=0.5
        )
        assert half.quantity == pytest.approx(full.quantity * 0.5, rel=1e-9)


class TestEqualWeight:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.2)

    def test_five_positions(self):
        # weight = min(1/5, 0.2) = 0.2 -> qty = 200
        result = self.calc.equal_weight(price=100.0, num_positions=5)
        assert result.quantity == pytest.approx(200.0)

    def test_ten_positions(self):
        # weight = min(0.1, 0.2) = 0.1 -> qty = 100
        result = self.calc.equal_weight(price=100.0, num_positions=10)
        assert result.quantity == pytest.approx(100.0)

    def test_single_position_capped_by_max(self):
        # 1/1=1.0 but max=0.2 -> qty=200
        result = self.calc.equal_weight(price=100.0, num_positions=1)
        assert result.quantity == pytest.approx(200.0)

    def test_many_positions_uncapped(self):
        # 1/50=0.02 < 0.2 -> qty=20
        result = self.calc.equal_weight(price=100.0, num_positions=50)
        assert result.quantity == pytest.approx(20.0)

    def test_zero_positions_raises(self):
        with pytest.raises(ValueError):
            self.calc.equal_weight(price=100.0, num_positions=0)

    def test_negative_positions_raises(self):
        with pytest.raises(ValueError):
            self.calc.equal_weight(price=100.0, num_positions=-3)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            self.calc.equal_weight(price=0.0, num_positions=5)

    def test_more_positions_smaller_size(self):
        r5  = self.calc.equal_weight(price=100.0, num_positions=5)
        r10 = self.calc.equal_weight(price=100.0, num_positions=10)
        assert r10.quantity < r5.quantity


class TestVolatilityAdjusted:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.2)

    def test_vol_equals_target(self):
        # allocation = min(0.1 * 1.0, 0.2) = 0.1 -> qty=100
        result = self.calc.volatility_adjusted(
            price=100.0, volatility=0.15, target_volatility=0.15, base_allocation=0.1
        )
        assert result.quantity == pytest.approx(100.0)

    def test_high_vol_reduces_position(self):
        # vol=0.30: allocation = min(0.1 * 0.5, 0.2) = 0.05 -> qty=50
        result = self.calc.volatility_adjusted(
            price=100.0, volatility=0.30, target_volatility=0.15, base_allocation=0.1
        )
        assert result.quantity == pytest.approx(50.0)

    def test_low_vol_increases_position(self):
        # vol=0.05: allocation = min(0.1 * 3.0, 0.2) = 0.2 -> qty=200
        result = self.calc.volatility_adjusted(
            price=100.0, volatility=0.05, target_volatility=0.15, base_allocation=0.1
        )
        assert result.quantity == pytest.approx(200.0)

    def test_capped_by_max_position_pct(self):
        result = self.calc.volatility_adjusted(
            price=100.0, volatility=0.01, target_volatility=0.15, base_allocation=0.1
        )
        assert result.notional_value <= self.calc.capital * self.calc.max_position_pct + 1e-9

    def test_zero_volatility_raises(self):
        with pytest.raises(ValueError):
            self.calc.volatility_adjusted(price=100.0, volatility=0.0)

    def test_higher_vol_produces_smaller_position(self):
        low  = self.calc.volatility_adjusted(price=100.0, volatility=0.10)
        high = self.calc.volatility_adjusted(price=100.0, volatility=0.30)
        assert low.quantity > high.quantity


class TestLeveraged:
    def setup_method(self):
        self.calc = PositionCalculator(capital=100000.0, max_position_pct=0.2)

    def test_two_times_leverage(self):
        # target = min($100k*0.1*2, $100k*0.2*2) = min($20k, $40k) = $20k -> qty=200
        result = self.calc.leveraged(price=100.0, leverage=2.0, base_percentage=0.1)
        assert result.quantity == pytest.approx(200.0)

    def test_one_times_leverage(self):
        # target = min($100k*0.1*1, $100k*0.2*1) = min($10k, $20k) = $10k -> qty=100
        result = self.calc.leveraged(price=100.0, leverage=1.0, base_percentage=0.1)
        assert result.quantity == pytest.approx(100.0)

    def test_zero_leverage_raises(self):
        with pytest.raises(ValueError):
            self.calc.leveraged(price=100.0, leverage=0.0)

    def test_negative_leverage_raises(self):
        with pytest.raises(ValueError):
            self.calc.leveraged(price=100.0, leverage=-1.0)

    def test_higher_leverage_produces_larger_position(self):
        r1x = self.calc.leveraged(price=100.0, leverage=1.0, base_percentage=0.05)
        r3x = self.calc.leveraged(price=100.0, leverage=3.0, base_percentage=0.05)
        assert r3x.quantity > r1x.quantity


class TestPortfolioPositionSizerInit:
    def test_positive_capital_accepted(self):
        sizer = PortfolioPositionSizer(capital=100000.0)
        assert sizer.capital == 100000.0

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            PortfolioPositionSizer(capital=0.0)

    def test_defaults(self):
        sizer = PortfolioPositionSizer(capital=100000.0)
        assert sizer.max_total_exposure  == 1.0
        assert sizer.max_single_position == 0.2

    def test_initial_exposure_is_zero(self):
        assert PortfolioPositionSizer(capital=100000.0).get_total_exposure() == pytest.approx(0.0)

    def test_initial_available_capital(self):
        assert PortfolioPositionSizer(capital=100000.0).get_available_capital() == pytest.approx(100000.0)


class TestPortfolioExposure:
    def setup_method(self):
        self.sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=1.0, max_single_position=0.2
        )

    def test_update_position_adds_exposure(self):
        self.sizer.update_position("AAPL", 20000.0)
        assert self.sizer.get_total_exposure() == pytest.approx(0.2)

    def test_update_position_with_zero_removes_it(self):
        self.sizer.update_position("AAPL", 20000.0)
        self.sizer.update_position("AAPL", 0.0)
        assert self.sizer.get_total_exposure() == pytest.approx(0.0)

    def test_multiple_positions_sum(self):
        self.sizer.update_position("AAPL",  20000.0)
        self.sizer.update_position("GOOGL", 30000.0)
        assert self.sizer.get_total_exposure() == pytest.approx(0.5)

    def test_available_capital_decreases(self):
        self.sizer.update_position("AAPL", 20000.0)
        assert self.sizer.get_available_capital() == pytest.approx(80000.0)

    def test_available_capital_floored_at_zero(self):
        self.sizer.update_position("AAPL",  100000.0)
        self.sizer.update_position("GOOGL",  50000.0)
        assert self.sizer.get_available_capital() == pytest.approx(0.0)

    def test_overwriting_position_replaces_not_adds(self):
        self.sizer.update_position("AAPL", 20000.0)
        self.sizer.update_position("AAPL", 10000.0)
        assert self.sizer.get_total_exposure() == pytest.approx(0.1)


class TestCanAddPosition:
    def setup_method(self):
        self.sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=1.0, max_single_position=0.2
        )

    def test_approved_within_limits(self):
        can_add, _ = self.sizer.can_add_position("AAPL", 10000.0)
        assert can_add is True

    def test_rejected_single_position_limit(self):
        # existing 20000 + adding 5000 = 25000 > 20000 limit
        self.sizer.update_position("AAPL", 20000.0)
        can_add, reason = self.sizer.can_add_position("AAPL", 5000.0)
        assert can_add is False
        assert len(reason) > 0

    def test_rejected_total_exposure_limit(self):
        sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=0.5, max_single_position=0.3
        )
        sizer.update_position("AAPL", 50000.0)
        can_add, _ = sizer.can_add_position("GOOGL", 1000.0)
        assert can_add is False

    def test_different_symbol_approved(self):
        self.sizer.update_position("AAPL", 20000.0)
        can_add, _ = self.sizer.can_add_position("GOOGL", 5000.0)
        assert can_add is True

    def test_reason_populated_on_rejection(self):
        sizer = PortfolioPositionSizer(capital=100000.0, max_total_exposure=0.1)
        sizer.update_position("AAPL", 10000.0)
        can_add, reason = sizer.can_add_position("GOOGL", 5000.0)
        assert can_add is False
        assert isinstance(reason, str) and len(reason) > 0


class TestSizeNewPosition:
    def setup_method(self):
        self.sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=1.0, max_single_position=0.2
        )

    def test_basic(self):
        # available=$100k, max_for_asset=$20k, desired=$10k -> actual=$10k, qty=100
        result = self.sizer.size_new_position("AAPL", price=100.0, desired_percentage=0.1)
        assert result is not None
        assert result.quantity       == pytest.approx(100.0)
        assert result.notional_value == pytest.approx(10000.0)

    def test_capped_by_max_single_position(self):
        # 50% desired but max_single=20% -> notional=$20k -> qty=200
        result = self.sizer.size_new_position("AAPL", price=100.0, desired_percentage=0.5)
        assert result is not None
        assert result.notional_value == pytest.approx(20000.0)

    def test_returns_none_when_no_capacity(self):
        sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=0.2, max_single_position=0.2
        )
        sizer.update_position("AAPL", 20000.0)
        assert sizer.size_new_position("GOOGL", price=100.0, desired_percentage=0.1) is None

    def test_capped_by_available_capital(self):
        # max_total=0.25, existing=$20k -> avail=$5k; desired $20k -> capped $5k -> qty=50
        sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=0.25, max_single_position=0.25
        )
        sizer.update_position("AAPL", 20000.0)
        result = sizer.size_new_position("GOOGL", price=100.0, desired_percentage=0.2)
        assert result is not None
        assert result.notional_value == pytest.approx(5000.0)
        assert result.quantity       == pytest.approx(50.0)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            self.sizer.size_new_position("AAPL", price=0.0, desired_percentage=0.1)

    def test_percent_of_capital_consistent(self):
        result = self.sizer.size_new_position("AAPL", price=100.0, desired_percentage=0.1)
        assert result is not None
        assert result.percent_of_capital == pytest.approx(
            result.notional_value / self.sizer.capital, rel=1e-9
        )


class TestPortfolioWorkflow:
    def test_sequential_allocation_and_exhaustion(self):
        sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=0.6, max_single_position=0.2
        )

        r1 = sizer.size_new_position("AAPL",  price=150.0, desired_percentage=0.2)
        assert r1 is not None
        sizer.update_position("AAPL", r1.notional_value)

        r2 = sizer.size_new_position("GOOGL", price=200.0, desired_percentage=0.2)
        assert r2 is not None
        sizer.update_position("GOOGL", r2.notional_value)

        assert sizer.get_total_exposure()    == pytest.approx(0.4)
        assert sizer.get_available_capital() == pytest.approx(20000.0)

        r3 = sizer.size_new_position("MSFT", price=100.0, desired_percentage=0.2)
        assert r3 is not None
        assert r3.notional_value == pytest.approx(20000.0)

        sizer.update_position("MSFT", r3.notional_value)
        assert sizer.size_new_position("AMZN", price=100.0, desired_percentage=0.1) is None

    def test_closing_position_frees_capacity(self):
        sizer = PortfolioPositionSizer(
            capital=100000.0, max_total_exposure=0.4, max_single_position=0.2
        )
        sizer.update_position("AAPL",  20000.0)
        sizer.update_position("GOOGL", 20000.0)

        assert sizer.get_available_capital() == pytest.approx(0.0)

        sizer.update_position("AAPL", 0.0)
        assert sizer.get_available_capital() == pytest.approx(20000.0)

        result = sizer.size_new_position("MSFT", price=100.0, desired_percentage=0.2)
        assert result is not None
        assert result.notional_value == pytest.approx(20000.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
