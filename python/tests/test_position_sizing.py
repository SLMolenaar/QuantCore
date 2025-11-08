"""
Tests for position sizing module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from position_sizing import PositionSizer, PortfolioPositionSizer, PositionSizeResult


class TestPositionSizer:

    def test_initialization(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.25)
        assert sizer.capital == 100000.0
        assert sizer.max_position_pct == 0.25

    def test_update_capital(self):
        sizer = PositionSizer(capital=100000.0)
        sizer.update_capital(150000.0)
        assert sizer.capital == 150000.0

    def test_fixed_percentage(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.fixed_percentage(price=100.0, percentage=0.1)

        assert isinstance(result, PositionSizeResult)
        assert result.quantity == 100.0
        assert result.notional_value == 10000.0
        assert result.percent_of_capital == 0.1
        assert "10.0%" in result.reasoning

    def test_fixed_percentage_respects_max(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.15)
        result = sizer.fixed_percentage(price=100.0, percentage=0.25)

        assert result.notional_value == 15000.0
        assert result.percent_of_capital == 0.15

    def test_fixed_percentage_minimum_quantity(self):
        sizer = PositionSizer(capital=100.0)
        result = sizer.fixed_percentage(price=1000.0, percentage=0.01, min_quantity=1)

        assert result.quantity >= 1.0

    def test_risk_based_long_position(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=1.0)
        result = sizer.risk_based(
            price=100.0,
            stop_loss_price=98.0,
            risk_per_trade=0.01
        )

        expected_quantity = 1000.0 / 2.0
        assert result.quantity == expected_quantity
        assert abs(result.notional_value - 50000.0) < 0.01

    def test_risk_based_short_position(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.risk_based(
            price=100.0,
            stop_loss_price=102.0,
            risk_per_trade=0.01
        )

        expected_quantity = 1000.0 / 2.0
        assert result.quantity == expected_quantity

    def test_risk_based_respects_max_position(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.2)
        result = sizer.risk_based(
            price=100.0,
            stop_loss_price=99.0,
            risk_per_trade=0.05
        )

        assert result.notional_value <= 20000.0

    def test_risk_based_invalid_stop(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="cannot equal"):
            sizer.risk_based(price=100.0, stop_loss_price=100.0)

    def test_kelly_criterion_positive_edge(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.kelly_criterion(
            price=100.0,
            win_rate=0.6,
            avg_win=150.0,
            avg_loss=100.0,
            kelly_fraction=0.5
        )

        assert result.quantity > 0
        assert result.percent_of_capital > 0
        assert "Kelly" in result.reasoning

    def test_kelly_criterion_no_edge(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.kelly_criterion(
            price=100.0,
            win_rate=0.5,
            avg_win=100.0,
            avg_loss=100.0,
            kelly_fraction=1.0
        )

        assert result.quantity >= 1.0
        assert result.percent_of_capital >= 0

    def test_kelly_criterion_invalid_win_rate(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="Win rate"):
            sizer.kelly_criterion(
                price=100.0,
                win_rate=1.5,
                avg_win=100.0,
                avg_loss=100.0
            )

    def test_kelly_criterion_zero_loss(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="Average loss"):
            sizer.kelly_criterion(
                price=100.0,
                win_rate=0.6,
                avg_win=100.0,
                avg_loss=0.0
            )

    def test_equal_weight_two_positions(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.equal_weight(price=100.0, num_positions=2)

        assert result.notional_value == 50000.0
        assert result.percent_of_capital == 0.5
        assert "1/2" in result.reasoning

    def test_equal_weight_five_positions(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.equal_weight(price=100.0, num_positions=5)

        assert result.notional_value == 20000.0
        assert result.percent_of_capital == 0.2

    def test_equal_weight_respects_max(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.15)
        result = sizer.equal_weight(price=100.0, num_positions=2)

        assert result.percent_of_capital <= 0.15

    def test_equal_weight_invalid_num_positions(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="must be positive"):
            sizer.equal_weight(price=100.0, num_positions=0)

    def test_volatility_adjusted_high_vol(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.volatility_adjusted(
            price=100.0,
            volatility=0.30,
            target_volatility=0.15,
            base_allocation=0.1
        )

        assert result.percent_of_capital < 0.1
        assert "Vol-adjusted" in result.reasoning

    def test_volatility_adjusted_low_vol(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.volatility_adjusted(
            price=100.0,
            volatility=0.10,
            target_volatility=0.15,
            base_allocation=0.1
        )

        assert result.percent_of_capital > 0.1

    def test_volatility_adjusted_respects_max(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.15)
        result = sizer.volatility_adjusted(
            price=100.0,
            volatility=0.05,
            target_volatility=0.15,
            base_allocation=0.2
        )

        assert result.percent_of_capital <= 0.15

    def test_volatility_adjusted_invalid_volatility(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="Volatility must be positive"):
            sizer.volatility_adjusted(
                price=100.0,
                volatility=0.0,
                target_volatility=0.15
            )

    def test_leveraged_2x(self):
        sizer = PositionSizer(capital=100000.0)
        result = sizer.leveraged(
            price=100.0,
            leverage=2.0,
            base_percentage=0.1
        )

        assert result.notional_value == 20000.0
        assert result.percent_of_capital == 0.2
        assert "2.0x leverage" in result.reasoning

    def test_leveraged_respects_max(self):
        sizer = PositionSizer(capital=100000.0, max_position_pct=0.25)
        result = sizer.leveraged(
            price=100.0,
            leverage=5.0,
            base_percentage=0.1
        )

        assert result.notional_value <= 125000.0

    def test_leveraged_invalid_leverage(self):
        sizer = PositionSizer(capital=100000.0)

        with pytest.raises(ValueError, match="Leverage must be positive"):
            sizer.leveraged(price=100.0, leverage=0.0)


class TestPortfolioPositionSizer:

    def test_initialization(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=1.5,
            max_single_position=0.2
        )

        assert portfolio_sizer.capital == 100000.0
        assert portfolio_sizer.max_total_exposure == 1.5
        assert portfolio_sizer.max_single_position == 0.2
        assert len(portfolio_sizer.current_positions) == 0

    def test_update_position(self):
        portfolio_sizer = PortfolioPositionSizer(capital=100000.0)

        portfolio_sizer.update_position("AAPL", 10000.0)
        assert portfolio_sizer.current_positions["AAPL"] == 10000.0

        portfolio_sizer.update_position("AAPL", 15000.0)
        assert portfolio_sizer.current_positions["AAPL"] == 15000.0

        portfolio_sizer.update_position("AAPL", 0)
        assert "AAPL" not in portfolio_sizer.current_positions

    def test_get_total_exposure(self):
        portfolio_sizer = PortfolioPositionSizer(capital=100000.0)

        portfolio_sizer.update_position("AAPL", 30000.0)
        portfolio_sizer.update_position("GOOGL", 40000.0)

        assert portfolio_sizer.get_total_exposure() == 0.7

    def test_get_total_exposure_with_shorts(self):
        portfolio_sizer = PortfolioPositionSizer(capital=100000.0)

        portfolio_sizer.update_position("AAPL", 30000.0)
        portfolio_sizer.update_position("TSLA", -20000.0)

        assert portfolio_sizer.get_total_exposure() == 0.5

    def test_get_available_capital(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=1.0
        )

        portfolio_sizer.update_position("AAPL", 60000.0)

        assert portfolio_sizer.get_available_capital() == 40000.0

    def test_get_available_capital_with_leverage(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=2.0
        )

        portfolio_sizer.update_position("AAPL", 150000.0)

        assert portfolio_sizer.get_available_capital() == 50000.0

    def test_can_add_position_allowed(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_single_position=0.2
        )

        can_add, reason = portfolio_sizer.can_add_position("AAPL", 15000.0)

        assert can_add is True
        assert "allowed" in reason.lower()

    def test_can_add_position_exceeds_single_limit(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_single_position=0.2
        )

        can_add, reason = portfolio_sizer.can_add_position("AAPL", 25000.0)

        assert can_add is False
        assert "single position limit" in reason.lower()

    def test_can_add_position_exceeds_total_exposure(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=1.0,
            max_single_position=0.3
        )

        portfolio_sizer.update_position("AAPL", 50000.0)
        portfolio_sizer.update_position("GOOGL", 40000.0)

        can_add, reason = portfolio_sizer.can_add_position("MSFT", 20000.0)

        assert can_add is False
        assert "total exposure limit" in reason.lower()

    def test_size_new_position(self):
        portfolio_sizer = PortfolioPositionSizer(capital=100000.0)

        result = portfolio_sizer.size_new_position(
            symbol="AAPL",
            price=100.0,
            desired_percentage=0.1
        )

        assert result is not None
        assert result.notional_value == 10000.0
        assert result.quantity == 100.0

    def test_size_new_position_with_existing_positions(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=1.0
        )

        portfolio_sizer.update_position("AAPL", 60000.0)

        result = portfolio_sizer.size_new_position(
            symbol="GOOGL",
            price=200.0,
            desired_percentage=0.15
        )

        assert result is not None
        assert result.notional_value <= 40000.0

    def test_size_new_position_no_capital_available(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_total_exposure=1.0
        )

        portfolio_sizer.update_position("AAPL", 100000.0)

        result = portfolio_sizer.size_new_position(
            symbol="GOOGL",
            price=200.0,
            desired_percentage=0.1
        )

        assert result is None

    def test_size_new_position_respects_single_limit(self):
        portfolio_sizer = PortfolioPositionSizer(
            capital=100000.0,
            max_single_position=0.15
        )

        result = portfolio_sizer.size_new_position(
            symbol="AAPL",
            price=100.0,
            desired_percentage=0.25
        )

        assert result is not None
        assert result.percent_of_capital <= 0.15


class TestPositionSizeResult:

    def test_position_size_result_creation(self):
        result = PositionSizeResult(
            quantity=100.0,
            notional_value=10000.0,
            percent_of_capital=0.1,
            reasoning="Test sizing"
        )

        assert result.quantity == 100.0
        assert result.notional_value == 10000.0
        assert result.percent_of_capital == 0.1
        assert result.reasoning == "Test sizing"


class TestIntegration:

    def test_different_methods_same_inputs(self):
        capital = 100000.0
        price = 150.0

        sizer = PositionSizer(capital=capital)

        fixed = sizer.fixed_percentage(price=price, percentage=0.1)
        equal = sizer.equal_weight(price=price, num_positions=10)

        assert abs(fixed.notional_value - equal.notional_value) < 1.0

    def test_risk_based_scales_with_stop_distance(self):
        capital = 100000.0
        price = 100.0
        risk = 0.01

        sizer = PositionSizer(capital=capital)

        tight_stop = sizer.risk_based(price=price, stop_loss_price=99.0, risk_per_trade=risk)
        wide_stop = sizer.risk_based(price=price, stop_loss_price=95.0, risk_per_trade=risk)

        assert tight_stop.quantity > wide_stop.quantity

    def test_volatility_adjusted_inverse_relationship(self):
        capital = 100000.0
        price = 100.0

        sizer = PositionSizer(capital=capital)

        low_vol = sizer.volatility_adjusted(price=price, volatility=0.1, target_volatility=0.2)
        high_vol = sizer.volatility_adjusted(price=price, volatility=0.4, target_volatility=0.2)

        assert low_vol.quantity > high_vol.quantity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])