"""
Tests for risk management module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc


class TestRiskLimits:
    def test_default_limits(self):
        limits = qc.RiskLimits()
        assert limits.max_position_pct == 0.20
        assert limits.max_leverage     == 2.0
        assert limits.max_loss_pct     == 0.50
        assert limits.enabled          == True

    def test_custom_limits(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.15
        limits.max_leverage     = 1.5
        limits.max_loss_pct     = 0.30

        assert limits.max_position_pct == 0.15
        assert limits.max_leverage     == 1.5
        assert limits.max_loss_pct     == 0.30

    def test_validate_position_pct(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 1.5
        limits.validate()  # should not raise — values > 1.0 allow per-asset leverage

        limits.max_position_pct = 0.0
        with pytest.raises(Exception):
            limits.validate()  # zero is invalid

        limits.max_position_pct = -0.1
        with pytest.raises(Exception):
            limits.validate()  # negative is invalid

    def test_validate_leverage(self):
        limits = qc.RiskLimits()
        limits.max_leverage = 15.0
        limits.validate()  # should not raise — high leverage is valid if explicitly set

        limits.max_leverage = 0.0
        with pytest.raises(Exception):
            limits.validate()  # zero is invalid

        limits.max_leverage = -1.0
        with pytest.raises(Exception):
            limits.validate()  # negative is invalid


class TestRiskManager:
    def test_creation(self):
        assert qc.RiskManager() is not None

    def test_creation_with_limits(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.15

        risk_mgr = qc.RiskManager(limits)
        assert risk_mgr.get_limits().max_position_pct == 0.15

    def test_set_capital(self):
        risk_mgr = qc.RiskManager()
        risk_mgr.set_capital(100000.0, 95000.0)

    def test_set_get_position(self):
        risk_mgr = qc.RiskManager()
        risk_mgr.set_position("AAPL", 100.0)

        assert risk_mgr.get_position("AAPL")  == 100.0
        assert risk_mgr.get_position("GOOGL") == 0.0

    def test_update_position(self):
        risk_mgr = qc.RiskManager()
        risk_mgr.set_position("AAPL", 0.0)

        risk_mgr.update_position("AAPL", qc.Side.BUY, 100.0)
        assert risk_mgr.get_position("AAPL") == 100.0

        risk_mgr.update_position("AAPL", qc.Side.SELL, 50.0)
        assert risk_mgr.get_position("AAPL") == 50.0

    def test_reset(self):
        risk_mgr = qc.RiskManager()
        risk_mgr.set_capital(100000.0, 100000.0)
        risk_mgr.set_position("AAPL", 100.0)

        risk_mgr.reset()

        assert risk_mgr.get_position("AAPL") == 0.0


class TestRiskChecks:
    def test_check_order_approved(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.20

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 100000.0)

        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 100, 100.0)

        assert check.is_approved()
        assert check.result == qc.RiskCheckResult.APPROVED

    def test_check_order_position_limit(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.10

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 100000.0)

        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 200, 100.0)

        assert not check.is_approved()
        assert check.result == qc.RiskCheckResult.REJECTED_POSITION_LIMIT
        assert "Position would be" in check.reason

    def test_check_order_leverage_limit(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.50
        limits.max_leverage     = 1.5

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 100000.0)

        # Must supply a price so the notional is tracked for the leverage check.
        risk_mgr.set_position("AAPL", 500, 100.0)  # notional = 500 * 100 = $50k

        # Adding 300 @ $150 = $45k → total notional $95k → leverage 0.95x < 1.5x, still approved.
        # Use a price that pushes total notional over the limit instead.
        # 500 @ $200 = $100k existing, then 300 @ $150 = $45k new → total $145k → 1.45x < 1.5x.
        # Let's set a tighter notional: 500 @ $250 = $125k existing → leverage 1.25x already.
        # Then 300 @ $150 = $45k new → total $170k → 1.7x > 1.5x → rejected.
        risk_mgr.reset()
        risk_mgr.set_capital(100000.0, 100000.0)
        risk_mgr.set_position("AAPL", 500, 250.0)  # notional = $125k → 1.25x leverage

        check = risk_mgr.check_order("GOOGL", qc.Side.BUY, 300, 150.0)

        assert not check.is_approved()
        assert check.result == qc.RiskCheckResult.REJECTED_LEVERAGE_LIMIT

    def test_check_order_loss_limit(self):
        limits = qc.RiskLimits()
        limits.max_loss_pct = 0.20

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 75000.0)  # down 25%

        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 100, 100.0)

        assert not check.is_approved()
        assert check.result == qc.RiskCheckResult.REJECTED_LOSS_LIMIT
        assert "down" in check.reason.lower()

    def test_check_order_disabled(self):
        limits = qc.RiskLimits()
        limits.max_position_pct = 0.01
        limits.enabled          = False

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 100000.0)

        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 1000, 100.0)

        assert check.is_approved()

    def test_check_order_max_order_value(self):
        limits = qc.RiskLimits()
        limits.max_order_value = 5000.0

        risk_mgr = qc.RiskManager(limits)
        risk_mgr.set_capital(100000.0, 100000.0)

        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 100, 100.0)

        assert not check.is_approved()
        assert check.result == qc.RiskCheckResult.REJECTED_ORDER_SIZE


class TestRiskCheckResponse:
    def test_is_approved(self):
        response        = qc.RiskCheckResponse()
        response.result = qc.RiskCheckResult.APPROVED
        assert response.is_approved()

    def test_is_rejected(self):
        response        = qc.RiskCheckResponse()
        response.result = qc.RiskCheckResult.REJECTED_POSITION_LIMIT
        response.reason = "Position too large"

        assert not response.is_approved()
        assert "too large" in response.reason


class TestBacktestEngineIntegration:
    def test_set_risk_limits(self):
        engine = qc.BacktestEngine(100000.0)

        limits = qc.RiskLimits()
        limits.max_position_pct = 0.15
        engine.set_risk_limits(limits)

        assert engine.get_risk_limits().max_position_pct == 0.15

    def test_get_risk_manager(self):
        engine   = qc.BacktestEngine(100000.0)
        risk_mgr = engine.get_risk_manager()

        assert risk_mgr is not None
        assert isinstance(risk_mgr, qc.RiskManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])