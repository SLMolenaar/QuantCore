"""
Tests for analytics module
All tests use exact expected values calculated manually.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantcore.analytics import (
    calculate_returns,
    calculate_total_return,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    analyze_trades,
    calculate_all_metrics,
    rolling_sharpe,
    rolling_volatility,
    PerformanceMetrics
)


class TestReturnCalculations:
    def test_calculate_returns(self):
        equity = np.array([100.0, 105.0, 103.0, 110.0])
        returns = calculate_returns(equity)

        # Manual calculation:
        # (105-100)/100 = 0.05
        # (103-105)/105 = -0.019047619...
        # (110-103)/103 = 0.067961165...
        expected = np.array([0.05, -0.019047619047619, 0.067961165048544])

        assert len(returns) == 3
        np.testing.assert_array_almost_equal(returns, expected, decimal=10)

    def test_calculate_returns_empty(self):
        """Empty array should return empty array"""
        equity = np.array([])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_calculate_returns_single_point(self):
        """Single point has no returns"""
        equity = np.array([100.0])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_calculate_total_return(self):
        """Test total return"""
        # 20% gain
        equity = np.array([100.0, 120.0])
        total = calculate_total_return(equity)
        assert np.isclose(total, 20.0, rtol=1e-9)  # Use np.isclose instead of ==

        # 20% loss
        equity = np.array([100.0, 80.0])
        total = calculate_total_return(equity)
        assert np.isclose(total, -20.0, rtol=1e-9)

        # Flat
        equity = np.array([100.0, 100.0])
        total = calculate_total_return(equity)
        assert np.isclose(total, 0.0, rtol=1e-9)

        # indirect 50% gain
        equity = np.array([100.0, 85.0, 105.0, 150.0])
        total = calculate_total_return(equity)
        assert np.isclose(total, 50.0, rtol=1e-9)

    def test_calculate_annualized_return_one_year(self):
        """Test annualized return for 1 year"""
        # 252 trading days, 20% return
        equity = np.array([100.0] + [0.0]*251 + [120.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)

        # (120/100)^(1/1) - 1 = 0.20 = 20%
        expected = 20.0
        assert np.isclose(annual, expected, rtol=1e-9)

    def test_calculate_annualized_return_two_years(self):
        """Test annualized return for 2 years"""
        # 504 trading days (2 years), compound to 144 (20% per year)
        # (1.2)^2 = 1.44
        equity = np.array([100.0] + [0.0]*503 + [144.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)

        # (144/100)^(1/2) - 1 = 0.2 = 20%
        expected = 20.0
        assert np.isclose(annual, expected, rtol=1e-9)

    def test_calculate_annualized_return_half_year(self):
        """Test annualized return for half a year"""
        # 126 days, 10% return
        equity = np.array([100.0] + [0.0]*125 + [110.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)

        # (1.10)^(252/126) - 1 = (1.10)^2 - 1 = 0.21 = 21%
        expected = 21.0
        assert np.isclose(annual, expected, rtol=1e-9)


class TestVolatility:
    """Test volatility calculations"""

    def test_calculate_volatility_zero(self):
        """Zero volatility when no variation"""
        returns = np.array([0.0, 0.0, 0.0, 0.0])
        vol = calculate_volatility(returns)
        assert vol == 0.0

    def test_calculate_volatility(self):
        # Returns: [0.01, -0.01, 0.02, -0.02]
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        vol = calculate_volatility(returns, periods_per_year=252)

        # Manual: std = 0.0158113883..., annualized = std * sqrt(252) * 100
        std = np.std(returns, ddof=0)  # Population std (not sample)
        expected = std * np.sqrt(252) * 100

        assert np.isclose(vol, expected, rtol=1e-9)
        assert np.isclose(vol, 25.099801, rtol=1e-5)

    def test_calculate_volatility_constant_returns(self):
        """Constant returns = zero volatility"""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        vol = calculate_volatility(returns)
        assert vol == 0.0

    def test_calculate_volatility_empty(self):
        """Empty returns = zero volatility"""
        returns = np.array([])
        vol = calculate_volatility(returns)
        assert vol == 0.0


class TestSharpeRatio:
    """Test Sharpe ratio with calculations"""

    def test_sharpe_ratio(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Manual calculation:
        mean_ret = np.mean(returns)  # 0.013
        std_ret = np.std(returns, ddof=1)  # Sample std
        expected = mean_ret / std_ret * np.sqrt(252)

        assert np.isclose(sharpe, expected, rtol=1e-9)
        # Mean = 0.013, Std = 0.004472136, Sharpe = 0.013/0.004472136 * sqrt(252) = 46.176
        assert np.isclose(sharpe, 46.176, rtol=1e-3)

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns"""
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        mean_ret = np.mean(returns)  # -0.013
        std_ret = np.std(returns, ddof=1)
        expected = mean_ret / std_ret * np.sqrt(252)

        assert np.isclose(sharpe, expected, rtol=1e-9)
        assert sharpe < 0
        assert np.isclose(sharpe, -46.176, rtol=1e-3)

    def test_sharpe_ratio_with_risk_free(self):
        """Test Sharpe with risk-free rate"""
        returns = np.array([0.03, 0.02, 0.04, 0.01, 0.02])
        rf_rate = 0.02  # 2% annual

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=rf_rate, periods_per_year=252)

        excess_returns = returns - rf_rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        expected = mean_excess / std_excess * np.sqrt(252)

        assert np.isclose(sharpe, expected, rtol=1e-9)

    def test_sharpe_ratio_zero_volatility(self):
        """Zero volatility = zero Sharpe"""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_empty(self):
        """Empty returns = zero Sharpe"""
        returns = np.array([])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0


class TestSortinoRatio:
    def test_sortino_ratio_all_positive(self):
        """All positive returns = infinite Sortino"""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        assert sortino == float('inf')

    def test_sortino_ratio_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.01, -0.015])
        sortino = calculate_sortino_ratio(returns)

        # All returns are downside
        mean_ret = np.mean(returns)
        downside_std = np.sqrt(np.mean(returns ** 2))
        expected = mean_ret / downside_std * np.sqrt(252)

        assert np.isclose(sortino, expected, rtol=1e-9)
        assert sortino < 0

    def test_sortino_ratio_mixed(self):
        """Test Sortino with mixed returns"""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Manual calculation:
        # Mean = 0.006
        # Downside returns = [-0.01, -0.02]
        # Downside deviation = sqrt(mean([-0.01^2, -0.02^2])) = sqrt(0.00025) = 0.0158113883
        mean_ret = np.mean(returns)  # 0.006
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))  # 0.0158113883
        expected = mean_ret / downside_std * np.sqrt(252)

        assert np.isclose(sortino, expected, rtol=1e-9)
        assert np.isclose(sortino, 6.023, rtol=1e-3)

    def test_sortino_ratio_empty(self):
        """Empty returns = zero Sortino"""
        returns = np.array([])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 0.0


class TestDrawdown:
    """Test drawdown calculations with values"""

    def test_max_drawdown_no_drawdown(self):
        """No drawdown when monotonically increasing"""
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        dd, duration = calculate_max_drawdown(equity)
        assert dd == 0.0
        assert duration == 1

    def test_max_drawdown_simple(self):
        """Test simple drawdown calculation"""
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd, duration = calculate_max_drawdown(equity)

        # Peak at 100, trough at 70 = (70-100)/100 = -30%
        expected_dd = -30.0
        assert np.isclose(dd, expected_dd, rtol=1e-9)
        assert duration == 4

    def test_max_drawdown_with_recovery(self):
        """Test drawdown with recovery"""
        equity = np.array([100.0, 90.0, 80.0, 90.0, 100.0])
        dd, duration = calculate_max_drawdown(equity)

        # Peak at 100, trough at 80 = (80-100)/100 = -20%
        expected_dd = -20.0
        assert np.isclose(dd, expected_dd, rtol=1e-9)
        assert duration == 4

    def test_max_drawdown_multiple_peaks(self):
        """Test with multiple peaks - should find worst"""
        equity = np.array([100.0, 105.0, 110.0, 90.0, 85.0, 95.0, 100.0])
        dd, duration = calculate_max_drawdown(equity)

        # Peak at 110 (idx 2), trough at 85 (idx 4)
        # DD = (85-110)/110 = -22.727272%
        expected_dd = (85.0 - 110.0) / 110.0 * 100.0
        assert np.isclose(dd, expected_dd, rtol=1e-9)
        assert np.isclose(dd, -22.727272, rtol=1e-5)

        # Duration from peak (idx 2) to recovery (idx 6) = 5 periods
        assert duration == 5

    def test_max_drawdown_no_recovery(self):
        """Test drawdown with no recovery"""
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd, duration = calculate_max_drawdown(equity)

        expected_dd = -30.0
        assert np.isclose(dd, expected_dd, rtol=1e-9)
        # Duration from peak (idx 0) to end (idx 3) = 4
        assert duration == 4


class TestCalmarRatio:
    def test_calmar_ratio(self):
        calmar = calculate_calmar_ratio(20.0, -10.0)
        assert calmar == 2.0

    def test_calmar_ratio_zero_drawdown(self):
        """Zero drawdown = zero Calmar"""
        calmar = calculate_calmar_ratio(20.0, 0.0)
        assert calmar == 0.0

    def test_calmar_ratio_negative_return(self):
        """Negative return with drawdown"""
        calmar = calculate_calmar_ratio(-10.0, -20.0)
        # annualized_return / abs(max_drawdown) = -10 / 20 = -0.5
        assert calmar == -0.5

    def test_calmar_ratio_tiny_drawdown(self):
        """Tiny drawdown should return zero (to avoid inf)"""
        calmar = calculate_calmar_ratio(20.0, -0.0001)
        assert calmar == 0.0


class TestTradeAnalysis:
    def test_analyze_trades_all_wins(self):
        """Test with all winning trades"""
        trades = [100.0, 200.0, 150.0, 175.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 4
        assert results['win_rate'] == 100.0
        assert results['profit_factor'] == float('inf')
        assert results['avg_win'] == 156.25  # (100+200+150+175)/4
        assert results['avg_loss'] == 0.0
        assert results['largest_win'] == 200.0
        assert results['largest_loss'] == 0.0

    def test_analyze_trades_all_losses(self):
        """Test with all losing trades"""
        trades = [-100.0, -200.0, -150.0, -175.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 4
        assert results['win_rate'] == 0.0
        assert results['profit_factor'] == 0.0
        assert results['avg_win'] == 0.0
        assert results['avg_loss'] == -156.25  # (-100-200-150-175)/4
        assert results['largest_win'] == 0.0
        assert results['largest_loss'] == -200.0

    def test_analyze_trades_mixed(self):
        """Test with mixed trades"""
        trades = [100.0, -50.0, 200.0, -75.0, 150.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 5
        assert results['win_rate'] == 60.0  # 3/5 wins

        # Wins: [100, 200, 150], sum = 450
        # Losses: [-50, -75], sum = -125
        # Profit factor = 450 / 125 = 3.6
        assert np.isclose(results['profit_factor'], 3.6, rtol=1e-9)

        assert results['avg_win'] == 150.0  # (100+200+150)/3
        assert results['avg_loss'] == -62.5  # (-50-75)/2
        assert results['largest_win'] == 200.0
        assert results['largest_loss'] == -75.0

    def test_analyze_trades_empty(self):
        """Empty trades list"""
        trades = []
        results = analyze_trades(trades)

        assert results['total_trades'] == 0
        assert results['win_rate'] == 0.0


class TestRollingMetrics:
    """Test rolling metrics with deterministic data"""

    def test_rolling_sharpe(self):
        """Test rolling Sharpe with known data"""
        # Create deterministic returns
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.01, 0.02, -0.01])
        rolling = rolling_sharpe(returns, window=5, periods_per_year=252)

        assert len(rolling) == 4  # 8 - 5 + 1 = 4 windows

        # Calculate first window manually: [0.01, 0.02, -0.01, 0.03, 0.00]
        window1 = returns[0:5]
        mean1 = np.mean(window1)
        std1 = np.std(window1, ddof=1)
        expected1 = mean1 / std1 * np.sqrt(252) if std1 > 0 else 0.0

        assert np.isclose(rolling[0], expected1, rtol=1e-9)

    def test_rolling_sharpe_insufficient_data(self):
        """Insufficient data returns empty array"""
        returns = np.array([0.01, 0.02])
        rolling = rolling_sharpe(returns, window=5)
        assert len(rolling) == 0

    def test_rolling_volatility(self):
        """Test rolling volatility with known data"""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.01])
        rolling = rolling_volatility(returns, window=3, periods_per_year=252)

        assert len(rolling) == 4  # 6 - 3 + 1 = 4

        # First window: [0.01, 0.02, -0.01]
        window1 = returns[0:3]
        expected1 = np.std(window1, ddof=0) * np.sqrt(252) * 100

        assert np.isclose(rolling[0], expected1, rtol=1e-9)


class TestPerformanceMetrics:
    """Test complete performance metrics calculation"""

    def test_calculate_all_metrics_deterministic(self):
        """Test with fully deterministic data"""
        equity = np.array([100000.0, 105000.0, 110000.0, 108000.0, 115000.0])
        trades = [500.0, -200.0, 800.0, -100.0, 600.0]

        metrics = calculate_all_metrics(equity, trade_pnls=trades, risk_free_rate=0.0)

        assert isinstance(metrics, PerformanceMetrics)

        # Total return: (115000-100000)/100000 = 15%
        assert np.isclose(metrics.total_return, 15.0, rtol=1e-9)

        # Trade metrics
        assert metrics.total_trades == 5
        assert metrics.win_rate == 60.0  # 3 wins out of 5
        assert metrics.avg_win == (500 + 800 + 600) / 3  # 633.333...
        assert metrics.avg_loss == (-200 - 100) / 2  # -150
        assert metrics.largest_win == 800.0
        assert metrics.largest_loss == -200.0


class TestEdgeCases:
    """Test edge cases that could cause financial errors"""

    def test_division_by_zero_protection(self):
        """Ensure no division by zero in any calculation"""
        # Zero volatility
        returns = np.array([0.0, 0.0, 0.0])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0  # Not inf or nan

        # Zero drawdown
        calmar = calculate_calmar_ratio(10.0, 0.0)
        assert calmar == 0.0  # Not inf

    def test_negative_equity_handling(self):
        """Test handling of negative equity (margin call scenarios)"""
        equity = np.array([100.0, -50.0, 75.0])
        returns = calculate_returns(equity)

        # Should handle negative values
        assert len(returns) == 2
        assert returns[0] == -1.5  # ((-50) - 100) / 100 = -1.5 = -150%
        assert np.isfinite(returns).all()

    def test_extreme_values(self):
        """Test with extreme but valid values"""
        # Large numbers
        equity = np.array([1e10, 1.5e10, 2e10])
        returns = calculate_returns(equity)
        assert len(returns) == 2
        assert np.isclose(returns[0], 0.5, rtol=1e-9)

        # Small numbers
        equity = np.array([1e-10, 2e-10, 1.5e-10])
        returns = calculate_returns(equity)
        assert len(returns) == 2
        assert np.isfinite(returns).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])