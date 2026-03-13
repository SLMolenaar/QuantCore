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
    PerformanceMetrics,
)

MAX_RATIO = 99.99


class TestReturnCalculations:
    def test_calculate_returns(self):
        equity = np.array([100.0, 105.0, 103.0, 110.0])
        returns = calculate_returns(equity)

        expected = np.array([0.05, -0.019047619047619, 0.067961165048544])

        assert len(returns) == 3
        np.testing.assert_array_almost_equal(returns, expected, decimal=10)

    def test_calculate_returns_empty(self):
        equity = np.array([])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_calculate_returns_single_point(self):
        equity = np.array([100.0])
        returns = calculate_returns(equity)
        assert len(returns) == 0

    def test_calculate_total_return(self):
        equity = np.array([100.0, 120.0])
        assert np.isclose(calculate_total_return(equity), 20.0, rtol=1e-9)

        equity = np.array([100.0, 80.0])
        assert np.isclose(calculate_total_return(equity), -20.0, rtol=1e-9)

        equity = np.array([100.0, 100.0])
        assert np.isclose(calculate_total_return(equity), 0.0, rtol=1e-9)

        equity = np.array([100.0, 85.0, 105.0, 150.0])
        assert np.isclose(calculate_total_return(equity), 50.0, rtol=1e-9)

    def test_calculate_annualized_return_one_year(self):
        equity = np.array([100.0] + [0.0] * 251 + [120.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)
        assert np.isclose(annual, 20.0, rtol=1e-9)

    def test_calculate_annualized_return_two_years(self):
        equity = np.array([100.0] + [0.0] * 503 + [144.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)
        assert np.isclose(annual, 20.0, rtol=1e-9)

    def test_calculate_annualized_return_half_year(self):
        equity = np.array([100.0] + [0.0] * 125 + [110.0])
        annual = calculate_annualized_return(equity, periods_per_year=252)
        assert np.isclose(annual, 21.0, rtol=1e-9)


class TestVolatility:
    def test_calculate_volatility_zero(self):
        returns = np.array([0.0, 0.0, 0.0, 0.0])
        assert calculate_volatility(returns) == 0.0

    def test_calculate_volatility(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        vol = calculate_volatility(returns, periods_per_year=252)

        expected = np.std(returns, ddof=0) * np.sqrt(252) * 100
        assert np.isclose(vol, expected, rtol=1e-9)
        assert np.isclose(vol, 25.099801, rtol=1e-5)

    def test_calculate_volatility_constant_returns(self):
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        assert calculate_volatility(returns) == 0.0

    def test_calculate_volatility_empty(self):
        assert calculate_volatility(np.array([])) == 0.0


class TestSharpeRatio:
    def test_sharpe_ratio(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
        assert np.isclose(sharpe, expected, rtol=1e-9)
        assert np.isclose(sharpe, 46.176, rtol=1e-3)

    def test_sharpe_ratio_negative(self):
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
        assert np.isclose(sharpe, expected, rtol=1e-9)
        assert sharpe < 0
        assert np.isclose(sharpe, -46.176, rtol=1e-3)

    def test_sharpe_ratio_with_risk_free(self):
        """risk_free_rate is annual; the code divides by periods_per_year before subtracting."""
        returns = np.array([0.03, 0.02, 0.04, 0.01, 0.02])
        rf_rate = 0.02
        periods = 252

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=rf_rate, periods_per_year=periods)

        rf_per_period  = rf_rate / periods
        excess_returns = returns - rf_per_period
        expected       = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods)

        assert np.isclose(sharpe, expected, rtol=1e-9)

    def test_sharpe_ratio_zero_volatility(self):
        """Constant positive returns → capped at MAX_RATIO_VALUE."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == MAX_RATIO

    def test_sharpe_ratio_empty(self):
        assert calculate_sharpe_ratio(np.array([])) == 0.0


class TestSortinoRatio:
    def test_sortino_ratio_all_positive(self):
        """No downside returns → capped at MAX_RATIO_VALUE."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        assert sortino == MAX_RATIO

    def test_sortino_ratio_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.01, -0.015])
        sortino = calculate_sortino_ratio(returns)

        mean_ret     = np.mean(returns)
        downside_std = np.sqrt(np.mean(returns ** 2))
        expected     = mean_ret / downside_std * np.sqrt(252)

        assert np.isclose(sortino, expected, rtol=1e-9)
        assert sortino < 0

    def test_sortino_ratio_mixed(self):
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        downside_returns = returns[returns < 0]
        downside_std     = np.sqrt(np.mean(downside_returns ** 2))
        expected         = np.mean(returns) / downside_std * np.sqrt(252)

        assert np.isclose(sortino, expected, rtol=1e-9)
        assert np.isclose(sortino, 6.023, rtol=1e-3)

    def test_sortino_ratio_empty(self):
        assert calculate_sortino_ratio(np.array([])) == 0.0


class TestDrawdown:
    """
    Duration = recovery_idx - peak_idx  (exclusive bar count).
    When there is no recovery, recovery_idx == trough_idx.
    When equity never drops, peak_idx == trough_idx == recovery_idx == 0, so duration == 0.
    """

    def test_max_drawdown_no_drawdown(self):
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        dd, duration = calculate_max_drawdown(equity)
        assert dd == 0.0
        assert duration == 0

    def test_max_drawdown_simple(self):
        # Peak=0, trough=3, no recovery → duration = 3 - 0 = 3
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd, duration = calculate_max_drawdown(equity)

        assert np.isclose(dd, -30.0, rtol=1e-9)
        assert duration == 3

    def test_max_drawdown_with_recovery(self):
        # Peak=0, trough=2, recovery at idx 4 → duration = 4 - 0 = 4
        equity = np.array([100.0, 90.0, 80.0, 90.0, 100.0])
        dd, duration = calculate_max_drawdown(equity)

        assert np.isclose(dd, -20.0, rtol=1e-9)
        assert duration == 4

    def test_max_drawdown_multiple_peaks(self):
        # Peak at idx 2 (110), trough at idx 4 (85).
        # equity[6] = 100 < 110 so there is NO recovery.
        # recovery_idx stays at trough_idx = 4 → duration = 4 - 2 = 2
        equity = np.array([100.0, 105.0, 110.0, 90.0, 85.0, 95.0, 100.0])
        dd, duration = calculate_max_drawdown(equity)

        expected_dd = (85.0 - 110.0) / 110.0 * 100.0
        assert np.isclose(dd, expected_dd, rtol=1e-9)
        assert np.isclose(dd, -22.727272, rtol=1e-5)
        assert duration == 2

    def test_max_drawdown_no_recovery(self):
        # Peak=0, trough=3, no recovery → duration = 3 - 0 = 3
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd, duration = calculate_max_drawdown(equity)

        assert np.isclose(dd, -30.0, rtol=1e-9)
        assert duration == 3


class TestCalmarRatio:
    def test_calmar_ratio(self):
        assert calculate_calmar_ratio(20.0, -10.0) == 2.0

    def test_calmar_ratio_zero_drawdown(self):
        """Near-zero drawdown with positive return → capped at MAX_RATIO_VALUE."""
        assert calculate_calmar_ratio(20.0, 0.0) == MAX_RATIO

    def test_calmar_ratio_negative_return(self):
        assert calculate_calmar_ratio(-10.0, -20.0) == -0.5

    def test_calmar_ratio_tiny_drawdown(self):
        """Tiny drawdown (< 0.01) with positive return → capped at MAX_RATIO_VALUE."""
        assert calculate_calmar_ratio(20.0, -0.0001) == MAX_RATIO


class TestTradeAnalysis:
    def test_analyze_trades_all_wins(self):
        trades  = [100.0, 200.0, 150.0, 175.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 4
        assert results['win_rate']     == 100.0
        assert results['profit_factor'] == MAX_RATIO
        assert results['avg_win']      == 156.25
        assert results['avg_loss']     == 0.0
        assert results['largest_win']  == 200.0
        assert results['largest_loss'] == 0.0

    def test_analyze_trades_all_losses(self):
        trades  = [-100.0, -200.0, -150.0, -175.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 4
        assert results['win_rate']     == 0.0
        assert results['profit_factor'] == 0.0
        assert results['avg_win']      == 0.0
        assert results['avg_loss']     == -156.25
        assert results['largest_win']  == 0.0
        assert results['largest_loss'] == -200.0

    def test_analyze_trades_mixed(self):
        trades  = [100.0, -50.0, 200.0, -75.0, 150.0]
        results = analyze_trades(trades)

        assert results['total_trades'] == 5
        assert results['win_rate']     == 60.0
        assert np.isclose(results['profit_factor'], 3.6, rtol=1e-9)
        assert results['avg_win']      == 150.0
        assert results['avg_loss']     == -62.5
        assert results['largest_win']  == 200.0
        assert results['largest_loss'] == -75.0

    def test_analyze_trades_empty(self):
        results = analyze_trades([])
        assert results['total_trades'] == 0
        assert results['win_rate']     == 0.0


class TestRollingMetrics:
    def test_rolling_sharpe(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.01, 0.02, -0.01])
        rolling = rolling_sharpe(returns, window=5, periods_per_year=252)

        assert len(rolling) == 4

        window1  = returns[0:5]
        expected = np.mean(window1) / np.std(window1, ddof=1) * np.sqrt(252)
        assert np.isclose(rolling[0], expected, rtol=1e-9)

    def test_rolling_sharpe_insufficient_data(self):
        rolling = rolling_sharpe(np.array([0.01, 0.02]), window=5)
        assert len(rolling) == 0

    def test_rolling_volatility(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.01])
        rolling = rolling_volatility(returns, window=3, periods_per_year=252)

        assert len(rolling) == 4

        window1  = returns[0:3]
        expected = np.std(window1, ddof=0) * np.sqrt(252) * 100
        assert np.isclose(rolling[0], expected, rtol=1e-9)


class TestPerformanceMetrics:
    def test_calculate_all_metrics_deterministic(self):
        equity = np.array([100000.0, 105000.0, 110000.0, 108000.0, 115000.0])
        trades = [500.0, -200.0, 800.0, -100.0, 600.0]

        metrics = calculate_all_metrics(equity, trade_pnls=trades, risk_free_rate=0.0)

        assert isinstance(metrics, PerformanceMetrics)
        assert np.isclose(metrics.total_return, 15.0, rtol=1e-9)
        assert metrics.total_trades  == 5
        assert metrics.win_rate      == 60.0
        assert metrics.avg_win       == (500 + 800 + 600) / 3
        assert metrics.avg_loss      == (-200 - 100) / 2
        assert metrics.largest_win   == 800.0
        assert metrics.largest_loss  == -200.0


class TestEdgeCases:
    def test_division_by_zero_protection(self):
        returns = np.array([0.0, 0.0, 0.0])
        assert calculate_sharpe_ratio(returns) == 0.0

        # Near-zero drawdown with positive return → MAX_RATIO_VALUE, not inf or nan
        calmar = calculate_calmar_ratio(10.0, 0.0)
        assert calmar == MAX_RATIO
        assert np.isfinite(calmar)

    def test_negative_equity_handling(self):
        equity  = np.array([100.0, -50.0, 75.0])
        returns = calculate_returns(equity)

        assert len(returns) == 2
        assert returns[0]   == -1.5
        assert np.isfinite(returns).all()

    def test_extreme_values(self):
        equity  = np.array([1e10, 1.5e10, 2e10])
        returns = calculate_returns(equity)
        assert len(returns) == 2
        assert np.isclose(returns[0], 0.5, rtol=1e-9)

        equity  = np.array([1e-10, 2e-10, 1.5e-10])
        returns = calculate_returns(equity)
        assert len(returns) == 2
        assert np.isfinite(returns).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])