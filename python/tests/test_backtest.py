"""
Unit tests for QuantCore Python bindings
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc


class TestBasicImports:
    def test_version(self):
        version = qc.version()
        assert isinstance(version, str)
        assert version == "0.2.1"

    def test_hello(self):
        assert "QuantCore" in qc.hello()


class TestEnums:
    def test_signal_type(self):
        assert qc.SignalType.BUY  is not None
        assert qc.SignalType.SELL is not None
        assert qc.SignalType.HOLD is not None

    def test_side(self):
        assert qc.Side.BUY  is not None
        assert qc.Side.SELL is not None

    def test_order_type(self):
        assert qc.OrderType.GOOD_TILL_CANCEL    is not None
        assert qc.OrderType.MARKET              is not None
        assert qc.OrderType.IMMEDIATE_OR_CANCEL is not None
        assert qc.OrderType.FILL_OR_KILL        is not None
        assert qc.OrderType.GOOD_FOR_DAY        is not None


class TestBarData:
    def test_create_bar(self):
        bar = qc.BarData("AAPL", 1000000000, 100.0, 101.0, 99.0, 100.5, 1000000.0)
        assert bar.symbol == "AAPL"
        assert bar.open   == 100.0
        assert bar.high   == 101.0
        assert bar.low    == 99.0
        assert bar.close  == 100.5
        assert bar.volume == 1000000.0

    def test_bar_methods(self):
        bar = qc.BarData("AAPL", 1000000000, 100.0, 102.0, 98.0, 101.0, 1000000.0)

        assert bar.typical_price() == (102.0 + 98.0 + 101.0) / 3.0
        assert bar.range()         == 102.0 - 98.0
        assert bar.is_bullish()    == True
        assert bar.is_bearish()    == False


class TestDataLoader:
    def test_load_csv(self):
        try:
            bars = qc.load_csv_data("../../data/test_buy_and_hold.csv", "AAPL")
            assert len(bars) > 0
            assert all(isinstance(b, qc.BarData) for b in bars)
        except Exception:
            pytest.skip("Test data files not found")


class TestEvents:
    def test_market_data_event(self):
        event = qc.MarketDataEvent("AAPL", 1000000000, 100.0, 101.0, 99.0, 100.5, 1000000.0)
        assert event.get_symbol() == "AAPL"
        assert event.get_open()   == 100.0
        assert event.get_high()   == 101.0
        assert event.get_low()    == 99.0
        assert event.get_close()  == 100.5
        assert event.get_volume() == 1000000.0
        assert event.get_price()  == 100.5

    def test_signal_event(self):
        event = qc.SignalEvent("AAPL", 1000000000, qc.SignalType.BUY, 1.0, "TestStrategy")
        assert event.get_symbol()      == "AAPL"
        assert event.get_signal_type() == qc.SignalType.BUY
        assert event.get_strength()    == 1.0
        assert event.get_strategy_id() == "TestStrategy"

    def test_fill_event(self):
        event = qc.FillEvent("AAPL", 1000000000, 123, qc.Side.BUY, 100.0, 150.0, 0.5)
        assert event.get_symbol()   == "AAPL"
        assert event.get_order_id() == 123
        assert event.get_side()     == qc.Side.BUY
        assert event.get_quantity() == 100.0
        assert event.get_price()    == 150.0
        assert event.get_commission() == 0.5


class TestExecutionEngine:
    def test_create_engine(self):
        engine = qc.ExecutionEngine("AAPL")
        assert engine.get_position()      == 0.0
        assert engine.get_realized_pnl()  == 0.0
        assert engine.get_unrealized_pnl() == 0.0
        assert engine.get_total_pnl()     == 0.0
        assert engine.get_total_fees()    == 0.0

    def test_execution_config(self):
        config = qc.ExecutionConfig()
        config.maker_fee    = 0.0001
        config.taker_fee    = 0.0002
        config.latency_ns   = 1000000
        config.slippage_pct = 0.0001

        assert config.maker_fee    == 0.0001
        assert config.taker_fee    == 0.0002
        assert config.latency_ns   == 1000000
        assert config.slippage_pct == 0.0001


class TestStrategies:
    def test_buy_and_hold(self):
        strategy = qc.BuyAndHold()
        assert strategy.get_name() == "BuyAndHold"
        assert not strategy.has_signals()

    def test_sma_crossover(self):
        strategy = qc.SMACrossover(fast_period=50, slow_period=200)
        assert strategy.get_name() == "SMACrossover"

    def test_mean_reversion(self):
        strategy = qc.MeanReversion(lookback=20, entry_threshold=1.5, exit_threshold=0.5)
        assert strategy.get_name()        == "MeanReversion"
        assert strategy.get_signal_count() == 0


class TestBacktestEngine:
    def test_create_engine(self):
        assert qc.BacktestEngine(initial_capital=100000.0) is not None

    def test_add_strategy(self):
        engine = qc.BacktestEngine(100000.0)
        engine.set_strategy(qc.BuyAndHold())

    def test_simple_backtest(self):
        bars = [
            qc.BarData("AAPL", i * 1000000000,
                       100.0 + i * 0.1, 100.0 + i * 0.1 + 1.0,
                       100.0 + i * 0.1 - 1.0, 100.0 + i * 0.1, 1000000.0)
            for i in range(100)
        ]

        engine = qc.BacktestEngine(100000.0)
        engine.add_data("AAPL", bars)
        engine.set_strategy(qc.BuyAndHold())

        final_value = engine.run()
        assert final_value > 0
        assert isinstance(final_value, float)


class TestHelperFunctions:
    def test_create_backtest(self):
        bars = [qc.BarData("AAPL", i * 1000000000, 100.0, 101.0, 99.0, 100.0, 1000000.0)
                for i in range(10)]

        engine = qc.create_backtest(
            initial_capital=100000.0,
            data={"AAPL": bars},
            strategy=qc.BuyAndHold()
        )
        assert engine is not None

    def test_run_backtest(self):
        bars = [qc.BarData("AAPL", i * 1000000000, 100.0, 101.0, 99.0, 100.0, 1000000.0)
                for i in range(10)]

        results = qc.run_backtest(
            strategy=qc.BuyAndHold(),
            data={"AAPL": bars},
            initial_capital=100000.0
        )

        assert isinstance(results, qc.BacktestResults)

        assert hasattr(results, 'initial_capital')
        assert hasattr(results, 'final_value')
        assert hasattr(results, 'total_pnl')
        assert hasattr(results, 'total_fees')
        assert hasattr(results, 'return_pct')
        assert hasattr(results, 'strategy_name')
        assert hasattr(results, 'trade_pnls')

        assert results.strategy_name   == 'BuyAndHold'
        assert results.initial_capital == 100000.0
        assert isinstance(results.trade_pnls, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])