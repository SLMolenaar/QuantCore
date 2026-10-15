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
        assert len(version) > 0

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
        assert event.get_symbol()     == "AAPL"
        assert event.get_order_id()   == 123
        assert event.get_side()       == qc.Side.BUY
        assert event.get_quantity()   == 100.0
        assert event.get_price()      == 150.0
        assert event.get_commission() == 0.5


class TestExecutionEngine:
    def test_create_engine(self):
        engine = qc.ExecutionEngine("AAPL")
        assert engine.get_position()       == 0.0
        assert engine.get_realized_pnl()   == 0.0
        assert engine.get_unrealized_pnl() == 0.0
        assert engine.get_total_pnl()      == 0.0
        assert engine.get_total_fees()     == 0.0

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
        assert strategy.get_name()         == "MeanReversion"
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


# ---------------------------------------------------------------------------
# Fill mode
# ---------------------------------------------------------------------------

class TestFillMode:
    DAY = 86_400 * 1_000_000_000

    def _bars(self):
        out = []
        for i in range(10):
            p = 100.0 if i == 0 else 110.0
            out.append(qc.BarData("T", (i + 1) * self.DAY, p, p + 1, p - 1, p, 1e6))
        return out

    def _entry(self, fill_mode):
        eng = qc.create_backtest(100_000.0, {"T": self._bars()}, qc.BuyAndHold(), fill_mode)
        eng.run()
        return eng.get_execution_engine("T").get_average_price()

    def test_default_is_same_bar_close(self):
        assert qc.BacktestEngine(1000.0).get_fill_mode() == qc.FillMode.SAME_BAR_CLOSE

    def test_next_bar_open_fills_at_gap_open(self):
        assert self._entry(None) == pytest.approx(100.0, abs=0.1)
        assert self._entry("next_bar_open") == pytest.approx(110.0, abs=0.1)
        assert self._entry(qc.FillMode.NEXT_BAR_OPEN) == pytest.approx(110.0, abs=0.1)

    def test_run_backtest_accepts_fill_mode(self):
        res = qc.run_backtest(qc.BuyAndHold(), {"T": self._bars()}, fill_mode="next_bar_open")
        assert res.final_value > 0

    def test_unknown_fill_mode_rejected(self):
        with pytest.raises(ValueError):
            qc.create_backtest(1000.0, fill_mode="whenever")

    def test_backtest_config_fill_mode(self):
        from quantcore.walk_forward import BacktestConfig
        from quantcore._engine_builder import build_engine
        eng = build_engine({"T": self._bars()}, 100_000.0, BacktestConfig(fill_mode="next_bar_open"))
        assert eng.get_fill_mode() == qc.FillMode.NEXT_BAR_OPEN

    def test_has_position_reflects_pending_order(self):
        # Buy on bar 1, sell on bar 3 gated only on has_position(): no duplicate
        # signals, so the run ends flat instead of opening an unintended short.
        seen = []

        class Gated(qc.Strategy):
            def __init__(self):
                super().__init__("Gated")
                self.n = 0

            def on_data(self, e):
                self.n += 1
                seen.append(self.has_position("T"))
                if self.n in (1, 2) and not self.has_position("T"):
                    self.generate_signal("T", qc.SignalType.BUY, 1.0, e.timestamp_ns)
                elif self.n in (4, 5) and self.has_position("T"):
                    self.generate_signal("T", qc.SignalType.SELL, 1.0, e.timestamp_ns)

        eng = qc.create_backtest(100_000.0, {"T": self._bars()}, Gated(), "next_bar_open")
        eng.run()
        assert seen[1] is True          # pending BUY already visible on the next bar
        assert eng.get_execution_engine("T").get_position() == 0.0


class TestCashSymmetry:
    """Long and short positions commit capital the same way (buying-power convention)."""

    def _run(self, sig):
        DAY = 86_400 * 1_000_000_000
        bars = [qc.BarData("A", (i + 1) * DAY, 100, 101, 99, 100, 1e6) for i in range(6)]

        class Once(qc.Strategy):
            def __init__(self):
                super().__init__("Once")
                self.n = 0

            def on_data(self, e):
                if self.n == 0:
                    self.generate_signal("A", sig, 1.0, e.timestamp_ns)
                self.n += 1

            def reset(self):
                super().reset()
                self.n = 0

        eng = qc.BacktestEngine(100_000.0)
        eng.add_data("A", bars)
        eng.set_strategy(Once())
        eng.run()
        return eng, eng.get_portfolio_context()

    def test_short_and_long_commit_capital_symmetrically(self):
        le, lp = self._run(qc.SignalType.BUY)
        se, sp = self._run(qc.SignalType.SELL)
        assert se.get_execution_engine("A").get_position() == pytest.approx(
            -le.get_execution_engine("A").get_position())
        assert sp.get_cash() == pytest.approx(lp.get_cash())
        assert sp.get_leverage() == pytest.approx(lp.get_leverage())

    def test_portfolio_value_equals_equity_for_shorts(self):
        eng, p = self._run(qc.SignalType.SELL)
        assert p.get_portfolio_value() == pytest.approx(eng.get_equity_curve()[-1])


class TestCashBalance:
    def test_cash_balance_includes_short_proceeds(self):
        t = TestCashSymmetry()
        le, lp = t._run(qc.SignalType.BUY)
        se, sp = t._run(qc.SignalType.SELL)
        equity = se.get_equity_curve()[-1]
        # long: cash spent; short: proceeds received (position value ~10k at 100 x 100)
        assert lp.get_cash_balance() == pytest.approx(lp.get_cash())
        assert sp.get_cash_balance() == pytest.approx(equity + 10_000.0, abs=1.0)
        assert sp.get_cash_balance() > sp.get_cash()

    def test_cash_balance_starts_at_initial_capital(self):
        assert qc.PortfolioContext(50_000.0).get_cash_balance() == 50_000.0


class TestMarketMakerRegression:
    def _run(self, spread):
        eng = qc.BacktestEngine(100_000.0)
        eng.configure_market_maker(levels=5, spread=spread, depth=100_000)
        DAY = 86_400 * 1_000_000_000
        eng.add_data("A", [qc.BarData("A", (i + 1) * DAY, 100, 101, 99, 100, 1e6) for i in range(6)])

        class Once(qc.Strategy):
            def __init__(self):
                super().__init__("Once")
                self.n = 0

            def on_data(self, e):
                if self.n == 0:
                    self.generate_signal("A", qc.SignalType.BUY, 1.0, e.timestamp_ns)
                self.n += 1

            def reset(self):
                super().reset()
                self.n = 0

        eng.set_strategy(Once())
        eng.run()
        return eng

    def test_runs_are_reproducible(self):
        vals = {self._run(0.0001).get_equity_curve()[-1] for _ in range(4)}
        assert len(vals) == 1

    @pytest.mark.parametrize("spread", [0.0, 1e-6, 1e-5])
    def test_tight_spread_still_fills(self, spread):
        assert self._run(spread).get_execution_engine("A").get_position() > 0

    def test_execution_engine_outlives_temporary_engine(self):
        ee = self._run(0.0001).get_execution_engine("A")
        assert ee.get_position() > 0


def test_strategy_portfolio_outlives_engine():
    import gc
    DAY = 86_400 * 10**9
    bars = [qc.BarData("A", (i + 1) * DAY, 100, 101, 99, 100, 1e6) for i in range(20)]
    st = qc.BuyAndHold()
    eng = qc.create_backtest(100_000.0, {"A": bars}, st)
    eng.run()
    p = st.get_portfolio()
    cash = p.get_cash()
    del eng
    gc.collect()
    assert p.get_cash() == cash
