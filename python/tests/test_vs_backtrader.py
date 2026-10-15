"""Cross-check QuantCore against Backtrader on an identical SMA crossover.

Both engines trade next-bar-open, no fees, same share count, so trade count,
per-trade PnL and final equity should agree (within market-maker spread).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import quantcore as qc

bt = pytest.importorskip("backtrader")
pd = pytest.importorskip("pandas")

SIZE = 100
N, FAST, SLOW, CASH, DAY = 1500, 10, 30, 1_000_000.0, 86_400 * 10**9


def _ohlc(seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    open_ = close * (1 + rng.uniform(-0.003, 0.003, N))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return open_, high, low, close


def _run_qc(o, h, l, c):
    bars = [qc.BarData("A", (i + 1) * DAY, o[i], h[i], l[i], c[i], 1e6) for i in range(N)]

    class Cross(qc.Strategy):
        def __init__(self):
            super().__init__("Cross")
            self.closes = []
            self.long = False  # own state: has_position() lags a pending next-bar fill

        def on_data(self, e):
            self.closes.append(e.close)
            if len(self.closes) < SLOW:
                return
            fast, slow = np.mean(self.closes[-FAST:]), np.mean(self.closes[-SLOW:])
            if fast > slow and not self.long:
                self.long = True
                self.generate_signal("A", qc.SignalType.BUY, 1.0, e.timestamp_ns)
            elif fast < slow and self.long:
                self.long = False
                self.generate_signal("A", qc.SignalType.SELL, 1.0, e.timestamp_ns)

        def reset(self):
            super().reset()
            self.closes = []
            self.long = False

    cfg = qc.ExecutionConfig()
    cfg.maker_fee = cfg.taker_fee = cfg.slippage_pct = 0.0
    eng = qc.BacktestEngine(CASH, cfg)
    eng.configure_market_maker(levels=5, spread=0.0001, depth=1_000_000)
    eng.set_fill_mode(qc.FillMode.NEXT_BAR_OPEN)
    eng.set_position_sizer(qc.FixedShares(SIZE))
    eng.add_data("A", bars)
    eng.set_strategy(Cross())
    eng.run()
    return eng


def _run_bt(o, h, l, c, size):
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6,
                       "openinterest": 0.0}, index=pd.date_range("2010-01-01", periods=N))
    trades = []

    class Cross(bt.Strategy):
        def __init__(self):
            self.f = bt.indicators.SMA(self.data.close, period=FAST)
            self.s = bt.indicators.SMA(self.data.close, period=SLOW)

        def next(self):
            if self.f[0] > self.s[0] and not self.position:
                self.buy(size=size)
            elif self.f[0] < self.s[0] and self.position:
                self.close()

        def notify_trade(self, t):
            if t.isclosed:
                trades.append(t.pnl)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(Cross)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(0.0)
    cerebro.run()
    return trades, cerebro.broker.getvalue()


def test_sma_crossover_matches_backtrader():
    o, h, l, c = _ohlc()
    eng = _run_qc(o, h, l, c)
    bt_pnls, bt_value = _run_bt(o, h, l, c, SIZE)
    qc_pnls = list(eng.get_trade_pnls())
    assert len(qc_pnls) >= 5
    assert len(qc_pnls) == len(bt_pnls)
    # QC fills on a cent grid around a synthetic market maker's spread, so allow a few
    # cents/share of round-trip difference; a real accounting or timing bug is orders larger.
    assert qc_pnls == pytest.approx(bt_pnls, abs=0.001 * SIZE * 100)
    assert eng.get_equity_curve()[-1] == pytest.approx(bt_value, abs=len(qc_pnls) * 10)


def test_accounting_identity_over_many_trades():
    o, h, l, c = _ohlc()
    eng = _run_qc(o, h, l, c)
    p, pos = eng.get_portfolio_context(), eng.get_execution_engine("A").get_position()
    equity = eng.get_equity_curve()[-1]
    assert p.get_portfolio_value() == pytest.approx(equity)
    assert p.get_cash_balance() + pos * c[-1] == pytest.approx(equity, abs=abs(pos) * 1.0)
    assert equity - CASH == pytest.approx(eng.get_total_pnl() - eng.get_total_fees(), abs=abs(pos) * 1.0 + 1e-6)
