"""Cross-check QuantCore against Backtrader on an identical SMA crossover.

Both engines trade next-bar-open, no fees, same share count, so trade count,
per-trade PnL and final equity should agree (within the market maker's one-cent quote floor).
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


def _run_qc(o, h, l, c, short=False):
    bars = [qc.BarData("A", (i + 1) * DAY, o[i], h[i], l[i], c[i], 1e6) for i in range(N)]

    class Cross(qc.Strategy):
        def __init__(self):
            super().__init__("Cross")
            self.closes = []

        def on_data(self, e):
            self.closes.append(e.close)
            if len(self.closes) < SLOW:
                return
            fast, slow = np.mean(self.closes[-FAST:]), np.mean(self.closes[-SLOW:])
            enter, exit_ = (fast < slow, fast > slow) if short else (fast > slow, fast < slow)
            open_sig, close_sig = (qc.SignalType.SELL, qc.SignalType.BUY) if short else (qc.SignalType.BUY, qc.SignalType.SELL)
            if enter and not self.has_position("A"):
                self.generate_signal("A", open_sig, 1.0, e.timestamp_ns)
            elif exit_ and self.has_position("A"):
                self.generate_signal("A", close_sig, 1.0, e.timestamp_ns)

        def reset(self):
            super().reset()
            self.closes = []

    cfg = qc.ExecutionConfig()
    cfg.maker_fee = cfg.taker_fee = cfg.slippage_pct = 0.0
    eng = qc.BacktestEngine(CASH, cfg)
    eng.configure_market_maker(levels=5, spread=0.0, depth=1_000_000)
    eng.set_fill_mode(qc.FillMode.NEXT_BAR_OPEN)
    eng.set_position_sizer(qc.FixedShares(SIZE))
    eng.add_data("A", bars)
    eng.set_strategy(Cross())
    eng.run()
    return eng


def _run_bt(o, h, l, c, size, short=False):
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6,
                       "openinterest": 0.0}, index=pd.date_range("2010-01-01", periods=N))
    trades = []

    class Cross(bt.Strategy):
        def __init__(self):
            self.f = bt.indicators.SMA(self.data.close, period=FAST)
            self.s = bt.indicators.SMA(self.data.close, period=SLOW)

        def next(self):
            enter, exit_ = (self.f[0] < self.s[0], self.f[0] > self.s[0]) if short else (self.f[0] > self.s[0], self.f[0] < self.s[0])
            if enter and not self.position:
                (self.sell if short else self.buy)(size=size)
            elif exit_ and self.position:
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


@pytest.mark.parametrize("short", [False, True], ids=["long", "short"])
def test_sma_crossover_matches_backtrader(short):
    o, h, l, c = _ohlc()
    eng = _run_qc(o, h, l, c, short)
    bt_pnls, bt_value = _run_bt(o, h, l, c, SIZE, short)
    qc_pnls = list(eng.get_trade_pnls())
    assert len(qc_pnls) >= 5
    assert len(qc_pnls) == len(bt_pnls)
    # QC fills at least one cent through the open (its market maker's quote floor), i.e.
    # up to ~$2 per 100-share round trip; a real accounting or timing bug is orders larger.
    assert qc_pnls == pytest.approx(bt_pnls, abs=0.035 * SIZE)
    assert eng.get_equity_curve()[-1] == pytest.approx(bt_value, abs=len(qc_pnls) * 0.035 * SIZE)


def test_accounting_identity_over_many_trades():
    o, h, l, c = _ohlc()
    eng = _run_qc(o, h, l, c)
    p, pos = eng.get_portfolio_context(), eng.get_execution_engine("A").get_position()
    equity = eng.get_equity_curve()[-1]
    assert p.get_portfolio_value() == pytest.approx(equity)
    assert p.get_cash_balance() + pos * c[-1] == pytest.approx(equity, abs=abs(pos) * 1.0)
    assert equity - CASH == pytest.approx(eng.get_total_pnl() - eng.get_total_fees(), abs=abs(pos) * 1.0 + 1e-6)
