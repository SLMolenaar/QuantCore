# QuantCore

![PyPI](https://img.shields.io/pypi/v/quantcore)
![Build](https://github.com/SLMolenaar/quantcore/actions/workflows/coverage.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![License](https://img.shields.io/badge/license-MIT-green)

High-performance backtesting engine for trading strategies, written in C++20 with a Python research interface.

---

## Overview

QuantCore is an event-driven backtester built around an enhanced version of [my limit order book simulator](https://github.com/SLMolenaar/orderbook-simulator-cpp). It processes market events chronologically through a priority queue, ensuring no look-ahead bias and no unrealistic assumptions about fill prices.

The C++ core handles all the performance-critical work: event dispatch, order matching, position tracking, and execution simulation. Python sits on top via pybind11 bindings and handles strategy development, parameter optimization, and visualization.

```
Market Data → EventQueue → Strategy → Signal → OrderBook → Fill → Portfolio
```

Both bar data and tick data are supported. The engine is bar-agnostic internally - ticks become
`MarketDataEvent` objects like bars do, so strategies work unchanged across both data types.

---

## Quick Start

### Bar data

```python
# pip install quantcore
import quantcore as qc

class MyStrategy(qc.Strategy):
    def on_data(self, event):
        if not self.has_position(event.symbol):
            self.generate_signal(event.symbol, qc.SignalType.BUY, 1.0, event.timestamp_ns)

results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': qc.load_csv_data('data/aapl.csv', 'AAPL')},
    initial_capital=100_000.0,
)
print(results)
```

### Tick data

```python
results = qc.run_tick_backtest(
    strategy=MyStrategy(),
    tick_data={'AAPL': qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')},
    initial_capital=100_000.0,
    mm_refresh_interval_ns=1_000_000_000,   # refresh MM quotes once per second
    equity_snapshot_interval_ns=60_000_000_000,  # snapshot equity once per minute
)
print(results)
```

The same strategy class works for both. `event.close` is the tick price when running on tick data.

You can also aggregate ticks to bars before running:

```python
ticks = qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')
bars  = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=60_000_000_000)  # 1-minute bars

results = qc.run_backtest(strategy=MyStrategy(), data={'AAPL': bars}, initial_capital=100_000.0)
```

The engine handles fills, position tracking, and PnL automatically. For a full tearsheet:

```python
from quantcore.analytics import calculate_all_metrics, calculate_returns
from quantcore.plotting import plot_full_tearsheet
import numpy as np

equity = np.array(results['equity_curve'])
returns = calculate_returns(equity)
print(calculate_all_metrics(equity))
plot_full_tearsheet(equity, returns)
```

For the full API reference and usage guide, see [docs/usage.md](docs/usage.md).

---

## Architecture
Every action goes through the event queue. When a strategy calls `generate_signal`, that signal becomes an `OrderEvent`, which goes through the order book, produces a `FillEvent`, which updates the portfolio. All in timestamp order. This is what prevents look-ahead bias: the strategy never sees data from the future.

![img.png](flowchart.png)

---

## Strategy Development

### Python Strategy

Subclass `qc.Strategy` and implement `on_data`. The same strategy works for bar and tick data - `event.close` is the close price for bars and the trade price for ticks.

```python
class BollingerBreakout(qc.Strategy):
    def __init__(self, window=20, n_std=2.0):
        super().__init__("BollingerBreakout")
        self.window = window
        self.n_std  = n_std
        self.prices = []

    def on_data(self, event):
        self.prices.append(event.close)
        if len(self.prices) < self.window:
            return

        window_prices = self.prices[-self.window:]
        mean = sum(window_prices) / self.window
        std  = (sum((p - mean) ** 2 for p in window_prices) / self.window) ** 0.5

        upper = mean + self.n_std * std
        lower = mean - self.n_std * std
        pos   = self.get_position(event.symbol)

        if event.close > upper and pos <= 0:
            self.generate_signal(event.symbol, qc.SignalType.BUY,  1.0, event.timestamp_ns)
        elif event.close < lower and pos >= 0:
            self.generate_signal(event.symbol, qc.SignalType.SELL, 1.0, event.timestamp_ns)

    def on_fill(self, fill):
        pass  # optional: react to fills
```

### Portfolio Context

Strategies can access full portfolio state:

```python
def on_data(self, event):
    portfolio = self.get_portfolio()
    if portfolio:
        equity     = portfolio.get_portfolio_value()
        cash       = portfolio.get_cash()
        position   = portfolio.get_position(event.symbol)
```

### Position Sizing

The engine ships with several sizing methods:

```python
from quantcore import FixedPercentage, RiskBased, KellyCriterion

engine = qc.BacktestEngine(100_000.0)
engine.set_position_sizer(qc.FixedPercentage(0.10))  # 10% of capital per trade
```

Built-in sizers: `FixedPercentage`, `RiskBased`, `KellyCriterion`, `EqualWeight`, `VolatilityTargeting`, `FixedShares`.

---

## Tick Data

### Loading

```python
# from CSV - columns: timestamp, price, quantity  (or with side: B/S/buy/sell)
ticks = qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')

# from Parquet
ticks = qc.load_tick_parquet('data/aapl_ticks.parquet', 'AAPL')

# numpy fast path - (N, 4) array: [timestamp_ns, price, quantity, side]
# side: 0.0 = Buy, 1.0 = Sell
arr = qc.load_tick_parquet('data/aapl_ticks.parquet', use_numpy=True)
engine.add_tick_data('AAPL', arr)
```

## Data Normalization

QuantCore does not adjust raw prices for corporate actions internally.
Feed it **adjusted data**, where close prices are continuous across splits
and dividends, for correct results.

Most data vendors offer adjusted data:
- Yahoo Finance: use `Adj Close` instead of `Close`
- Polygon.io: use the `adjusted=true` parameter
- CRSP: adjusted by default
- Quandl/NASDAQ Data Link: `WIKI/PRICES` table uses adjusted prices

If you have raw unadjusted data, use `CorporateActionsAdjuster`:
```python
from quantcore import CorporateActionsAdjuster

adjuster = CorporateActionsAdjuster.from_csv(
    splits_csv='data/aapl_splits.csv',
    dividends_csv='data/aapl_dividends.csv',
)
raw_bars    = qc.load_csv_data('data/aapl_raw.csv', 'AAPL')
adj_bars    = adjuster.adjust(raw_bars)
results     = qc.run_backtest(strategy=MyStrategy(), data={'AAPL': adj_bars}, ...)
```

### Aggregation

```python
# aggregate to any bar duration
bars_1min  = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=60_000_000_000)
bars_1hour = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=3_600_000_000_000)
bars_1day  = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=86_400_000_000_000)
```

### Engine configuration for tick data

Two settings matter most for tick performance:

```python
engine = qc.BacktestEngine(100_000.0)
engine.add_tick_data('AAPL', ticks)

# How often the market maker refreshes its quotes.
# 0 = every tick (default, slowest). 1s interval gives ~5x speedup on 1-second tick data.
engine.set_mm_refresh_interval(1_000_000_000)       # 1 second

# How often the equity curve is snapshotted.
# 0 = every tick (default). Has negligible performance impact but keeps the
# equity curve manageable for large tick datasets.
engine.set_equity_snapshot_interval(60_000_000_000) # 1 minute
```

`run_tick_backtest()` sets both to sensible defaults (1s and 60s respectively).

### CSV format

```
timestamp,price,quantity
1700000000,150.25,100
1700000001,150.30,50

# with aggressor side
timestamp,price,quantity,side
1700000000,150.25,100,buy
1700000001,150.30,50,sell
```

Timestamps in seconds, milliseconds, microseconds, or nanoseconds - detected automatically.

---

## Execution Simulation

### Order Types

`GOOD_TILL_CANCEL`, `IMMEDIATE_OR_CANCEL`, `FILL_OR_KILL`, `MARKET`, `GOOD_FOR_DAY`

### Fees & Slippage

```python
from quantcore import ExecutionConfig

config = ExecutionConfig()
config.maker_fee     = 0.001   # 0.1% maker
config.taker_fee     = 0.002   # 0.2% taker
config.slippage_pct  = 0.0005  # 0.05% slippage
config.latency_ns    = 1_000_000  # 1ms order latency

engine = qc.BacktestEngine(100_000.0, config)
```

### Risk Management

```python
from quantcore import RiskLimits

limits = qc.RiskLimits()
limits.max_position_pct = 0.20   # max 20% per position
limits.max_leverage     = 2.0
limits.max_loss_pct     = 0.15   # halt at 15% drawdown

engine.set_risk_limits(limits)
```

---

## Performance

Single-threaded. Measured on Windows (Release build, MSVC). Full results in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Order book**

| Pattern | Ops/s |
|---|---|
| Add + cancel (market-maker quote refresh) | 13.0 M ops/s |
| Add + match (taker sweep) | 4.9 M ops/s |

**End-to-end backtest - bar mode**

| Scenario | Bars/s | Latency (p99) |
|---|---|---|
| 1-year (252 bars) | ~270 K bars/s | 0.93 ms |
| 5-year (1,260 bars) | ~270 K bars/s | - |
| 1,000-year stress (252,000 bars) | ~290 K bars/s | - |

**End-to-end backtest - tick mode**

| Scenario | Ticks/s | Wall time |
|---|---|---|
| 10K ticks, no MM throttle | 315 K/s | 31.7 ms |
| 10K ticks, 1s MM throttle | 1.66 M/s | 6.0 ms |
| 10K ticks, 10s MM throttle | 2.78 M/s | 3.6 ms |
| 1M ticks → 1-min bars (aggregation) | 247 M/s | 4.1 ms |

The market-maker refresh interval is the main performance lever for tick data. With a 1-second
throttle the engine processes 1-second tick data at ~5x the unthrottled rate. See
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for the full breakdown.

Run the benchmarks yourself:

```bash
cmake --build build --target bench_backtest_engine bench_tick_data
./build/bench_backtest_engine
./build/bench_tick_data

python benchmarks/bench_python.py
python benchmarks/bench_tick_python.py
```

---

## Analytics

After running a backtest, the results dict contains an equity curve and trade log you can feed straight into the analytics module.

```python
from quantcore.analytics import calculate_all_metrics, calculate_returns

equity  = np.array(results['equity_curve'])
returns = calculate_returns(equity)
metrics = calculate_all_metrics(equity)

print(metrics)
# Total Return:     24.31%
# Annualized:       11.82%
# Sharpe Ratio:     1.43
# Sortino Ratio:    2.01
# Max Drawdown:     -8.74%
# Win Rate:         58.3%
```

Available metrics: total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, drawdown duration, win rate, profit factor, avg win/loss, largest win/loss.

### Visualizations

```python
from quantcore.plotting import (
    plot_full_tearsheet,
    plot_equity_curve,
    plot_underwater,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns_heatmap,
)

plot_full_tearsheet(equity, returns, timestamps=ts)
```

---

## Example Notebooks

| Notebook | Strategy | Concepts |
|---|---|---|
| [`mean_reversion.ipynb`](examples/mean_reversion.ipynb) | Z-score mean reversion | Parameter sensitivity, OU process |
| [`sma_crossover.ipynb`](examples/sma_crossover.ipynb) | SMA crossover | Trend following, signal generation |
| [`pairs_trading.ipynb`](examples/pairs_trading.ipynb) | Statistical arbitrage | Cointegration, spread trading |
| [`build_your_own_strategy.ipynb`](examples/build_your_own_strategy.ipynb) | Bollinger Band Breakout | Full walkthrough from scratch |

---

## Installation

### Prerequisites

- CMake 3.15+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 2022)
- Python 3.8+
- pybind11 (`pip install pybind11`)

### Build

```bash
git clone https://github.com/SLMolenaar/quantcore.git
cd quantcore

# build the C++ core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# build the Python bindings
cd python
pip install pybind11
python build_module.py

# verify
python -c "import quantcore; print(quantcore.version())"
```

### Run Tests

```bash
cmake --build build --target quantcore_tests
./build/quantcore_tests
```

For the full Python API reference, see [`docs/usage.md`](docs/usage.md).

---

## Project Structure

```
quantcore/
├── cpp/
│   ├── backtesting/          # Engine, events, portfolio
│   │   ├── tick_data.h       # TickData struct and TickSeries
│   │   └── tick_data_loader.h# CSV loader and aggregate_to_bars
│   ├── strategies/           # C++ strategy implementations
│   ├── orderbook/            # Order book (from orderbook-simulator-cpp)
│   └── tests/                # GoogleTest suite
├── python/
│   ├── quantcore/            # Python package
│   │   ├── __init__.py       # Public API
│   │   ├── analytics.py      # Performance metrics
│   │   ├── plotting.py       # Visualizations
│   │   └── tick_parquet_loader.py  # Parquet loader for tick data
│   ├── bindings.cpp          # pybind11 bindings
│   └── build_module.py       # Build helper
├── examples/                 # Jupyter notebooks
├── benchmarks/               # Benchmark suite
│   ├── bench_backtest_engine.cpp
│   ├── bench_tick_data.cpp
│   ├── bench_python.py
│   ├── bench_tick_python.py
│   └── RESULTS.md
├── CMakeLists.txt
└── README.md
```

---

## vs. Alternatives

| | QuantCore        | Backtrader | Zipline |
|---|------------------|------------|---|
| Core language | C++20            | Python     | Python |
| Order book simulation | ✅ Real LOB       | ❌          | ❌ |
| Tick data support | ✅ Native         | ❌          | ❌ |
| Event-driven | ✅                | ✅          | ✅ |
| Look-ahead prevention | ✅ Priority queue | ✅          | ✅ |
| Python strategy API | ✅ pybind11       | ✅ native   | ✅ native |
| Throughput (bars/s) | ~300K             | ~7.7K       | unverified |
| Maintenance | Active           | Stale      | Inactive |

*Throughput measured on SMA(50/200) crossover, 50K daily bars, Release build, Windows.
Reproduce: `python benchmarks/qcVsBacktrader.py --bars 50000 --runs 20`*

The main differentiator is the order book. Backtrader and Zipline assume you fill at the bar's close price. QuantCore routes orders through a real price-time priority matching engine, which gives you realistic partial fills, spread simulation, and tick-level execution when you have tick data.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open areas if you want to dig in:

- **VWAP / TWAP algos**: `ExecutionEngine`, child order slicing
- **Multi-strategy portfolio**: shared capital across strategies with a meta-allocator
- **Parallel sweeps on Linux**: `n_jobs` exists but Windows spawn overhead kills it; a Linux worker pool would make it actually useful

The engine doesn't handle corporate actions, survivorship bias, or timezone normalization. Feed it clean adjusted data and none of those are problems.

---

## License

MIT: see [LICENSE](LICENSE).
