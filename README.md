# QuantCore

High-performance backtesting engine for trading strategies, written in C++20 with a Python research interface.

---

## Overview

QuantCore is an event-driven backtester built around [my own limit order book](https://github.com/SLMolenaar/orderbook-simulator-cpp). It processes market events chronologically through a priority queue. No look-ahead bias, no shortcut assumptions about fill prices.

The C++ core handles all the performance-critical work: event dispatch, order matching, position tracking, and execution simulation. Python sits on top via pybind11 bindings and handles strategy development, parameter optimization, and visualization.

```
Market Data → EventQueue → Strategy → Signal → OrderBook → Fill → Portfolio
```

---

## Quick Start

```python
import quantcore as qc

class MyStrategy(qc.Strategy):
    def on_data(self, event):
        if not self.has_position(event.symbol):
            self.generate_signal(event.symbol, qc.SignalType.BUY, 1.0, event.timestamp_ns)

results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': qc.load_csv_data('AAPL', 'data/aapl.csv')},
    initial_capital=100_000.0,
)
print(results)
```

That's it. The engine handles fills, position tracking, and PnL automatically. For a full tearsheet:

```python
from quantcore.analytics import calculate_all_metrics, calculate_returns
from quantcore.plotting import plot_full_tearsheet
import numpy as np

equity = np.array(results['equity_curve'])
returns = calculate_returns(equity)
print(calculate_all_metrics(equity))
plot_full_tearsheet(equity, returns)
```

---

## Architecture
Every action goes through the event queue. When a strategy calls `generate_signal`, that signal becomes an `OrderEvent`, which goes through the order book, produces a `FillEvent`, which updates the portfolio. All in timestamp order. This is what prevents look-ahead bias: the strategy never sees data from the future.

![img.png](flowchart.png)

---

## Strategy Development

### Python Strategy

Subclass `qc.Strategy` and implement `on_data`. Signals drive order execution. You don't place orders directly, you generate signals and the engine handles the rest.

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

The engine is single-threaded. Measured on Windows (AMD/Intel, Release build, MSVC):

| Scenario | Bars/s | Latency (p99) |
|---|---|---|
| 1-year backtest (252 bars) | ~265 K bars/s | 0.95 ms |
| 5-year backtest (1,260 bars) | ~270 K bars/s | 1.75 ms |
| Order book — add + cancel | 12.9 M ops/s | — |
| Order book — add + match | 4.9 M ops/s | — |

All three 1-year latency checks pass the < 5ms target. The order book isolation numbers are the ceiling. The engine loop itself is the bottleneck at this scale, not the matching engine.

Run the benchmark suite yourself:

```bash
cmake --build build --target quantcore_benchmarks
./build/quantcore_benchmarks
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

---

## Project Structure

```
quantcore/
├── cpp/
│   ├── backtesting/          # Engine, events, portfolio
│   ├── strategies/           # C++ strategy implementations
│   ├── orderbook/            # Order book (from orderbook-simulator-cpp)
│   └── tests/                # GoogleTest suite
├── python/
│   ├── quantcore/            # Python package
│   │   ├── __init__.py       # Public API
│   │   ├── analytics.py      # Performance metrics
│   │   └── plotting.py       # Visualizations
│   ├── bindings.cpp          # pybind11 bindings
│   └── build_module.py       # Build helper
├── examples/                 # Jupyter notebooks
├── benchmarks/               # Benchmark suite
├── CMakeLists.txt
└── README.md
```

---

## vs. Alternatives

| | QuantCore | Backtrader | Zipline |
|---|---|---|---|
| Core language | C++20 | Python | Python |
| Order book simulation | ✅ Real LOB | ❌ | ❌ |
| Event-driven | ✅ | ✅ | ✅ |
| Look-ahead prevention | ✅ Priority queue | ✅ | ✅ |
| Python strategy API | ✅ pybind11 | ✅ native | ✅ native |
| Throughput (bars/s) | ~265 K | ~50 K | ~100 K |
| Maintenance | Active | Stale | Inactive |

The main differentiator is the order book. Backtrader and Zipline assume you fill at the bar's close price. QuantCore routes orders through a real price-time priority matching engine, which gives you realistic partial fills, spread simulation, and tick-level execution when you have tick data.

---

## License

MIT: see [LICENSE](LICENSE).