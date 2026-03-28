# QuantCore

![PyPI](https://img.shields.io/pypi/v/quantcore)
![Build](https://github.com/SLMolenaar/quantcore/actions/workflows/coverage.yml/badge.svg)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![License](https://img.shields.io/badge/license-MIT-green)

High-performance backtesting engine written in C++20 with a Python research interface.

---

## Overview

QuantCore is an event-driven backtester built around an enhanced version of [my limit order book simulator](https://github.com/SLMolenaar/orderbook-simulator-cpp). Market events are processed chronologically through a priority queue, so there's no look-ahead bias and no assumptions about fill prices. Orders go through a real price-time priority matching engine.

Both bar and tick data are supported. Strategies work unchanged across both.

```
Market Data → EventQueue → Strategy → Signal → OrderBook → Fill → Portfolio
```

---

## Quick Start

```python
pip install quantcore
```

```python
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

Tick data works the same way:

```python
results = qc.run_tick_backtest(
    strategy=MyStrategy(),
    tick_data={'AAPL': qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')},
    initial_capital=100_000.0,
)
```

`event.close` is the tick price when running on tick data. No strategy changes needed.

For the full API reference, see [docs/usage.md](docs/usage.md).

---

## Performance

Single-threaded. Measured on Windows (Release build, MSVC).

**Order book**

| Pattern | Ops/s |
|---|---|
| Add + cancel (MM quote refresh) | 13.0 M ops/s |
| Add + match (taker sweep) | 4.9 M ops/s |

**Bar mode**

| Scenario | Bars/s | p99 latency |
|---|---|---|
| 1-year (252 bars) | ~270 K/s | 0.93 ms |
| 5-year (1,260 bars) | ~270 K/s | - |
| 1,000-year stress (252,000 bars) | ~290 K/s | - |

**Tick mode**

| Scenario | Ticks/s | Wall time |
|---|---|---|
| 10K ticks, no MM throttle | 315 K/s | 31.7 ms |
| 10K ticks, 1s MM throttle | 1.66 M/s | 6.0 ms |
| 1M ticks aggregated to 1-min bars | 247 M/s | 4.1 ms |

The MM refresh interval is the main lever for tick performance. See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for the full breakdown.

---

## vs. Alternatives

| | QuantCore | Backtrader | Zipline |
|---|---|---|---|
| Core language | C++20 | Python | Python |
| Order book simulation | Yes | No | No |
| Tick data | Yes | No | No |
| Event-driven | Yes | Yes | Yes |
| Look-ahead prevention | Priority queue | Yes | Yes |
| Python strategy API | pybind11 | Native | Native |
| Throughput (bars/s) | ~300K | ~7.7K | unverified |
| Maintenance | Active | Stale | Inactive |

*Throughput: SMA(50/200) crossover, 50K daily bars, Release build, Windows.*
*Reproduce: `python benchmarks/qcVsBacktrader.py --bars 50000 --runs 20`*

The core differentiator is the order book. Backtrader and Zipline fill at close price. QuantCore routes orders through a real matching engine, giving you realistic partial fills, spread simulation, and tick-level execution.

---

## Installation

**Prerequisites:** CMake 3.15+, C++20 compiler (GCC 10+, Clang 12+, MSVC 2022), Python 3.8+

```bash
git clone https://github.com/SLMolenaar/quantcore.git
cd quantcore

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

cd python
pip install pybind11
python build_module.py

python -c "import quantcore; print(quantcore.version())"
```

**Run tests:**

```bash
cmake --build build --target quantcore_tests
./build/quantcore_tests
```

---

## Examples

| Notebook | Strategy | Concepts |
|---|---|---|
| [`mean_reversion.ipynb`](examples/mean_reversion.ipynb) | Z-score mean reversion | Parameter sensitivity, OU process |
| [`sma_crossover.ipynb`](examples/sma_crossover.ipynb) | SMA crossover | Trend following, signal generation |
| [`pairs_trading.ipynb`](examples/pairs_trading.ipynb) | Statistical arbitrage | Cointegration, spread trading |
| [`build_your_own_strategy.ipynb`](examples/build_your_own_strategy.ipynb) | Bollinger Band Breakout | Full walkthrough from scratch |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open areas:

- **VWAP / TWAP algos**: child order slicing in `ExecutionEngine`
- **Multi-strategy portfolio**: shared capital with a meta-allocator
- **Parallel sweeps**: `n_jobs` works but Windows process spawn overhead (~500ms per worker) makes it slower than sequential for short backtests. On Linux the break-even is much lower and it works as expected. Use Linux or WSL for parallel parameter sweeps.

The engine doesn't handle survivorship bias or timezone normalization. Feed it clean adjusted data and those aren't problems.

---

## References

Architecture and methodology draws on the following.

**Event-driven design**
- [QuantStart, Event-Driven Backtesting with Python](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/); the series that popularized the signal → order → fill event loop pattern for retail quant backtesting.
- [Martin Fowler, Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html); good primer on the broader EDA pattern and why a priority queue beats a polling loop.

**Backtesting methodology**
- [Marcos Lopez de Prado, Advances in Financial Machine Learning (2018)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086); chapters 11–12 cover combinatorial purged cross-validation and walk-forward analysis; the basis for how WFA is framed in this library.
- [Bailey et al., The Deflated Sharpe Ratio (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551); why naive Sharpe optimization overfits and what to do about it. Informed the Monte Carlo validation approach.
- [Chan, Algorithmic Trading (2013)](https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146); practical treatment of mean reversion, momentum, and WFA for retail-scale strategies.

---

## License

MIT: see [LICENSE](LICENSE).