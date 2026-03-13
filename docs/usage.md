# Usage Guide

This document covers everything you can do with QuantCore's Python interface. It assumes you have built the C++ extension and can `import quantcore as qc`. See the README for build instructions.

--- 

## Table of Contents

1. [Loading Data](#1-loading-data)
2. [Writing a Strategy](#2-writing-a-strategy)
3. [Running a Backtest](#3-running-a-backtest)
4. [Reading Results](#4-reading-results)
5. [Execution Simulation](#5-execution-simulation)
6. [Position Sizing](#6-position-sizing)
7. [Risk Management](#7-risk-management)
8. [Analytics](#8-analytics)
9. [Visualizations](#9-visualizations)
10. [Built-in Strategies](#10-built-in-strategies)
11. [Multi-Asset Backtests](#11-multi-asset-backtests)
12. [Parameter Sweeps](#12-parameter-sweeps)

---

## 1. Loading Data

### From CSV

The CSV loader expects columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`. Timestamps can be Unix epoch integers (any unit; the loader detects seconds, milliseconds, microseconds, nanoseconds) or ISO 8601 strings.

```python
import quantcore as qc

bars = qc.load_csv_data('data/aapl.csv', 'AAPL')
```

The first argument is the filepath. The second is the symbol name that will appear in `MarketDataEvent.symbol`. The loader returns a `List[BarData]`.

You can also use `CSVDataLoader` directly if you need lower-level access:

```python
bars = qc.CSVDataLoader.load('data/aapl.csv', 'AAPL')
```

### From Parquet

```python
from quantcore import load_parquet_data

bars = load_parquet_data('data/aapl.parquet', symbol='AAPL')
```

The Parquet loader accepts any column naming convention that maps to the five required fields (timestamp, open, high, low, close, volume). It handles `datetime64` index columns and integer epoch columns in all common units.

### BarData fields

Each `BarData` object exposes:

```python
bar.symbol        # str
bar.timestamp_ns  # int, nanoseconds since epoch
bar.open          # float
bar.high          # float
bar.low           # float
bar.close         # float
bar.volume        # float
bar.typical_price()   # (high + low + close) / 3
bar.mid_price()       # (high + low) / 2
```

### Inspecting loaded data

```python
print(f"Loaded {len(bars)} bars")
print(f"First bar: {bars[0].symbol} open={bars[0].open} close={bars[0].close}")
```

---

## 2. Writing a Strategy

Subclass `qc.Strategy` and implement `on_data`. Implementing `on_fill` is optional.

```python
import quantcore as qc

class MyStrategy(qc.Strategy):
    def on_data(self, event: qc.MarketDataEvent):
        # Called once per bar, per symbol
        pass

    def on_fill(self, fill: qc.FillEvent):
        # Called after each execution confirmation
        pass
```

### MarketDataEvent fields

```python
event.symbol        # str
event.timestamp_ns  # int
event.open          # float
event.high          # float
event.low           # float
event.close         # float  (use this as "current price")
event.volume        # float
```

### Generating signals

You do not place orders directly. You generate a signal and the engine converts it to an order using the configured position sizer and execution settings.

```python
self.generate_signal(
    symbol,            # str
    qc.SignalType.BUY, # or SELL, HOLD
    1.0,               # signal strength, 0.0–1.0, scales position size
    event.timestamp_ns # timestamp to stamp the signal
)
```

`SignalType.HOLD` is a no-op and is provided for explicitness in strategies that want to log a neutral state.

### Checking current position

```python
position = self.get_position(symbol)  # float, negative = short
has_pos  = self.has_position(symbol)  # bool, True if abs(position) > 0
```

### Accessing portfolio state

```python
portfolio = self.get_portfolio()  # PortfolioContext, or None if not yet attached
if portfolio:
    cash  = portfolio.get_cash()
    value = portfolio.get_portfolio_value()
```

### A complete example

```python
class ZScoreMeanReversion(qc.Strategy):
    def __init__(self, lookback: int = 20, entry_z: float = 1.5, exit_z: float = 0.5):
        super().__init__("ZScoreMeanReversion")
        self.lookback = lookback
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self._prices  = {}

    def on_data(self, event):
        sym = event.get_symbol()
        if sym not in self._prices:
            self._prices[sym] = []

        self._prices[sym].append(event.get_close())
        if len(self._prices[sym]) > self.lookback:
            self._prices[sym].pop(0)

        if len(self._prices[sym]) < self.lookback:
            return

        prices = self._prices[sym]
        mean   = sum(prices) / len(prices)
        std    = (sum((p - mean) ** 2 for p in prices) / len(prices)) ** 0.5

        if std == 0:
            return

        z_score  = (event.get_close() - mean) / std
        position = self.get_position(sym)

        if z_score < -self.entry_z and position == 0:
            self.generate_signal(sym, qc.SignalType.BUY, 1.0, event.get_timestamp())
        elif z_score > self.entry_z and position == 0:
            self.generate_signal(sym, qc.SignalType.SELL, 1.0, event.get_timestamp())
        elif position > 0 and z_score > -self.exit_z:
            self.generate_signal(sym, qc.SignalType.SELL, 1.0, event.get_timestamp())
        elif position < 0 and z_score < self.exit_z:
            self.generate_signal(sym, qc.SignalType.BUY, 1.0, event.get_timestamp())
```

### Using signal strength

Signal strength scales the position size calculated by the position sizer. A strength of `0.5` produces half the shares that `1.0` would:

```python
# Full conviction
self.generate_signal(sym, qc.SignalType.BUY, 1.0, event.get_timestamp())

# Half size (e.g., lower confidence)
self.generate_signal(sym, qc.SignalType.BUY, 0.5, event.get_timestamp())
```

---

## 3. Running a Backtest

### run_backtest (convenience function)

```python
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': qc.load_csv_data('data/aapl.csv', 'AAPL')},
    initial_capital=100_000.0,
)
```

This is the recommended entry point. It constructs a `BacktestEngine`, loads data, runs, and returns a `BacktestResults` object.

### BacktestEngine (direct)

Use this when you need to configure execution, position sizing, or risk limits before running.

```python
engine = qc.BacktestEngine(100_000.0)

engine.add_data('AAPL', qc.load_csv_data('data/aapl.csv', 'AAPL'))
engine.set_strategy(MyStrategy())

final_value = engine.run()
```

`run()` returns the final portfolio value as a `float`. It also resets internal state, so you can call it multiple times (useful for parameter sweeps).

### Passing an ExecutionConfig

```python
config = qc.ExecutionConfig()
config.maker_fee    = 0.001     # 0.1%
config.taker_fee    = 0.002     # 0.2%
config.slippage_pct = 0.0005    # 0.05%
config.latency_ns   = 1_000_000 # 1ms

engine = qc.BacktestEngine(100_000.0, config)
```

### Passing a PositionSizer

```python
sizer = qc.FixedPercentage(0.10)  # 10% of capital per trade
engine.set_position_sizer(sizer)
```

### Passing RiskLimits

```python
limits = qc.RiskLimits()
limits.max_position_pct = 0.20
limits.max_leverage     = 2.0
limits.max_loss_pct     = 0.15

engine.set_risk_limits(limits)
```

### create_backtest

`create_backtest` is an alternative to `run_backtest` that returns the engine without running it, letting you configure it further before calling `run()`:

```python
engine = qc.create_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars},
    initial_capital=100_000.0,
)
engine.set_position_sizer(qc.RiskBased(0.01))
engine.run()
```

---

## 4. Reading Results

### BacktestResults object

`run_backtest` returns a `BacktestResults` instance:

```python
print(results)
# ============================================================
#   Backtest Results - MyStrategy
# ============================================================
# Initial Capital:       $100,000.00
# Final Value:           $124,310.00
# Total PnL:              $24,310.00
# Total Fees:                $412.00
# Net PnL:                $24,310.00
# Return:                     24.31%
# ============================================================
```

Fields:

```python
results.strategy_name    # str
results.initial_capital  # float
results.final_value      # float
results.total_pnl        # float
results.total_fees       # float
results.net_pnl          # float (property: final_value - initial_capital)
results.return_pct       # float
results.equity_curve     # List[float]
results.timestamps       # List[int], nanoseconds
```

### Computing metrics

Call `.compute()` to calculate all performance metrics and cache them on the object:

```python
results.compute()
print(results.metrics)
```

The `metrics` property raises `RuntimeError` if you access it before calling `compute()`.

### Reading from the engine directly

If you used `BacktestEngine` directly:

```python
engine.get_total_pnl()         # float
engine.get_total_fees()        # float
engine.get_equity_curve()      # List[float]
engine.get_timestamps()        # List[int]
engine.get_portfolio_context() # PortfolioContext
```

`PortfolioContext` exposes current capital, equity, and a per-symbol position map.

---

## 5. Execution Simulation

### ExecutionConfig fields

```python
config = qc.ExecutionConfig()

config.maker_fee    = 0.001      # Fraction of notional, applied when order adds liquidity
config.taker_fee    = 0.002      # Fraction of notional, applied when order removes liquidity
config.slippage_pct = 0.0005     # Fraction of price, applied in direction of trade
config.latency_ns   = 1_000_000  # Nanoseconds delay between signal and order reaching the book
```

Fees and slippage are applied per fill. For a 100-share buy at $100 with `taker_fee=0.002` and `slippage_pct=0.0005`:

- Fill price: $100.05 (slippage pushes up)
- Fee: $100.05 × 100 × 0.002 = $20.01

### Order types

The order type determines how the order is handled if it cannot be fully filled immediately:

```python
qc.OrderType.GOOD_TILL_CANCEL   # Rests on the book until filled or canceled
qc.OrderType.MARKET             # Fills at best available price, no resting
qc.OrderType.IMMEDIATE_OR_CANCEL # Fills what it can immediately, cancels the rest
qc.OrderType.FILL_OR_KILL       # Fills entirely or not at all
qc.OrderType.GOOD_FOR_DAY       # Canceled at end of session if unfilled
```

The default order type used by the engine when converting a signal is `GOOD_TILL_CANCEL`. The order book simulates partial fills correctly for limit orders.

### ExecutionEngine (per-symbol access)

```python
exec_engine = engine.get_execution_engine('AAPL')

exec_engine.get_position()      # float, shares held (negative = short)
exec_engine.get_realized_pnl()  # float
exec_engine.get_total_fees()    # float
```

---

## 6. Position Sizing

Position sizers control how many shares are ordered when a signal is generated. All sizers respect constraints set via `set_max_position_size`, `set_min_position_size`, and `set_max_leverage`.

### FixedPercentage

Allocates a fixed fraction of current capital per position.

```python
sizer = qc.FixedPercentage(0.10)  # 10% per trade
```

### RiskBased

Sizes based on a fixed risk amount (fraction of capital) divided by stop-loss distance.

```python
sizer = qc.RiskBased(0.01)  # Risk 1% of capital per trade
engine.set_position_sizer(sizer)
engine.set_volatility_params(
    default_vol=0.02,   # fallback volatility estimate
    stop_distance=0.05, # stop-loss distance as fraction of price (e.g. 5%)
    lookback=20         # bars used for rolling volatility
)
```

`set_volatility_params` controls the `stop_loss_distance` the engine injects into every `PositionSizingContext`. Without it the stop distance falls back to the engine's internal default and `RiskBased` will produce zero sizes. In practice, set `stop_distance` to a fixed ATR-based estimate before running.

### KellyCriterion

Sizes using the Kelly formula: `f* = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win`.

```python
sizer = qc.KellyCriterion(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
```

Kelly sizing can produce large positions. Use `set_max_leverage` to cap it:

```python
sizer.set_max_leverage(1.0)  # No leverage
```

### EqualWeight

Divides capital equally across `n` positions.

```python
sizer = qc.EqualWeight(n_positions=5)  # 20% per position
```

### VolatilityTargeting

Scales leverage to hit a target annualized portfolio volatility.

```python
sizer = qc.VolatilityTargeting(target_vol=0.15)  # Target 15% annualized vol
```

Requires `portfolio_volatility` in the context. The engine provides a running volatility estimate if you have sufficient history; otherwise the sizer returns zero until enough data is available.

### FixedShares

Always orders the same number of shares regardless of capital or price.

```python
sizer = qc.FixedShares(100)  # Always order 100 shares
```

### Applying constraints

All sizers support the same constraint API:

```python
sizer.set_max_position_size(500)   # Max 500 shares per position
sizer.set_min_position_size(10)    # Don't order fewer than 10 shares
sizer.set_max_leverage(2.0)        # Notional cannot exceed 2x capital
```

### Python-side utilities: PositionCalculator

`PositionCalculator` is a standalone Python class for pre-trade sizing calculations outside the engine (useful in research notebooks):

```python
from quantcore import PositionCalculator

calc = PositionCalculator(capital=100_000.0)

shares = calc.fixed_percentage(price=150.0, pct=0.10)
shares = calc.risk_based(price=150.0, stop_price=145.0, risk_pct=0.01)
shares = calc.kelly(win_rate=0.55, avg_win=0.02, avg_loss=0.01, price=150.0)
shares = calc.equal_weight(price=150.0, n_positions=5)
```

### PortfolioPositionSizer

For multi-asset research, `PortfolioPositionSizer` computes sizes for an entire portfolio at once:

```python
from quantcore import PortfolioPositionSizer

sizer = PortfolioPositionSizer(capital=100_000.0)

allocations = sizer.equal_weight(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    prices={'AAPL': 175.0, 'GOOGL': 140.0, 'MSFT': 420.0}
)
# Returns dict: {'AAPL': 190, 'GOOGL': 238, 'MSFT': 79}
```

---

## 7. Risk Management

### RiskLimits fields

```python
limits = qc.RiskLimits()

limits.enabled            = True   # Set False to disable all checks
limits.max_position_pct   = 0.20   # Max position value / capital (per symbol)
limits.max_leverage       = 2.0    # Max total notional / capital
limits.max_loss_pct       = 0.15   # Halt if drawdown from peak exceeds this
limits.max_order_value    = 50_000 # Max notional per single order (0 = no limit)
```

### Attaching limits to the engine

```python
engine.set_risk_limits(limits)
retrieved = engine.get_risk_limits()
```

### RiskManager (direct access)

```python
risk_mgr = engine.get_risk_manager()

response = risk_mgr.check_order('AAPL', qc.Side.BUY, quantity=100, price=175.0)

if response.is_approved():
    print("Order approved")
else:
    print(f"Rejected: {response.result}, {response.reason}")
```

`RiskCheckResult` values: `APPROVED`, `REJECTED_POSITION_LIMIT`, `REJECTED_LEVERAGE`, `REJECTED_MAX_LOSS`, `REJECTED_ORDER_SIZE`.

### Behavior on rejection

A rejected order silently does not execute. The strategy is not notified of the rejection; it simply does not receive a `FillEvent`. The engine continues running normally.

If you need to detect rejections in strategy logic, check your position after expected fills do not arrive, or query `get_position()` on `on_data` before generating a signal.

---

## 8. Analytics

All analytics functions live in `quantcore.analytics` and operate on NumPy arrays.

### Setup

```python
import numpy as np
from quantcore.analytics import (
    calculate_returns,
    calculate_all_metrics,
    rolling_sharpe,
    rolling_volatility,
    monthly_returns,
    underwater_plot_data,
)

equity     = np.array(results.equity_curve)
timestamps = np.array(results.timestamps)
returns    = calculate_returns(equity)
```

### calculate_returns

Converts an equity curve to period returns:

```python
returns = calculate_returns(equity)
# array of floats, length len(equity) - 1
# returns[i] = (equity[i+1] - equity[i]) / equity[i]
```

### calculate_all_metrics

Returns a `PerformanceMetrics` dataclass with all metrics populated. Printing it gives a formatted summary:

```python
metrics = calculate_all_metrics(equity)
print(metrics)
# Total Return:     24.31%
# Annualized:       11.82%
# Sharpe Ratio:     1.43
# Sortino Ratio:    2.01
# Calmar Ratio:     1.35
# Max Drawdown:     -8.74%
# Win Rate:         58.3%
# Profit Factor:    1.82
# Avg Win:          $124.00
# Avg Loss:         $-68.00
# Largest Win:      $341.00
# Largest Loss:     $-187.00
```

To include trade-level metrics, pass a list of per-trade PnL values:

```python
metrics = calculate_all_metrics(equity, trade_pnls=[100.0, -50.0, 200.0])
print(metrics.total_trades)   # 3
print(metrics.win_rate)       # float, 0–100
print(metrics.profit_factor)  # float
```

### Individual metric functions

```python
from quantcore.analytics import (
    calculate_total_return,
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    analyze_trades,
)

total_return = calculate_total_return(equity)                        # float, %
ann_ret      = calculate_annualized_return(equity, periods_per_year=252)  # float, %
sharpe       = calculate_sharpe_ratio(returns)                       # float
sortino      = calculate_sortino_ratio(returns)                      # float
vol          = calculate_volatility(returns, periods_per_year=252)   # float, %
max_dd, dur  = calculate_max_drawdown(equity)                        # (float %, int bars)
calmar       = calculate_calmar_ratio(ann_ret, max_dd)               # float

trade_stats  = analyze_trades([100.0, -50.0, 200.0, -30.0, 80.0])
# dict with keys: total_trades, win_rate, profit_factor,
#                 avg_win, avg_loss, largest_win, largest_loss
```

### Rolling metrics

```python
roll_sharpe = rolling_sharpe(returns, window=60, periods_per_year=252)
# np.ndarray of length len(returns) - window + 1

roll_vol = rolling_volatility(returns, window=60, periods_per_year=252)
# np.ndarray, annualized volatility in %
```

### Monthly returns table

Requires timestamps:

```python
monthly = monthly_returns(equity, timestamps)
# pandas DataFrame, index=year, columns=month name ('Jan', 'Feb', ...)
print(monthly)
```

### Underwater (drawdown) data

```python
dd = underwater_plot_data(equity)
# np.ndarray, same length as equity_curve
# values are drawdown % from running peak, always <= 0
```

---

## 9. Visualizations

All plotting functions live in `quantcore.plotting` and return `matplotlib.figure.Figure`. You can call `.show()` on the returned figure or save it.

```python
from quantcore.plotting import (
    plot_equity_curve,
    plot_underwater,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns_heatmap,
    plot_trade_analysis,
    plot_full_tearsheet,
    save_all_plots,
)
```

### plot_equity_curve

```python
fig = plot_equity_curve(
    equity_curve=equity,
    timestamps=timestamps,   # optional, enables date axis
    title="My Strategy",
    figsize=(14, 7),
    show_drawdown=True       # shade drawdown periods in red
)
fig.show()
```

### plot_underwater

Dedicated drawdown chart:

```python
fig = plot_underwater(equity, timestamps=timestamps)
```

### plot_returns_distribution

Histogram of period returns with mean line:

```python
fig = plot_returns_distribution(returns, title="Return Distribution")
```

### plot_rolling_metrics

Two-panel chart: rolling Sharpe (top) and rolling volatility (bottom):

```python
fig = plot_rolling_metrics(
    returns,
    timestamps=timestamps,
    window=60,
    title="Rolling Metrics"
)
```

### plot_monthly_returns_heatmap

Requires a monthly returns DataFrame from `analytics.monthly_returns`:

```python
from quantcore.analytics import monthly_returns

monthly = monthly_returns(equity, timestamps)
fig = plot_monthly_returns_heatmap(monthly)
```

### plot_trade_analysis

Scatter plot of individual trade returns plus cumulative:

```python
entry_prices = [100.0, 105.0, 110.0]
exit_prices  = [103.0, 102.0, 115.0]

fig = plot_trade_analysis(entry_prices, exit_prices)
```

### plot_full_tearsheet

All charts in one figure (equity curve, drawdown, returns distribution, rolling Sharpe, rolling volatility):

```python
fig = plot_full_tearsheet(equity, returns, timestamps=timestamps)
fig.show()
```

### save_all_plots

Saves individual plot files to a directory:

```python
save_all_plots(
    equity_curve=equity,
    returns=returns,
    timestamps=timestamps,
    output_dir='plots',
    strategy_name='mean_reversion'
)
# Writes: mean_reversion_equity.png, _underwater.png, _returns_dist.png,
#         _rolling.png, _tearsheet.png
```

---

## 10. Built-in Strategies

These are available as both C++ classes (for performance) and as Python subclasses via pybind11.

### BuyAndHold

Buys on the first bar, holds until the end. Used as a benchmark and in tests.

```python
strategy = qc.BuyAndHold()
```

### SMACrossover

Generates buy signals when the fast SMA crosses above the slow SMA, sell signals on the opposite crossover.

```python
strategy = qc.SMACrossover(
    fast_period=50,
    slow_period=200
)
```

### MeanReversion

Z-score based mean reversion. Buys when price is more than `entry_threshold` standard deviations below the rolling mean, sells when it reverts to within `exit_threshold` standard deviations.

```python
strategy = qc.MeanReversion(
    lookback=20,
    entry_threshold=1.5,
    exit_threshold=0.5
)
```

### PairsTrading

Statistical arbitrage on two correlated assets. Monitors the spread, buys the underperformer and sells the outperformer when the spread diverges beyond a threshold.

```python
strategy = qc.PairsTrading(
    symbol1='AAPL',
    symbol2='MSFT',
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5
)
```

Requires both symbols to be loaded via `add_data` or passed in the `data` dict.

---

## 11. Multi-Asset Backtests

Pass multiple symbols in the `data` dict. The engine interleaves their bars in timestamp order automatically.

```python
results = qc.run_backtest(
    strategy=MyMultiAssetStrategy(),
    data={
        'AAPL':  qc.load_csv_data('data/aapl.csv',  'AAPL'),
        'GOOGL': qc.load_csv_data('data/googl.csv', 'GOOGL'),
        'MSFT':  qc.load_csv_data('data/msft.csv',  'MSFT'),
    },
    initial_capital=100_000.0,
)
```

Inside the strategy, `event.get_symbol()` tells you which asset triggered the call:

```python
class MultiAssetStrategy(qc.Strategy):
    def on_data(self, event):
        if event.get_symbol() == 'AAPL':
            # AAPL-specific logic
            pass
        elif event.get_symbol() == 'GOOGL':
            # GOOGL-specific logic
            pass
```

Capital is shared across all positions. The position sizer and risk limits apply per-symbol.

---

## 12. Parameter Sweeps

The engine is reentrant: `run()` resets internal state and can be called multiple times. Use this for grid searches.

### Simple grid search

```python
import itertools
import numpy as np
from quantcore.analytics import calculate_sharpe_ratio, calculate_returns

bars = qc.load_csv_data('data/aapl.csv', 'AAPL')

fast_periods = [10, 20, 50]
slow_periods = [100, 150, 200]
results_grid = {}

for fast, slow in itertools.product(fast_periods, slow_periods):
    if fast >= slow:
        continue

    strategy = qc.SMACrossover(fast_period=fast, slow_period=slow)
    engine   = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', bars)
    engine.set_strategy(strategy)
    engine.run()

    equity  = np.array(engine.get_equity_curve())
    returns = calculate_returns(equity)
    sharpe  = calculate_sharpe_ratio(returns)

    results_grid[(fast, slow)] = sharpe
    print(f"SMA({fast}/{slow}): Sharpe={sharpe:.2f}")

best = max(results_grid, key=results_grid.get)
print(f"\nBest parameters: SMA({best[0]}/{best[1]}) Sharpe={results_grid[best]:.2f}")
```

### Parallel sweep with multiprocessing

Each engine is independent with no shared state, so a parameter sweep is embarrassingly parallel:

```python
from multiprocessing import Pool

def run_single(params):
    fast, slow = params
    bars     = qc.load_csv_data('data/aapl.csv', 'AAPL')
    strategy = qc.SMACrossover(fast_period=fast, slow_period=slow)
    engine   = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', bars)
    engine.set_strategy(strategy)
    engine.run()

    equity  = np.array(engine.get_equity_curve())
    returns = calculate_returns(equity)
    return (fast, slow, calculate_sharpe_ratio(returns))

param_grid = [
    (f, s) for f in [10, 20, 50]
    for s in [100, 150, 200]
    if f < s
]

with Pool() as pool:
    results = pool.map(run_single, param_grid)

for fast, slow, sharpe in sorted(results, key=lambda x: -x[2]):
    print(f"SMA({fast}/{slow}): {sharpe:.2f}")
```

### Walk-forward analysis

A basic walk-forward setup: optimize in-sample, evaluate out-of-sample, roll forward:

```python
import numpy as np
from quantcore.analytics import calculate_sharpe_ratio, calculate_returns

all_bars   = qc.load_csv_data('data/aapl.csv', 'AAPL')
n_bars     = len(all_bars)
is_window  = 252   # 1 year in-sample
oos_window = 63    # 1 quarter out-of-sample

oos_sharpes = []

for start in range(0, n_bars - is_window - oos_window, oos_window):
    is_bars  = all_bars[start : start + is_window]
    oos_bars = all_bars[start + is_window : start + is_window + oos_window]

    # In-sample optimization
    best_sharpe, best_params = -np.inf, (50, 200)
    for fast, slow in [(10, 50), (20, 100), (50, 200)]:
        engine = qc.BacktestEngine(100_000.0)
        engine.add_data('AAPL', is_bars)
        engine.set_strategy(qc.SMACrossover(fast, slow))
        engine.run()
        eq  = np.array(engine.get_equity_curve())
        sr  = calculate_sharpe_ratio(calculate_returns(eq))
        if sr > best_sharpe:
            best_sharpe, best_params = sr, (fast, slow)

    # Out-of-sample evaluation
    engine = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', oos_bars)
    engine.set_strategy(qc.SMACrossover(*best_params))
    engine.run()
    eq  = np.array(engine.get_equity_curve())
    oos_sr = calculate_sharpe_ratio(calculate_returns(eq))
    oos_sharpes.append(oos_sr)

    print(f"Period {start}–{start+is_window+oos_window}: "
          f"best params={best_params}, OOS Sharpe={oos_sr:.2f}")

print(f"\nMean OOS Sharpe: {np.mean(oos_sharpes):.2f}")
```