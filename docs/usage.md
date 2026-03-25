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
12. [Parameter Sweeps and Optimization](#12-parameter-sweeps-and-optimization)
13. [Walk-Forward Analysis](#13-walk-forward-analysis)
14. [Monte Carlo Validation](#14-monte-carlo-validation)
15. [Tick Data](#15-tick-data)

---

## 1. Loading Data

### From CSV

The CSV loader expects columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`. Timestamps can be Unix epoch integers (any unit; the loader detects seconds, milliseconds, microseconds, nanoseconds) or ISO 8601 strings. The file can have 6 columns (timestamp + OHLCV) or 7 columns (symbol + timestamp + OHLCV).

```python
import quantcore as qc

bars = qc.load_csv_data('data/aapl.csv', 'AAPL')
```

The first argument is the filepath. The second is the symbol name assigned when the file has no symbol column. The loader returns a `List[BarData]`.

You can also call `CSVDataLoader` directly for lower-level access, including control over the skip threshold:

```python
bars = qc.CSVDataLoader.load(
    'data/aapl.csv',
    symbol='AAPL',
    has_header=True,
    max_skip_pct=0.20   # raise an error if more than 20% of rows fail to parse
)
```

`max_skip_pct` defaults to `0.20`. Set it lower to enforce stricter data quality.

To filter out weekends and exchange holidays before the data reaches the engine, pass a `calendar` argument:

```python
bars = qc.load_csv_data('data/aapl.csv', 'AAPL', calendar='NYSE')
```

See [Trading Calendar](#16-trading-calendar) for details.

### From Parquet

```python
from quantcore import load_parquet_data

# Returns List[BarData] - same as load_csv_data
bars = load_parquet_data('data/aapl.parquet', symbol='AAPL', use_numpy=False)

# Returns a (N, 6) float64 numpy array - faster path for add_data (see below)
arr = load_parquet_data('data/aapl.parquet', symbol='AAPL', use_numpy=True)

# Filter to trading days before returning (forces use_numpy=False internally)
bars = load_parquet_data('data/aapl.parquet', symbol='AAPL', calendar='NYSE')
```

The Parquet loader accepts any column naming convention that maps to the five required fields. It handles `datetime64` index columns and integer epoch columns in all common units. You can also call `ParquetDataLoader` directly:

```python
from quantcore import ParquetDataLoader

bars = ParquetDataLoader.load('data/aapl.parquet', symbol='AAPL')  # List[BarData]
arr  = ParquetDataLoader.load_numpy('data/aapl.parquet')            # (N, 6) float64 ndarray
```

`load_numpy` returns columns in the order `[timestamp_ns, open, high, low, close, volume]` and is designed for the fast numpy `add_data` overload (see [Running a Backtest](#3-running-a-backtest)).

### Tick data from CSV

`load_tick_csv` loads trade tick data. Accepted column layouts (auto-detected by column count):

```
3 cols: timestamp, price, quantity
4 cols: timestamp, price, quantity, side   (B/b/buy/BUY or S/s/sell/SELL)
5 cols: symbol, timestamp, price, quantity, side
```

```python
ticks = qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')
```

You can also call `TickDataLoader` directly:

```python
ticks = qc.TickDataLoader.load('data/aapl_ticks.csv', symbol='AAPL', has_header=True)
```

### Tick data from Parquet

```python
from quantcore import load_tick_parquet

# Returns List[TickData]
ticks = load_tick_parquet('data/aapl_ticks.parquet', symbol='AAPL', use_numpy=False)

# Returns a (N, 4) float64 numpy array: [timestamp_ns, price, quantity, side]
# side encoding: 0.0 = Buy, 1.0 = Sell - use directly with add_tick_data(symbol, array)
arr = load_tick_parquet('data/aapl_ticks.parquet', use_numpy=True)
```

You can also call `TickParquetLoader` directly:

```python
from quantcore import TickParquetLoader

ticks = TickParquetLoader.load('data/aapl_ticks.parquet', symbol='AAPL')  # List[TickData]
arr   = TickParquetLoader.load_numpy('data/aapl_ticks.parquet')            # (N, 4) float64 ndarray
```

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

bar.typical_price()  # (high + low + close) / 3
bar.range()          # high - low
bar.is_bullish()     # bool, True if close > open
bar.is_bearish()     # bool, True if close < open
```

### TickData fields

```python
tick.symbol         # str
tick.timestamp_ns   # int, nanoseconds since epoch
tick.price          # float
tick.quantity       # float
tick.aggressor_side # qc.Side.BUY or qc.Side.SELL
```

### Aggregating ticks to bars

If your strategy is bar-based, aggregate before running:

```python
# bar_duration_ns common values:
#   1_000_000_000      - 1 second
#   60_000_000_000     - 1 minute
#   3_600_000_000_000  - 1 hour
#   86_400_000_000_000 - 1 day
bars = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=60_000_000_000)
```

### Inspecting loaded data

```python
print(f"Loaded {len(bars)} bars")
print(f"First bar: {bars[0].symbol} open={bars[0].open} close={bars[0].close}")
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

---

## 2. Writing a Strategy

Subclass `qc.Strategy` and implement `on_data`. Implementing `on_fill` and `on_rejected` is optional. The same strategy class works for both bar and tick data - `event.close` is the close price for bars and the trade price for ticks.

```python
import quantcore as qc

class MyStrategy(qc.Strategy):
    def on_data(self, event: qc.MarketDataEvent):
        # Called once per bar or tick, per symbol
        pass

    def on_fill(self, fill: qc.FillEvent):
        # Called after each execution confirmation
        pass

    def on_rejected(self, symbol: str, reason: str):
        # Called when a signal is rejected by risk limits
        pass
```

### MarketDataEvent fields

Fields are accessible as both attributes and getter methods - both styles work and are used interchangeably:

```python
# Attribute access
event.symbol        # str
event.timestamp_ns  # int  (nanoseconds since epoch)
event.open          # float
event.high          # float
event.low           # float
event.close         # float  (close price for bars; trade price for ticks)
event.volume        # float

# Equivalent getter methods
event.get_symbol()      # str
event.get_timestamp()   # int  (nanoseconds since epoch)
event.get_open()        # float
event.get_high()        # float
event.get_low()         # float
event.get_close()       # float
event.get_volume()      # float
event.get_price()       # float  (alias for get_close)
```

### FillEvent fields

`on_fill` receives a `FillEvent` with the following fields:

```python
fill.get_symbol()     # str
fill.get_timestamp()  # int, nanoseconds since epoch
fill.get_order_id()   # int
fill.get_side()       # qc.Side.BUY or qc.Side.SELL
fill.get_quantity()   # float, shares filled
fill.get_price()      # float, fill price (after slippage)
fill.get_commission() # float, fee charged
fill.get_total_cost() # float, notional ± commission

# Also available as read-only attributes
fill.symbol
fill.timestamp_ns
fill.order_id
fill.side
fill.quantity
fill.price
fill.commission
```

### on_rejected

`on_rejected` is called whenever a signal is blocked by the risk manager before an order is placed. The strategy is not notified of rejections by default - override this method if you need to react to them (e.g. log, reduce position targets, or halt trading).

```python
def on_rejected(self, symbol: str, reason: str):
    # symbol: the asset whose signal was rejected
    # reason: human-readable explanation from the risk manager
    print(f"Order rejected for {symbol}: {reason}")
```

Possible rejection reasons correspond to `RiskCheckResult` values: position limit breached, leverage limit breached, no capital available, max loss limit exceeded, or single-order notional limit breached.

### Generating signals

You do not place orders directly. You generate a signal and the engine converts it to an order using the configured position sizer and execution settings.

```python
self.generate_signal(
    symbol,            # str
    qc.SignalType.BUY, # or SELL, HOLD
    1.0,               # signal strength, 0.0–1.0, scales position size
    event.get_timestamp()
)
```

`SignalType.HOLD` is a no-op provided for explicitness in strategies that want to log a neutral state.

### Checking current position

```python
position = self.get_position(symbol)  # float, negative = short
has_pos  = self.has_position(symbol)  # bool, True if abs(position) > 0
```

### Accessing portfolio state

`PortfolioContext` is available inside `on_data` and `on_fill` via `self.get_portfolio()`. It returns `None` before the engine attaches it at `run()`, so guard accordingly:

```python
# Inside on_data or on_fill:
portfolio = self.get_portfolio()   # PortfolioContext, or None if called before run()
if portfolio:
    cash    = portfolio.get_cash()
    value   = portfolio.get_portfolio_value()
    pos     = portfolio.get_position('AAPL')
    weight  = portfolio.get_position_weight('AAPL')
```

See [PortfolioContext](#portfoliocontext) for the full API.

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

    def on_fill(self, fill):
        print(f"Filled: {fill.get_side()} {fill.get_quantity()} @ {fill.get_price():.2f}")

    def on_rejected(self, symbol, reason):
        print(f"Rejected: {symbol} - {reason}")
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

`run_backtest` returns a `BacktestResults` object directly. You can call `.compute()` on it immediately to calculate all performance metrics:

```python
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars},
    initial_capital=100_000.0,
).compute()

print(results)
print(results.metrics)
```

To filter weekends and exchange holidays before running, pass a `calendar` argument. The same calendar is applied to every symbol in `data`:

```python
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars_aapl, 'MSFT': bars_msft},
    initial_capital=100_000.0,
    calendar='NYSE',
)
```

See [Trading Calendar](#16-trading-calendar) for supported exchanges and behaviour.

### run_tick_backtest (convenience function)

```python
results = qc.run_tick_backtest(
    strategy=MyStrategy(),
    tick_data={'AAPL': ticks},
    initial_capital=100_000.0,
    mm_refresh_interval_ns=1_000_000_000,       # refresh MM quotes once per second
    equity_snapshot_interval_ns=60_000_000_000, # snapshot equity once per minute
)
```

The same strategy class works for bar and tick data. `event.close` is the tick price. Both `mm_refresh_interval_ns` and `equity_snapshot_interval_ns` have sensible defaults - see below for what they control.

### BacktestEngine (direct)

Use this when you need to configure execution, position sizing, or risk limits before running.

```python
engine = qc.BacktestEngine(100_000.0)

engine.add_data('AAPL', qc.load_csv_data('data/aapl.csv', 'AAPL'))
engine.set_strategy(MyStrategy())

final_value = engine.run()
```

`run()` returns the final portfolio value as a `float`. It resets all internal state first, so you can call it multiple times on the same engine (useful for parameter sweeps).

Passing zero or negative initial capital raises an exception.

### Fast data loading with numpy

`add_data` accepts either a `List[BarData]` or a `(N, 6)` float64 numpy array with columns `[timestamp_ns, open, high, low, close, volume]`. The numpy path uses a single boundary crossing instead of N individual pybind11 object constructions and is roughly 3–5x faster for large datasets:

```python
import numpy as np

arr = qc.ParquetDataLoader.load_numpy('data/aapl.parquet')  # (N, 6) float64
engine.add_data('AAPL', arr)
```

Note: `timestamp_ns` is stored as `float64` in the array. `float64` has a 53-bit mantissa so timestamps beyond year 2255 lose sub-microsecond precision - fine for current UNIX nanosecond timestamps.

### Tick data

Load tick data with `add_tick_data` instead of `add_data`. Bar and tick series can be mixed across symbols in the same engine.

```python
engine = qc.BacktestEngine(100_000.0)
engine.add_tick_data('AAPL', ticks)   # List[TickData]
engine.add_tick_data('AAPL', arr)     # (N, 4) float64 numpy array
```

The numpy path for tick data uses a single boundary crossing instead of N individual pybind11 object constructions and is roughly 3–5x faster for large datasets. Column order: `[timestamp_ns, price, quantity, side]` with side encoded as `0.0 = Buy, 1.0 = Sell`.

`add_data` and `add_tick_data` are mutually exclusive per symbol - calling one clears any previously registered data for that symbol.

### Market maker refresh interval

The engine runs a synthetic market maker that refreshes its quotes on every event by default. For dense tick data this dominates cost. Throttle it:

```python
engine.set_mm_refresh_interval(1_000_000_000)   # refresh at most once per second
engine.get_mm_refresh_interval()                 # int, nanoseconds (0 = every event)
```

A 1-second interval gives ~5x speedup on 1-second tick data. `run_tick_backtest()` sets this to 1 second by default.

### Equity snapshot interval

Controls how often the equity curve is snapshotted during a run:

```python
engine.set_equity_snapshot_interval(60_000_000_000)  # snapshot every minute
engine.get_equity_snapshot_interval()                 # int, nanoseconds (0 = every event)
```

In bar mode the snapshot happens once per bar regardless of this setting. In tick mode, setting this to 1 minute keeps the equity curve manageable without any meaningful performance impact - the snapshot itself is cheap. `run_tick_backtest()` sets this to 1 minute by default.

### Checking whether a symbol uses tick data

```python
engine.has_tick_data('AAPL')  # bool
```

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

# Retrieve the currently configured sizer
sizer = engine.get_position_sizer()
```

### Passing RiskLimits

```python
limits = qc.RiskLimits()
limits.max_position_pct = 0.20
limits.max_leverage     = 2.0
limits.max_loss_pct     = 0.15

engine.set_risk_limits(limits)
```

### Bars per year

The engine uses a `bars_per_year` value to annualize volatility internally (used by `VolatilityTargeting` and the rolling volatility estimate passed to position sizers). The default is `252` (daily bars). Override when using intraday or weekly bars:

```python
engine.set_bars_per_year(252)      # daily bars (default)
engine.set_bars_per_year(52)       # weekly bars
engine.set_bars_per_year(0)        # raises an exception
print(engine.get_bars_per_year())  # int
```

### Configuring the market maker

The engine runs a synthetic market maker to ensure strategies always have liquidity to trade against. The defaults work for most backtests; override them to simulate tighter or wider spreads:

```python
engine.configure_market_maker(
    levels=5,       # number of price levels per side
    spread=0.0001,  # base spread as a fraction of price (0.01%)
    depth=100_000   # base quantity at each level
)
```

### create_backtest

`create_backtest` returns a configured engine without running it:

```python
engine = qc.create_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars},
    initial_capital=100_000.0,
)
engine.set_position_sizer(qc.RiskBased(0.01))
engine.run()
```

### Error conditions

The engine raises an exception if:

- `run()` is called without a strategy set
- `run()` is called without any data loaded
- `set_strategy(None)` or `set_position_sizer(None)` is called
- An empty bar or tick series is passed to `add_data` or `add_tick_data`
- Initial capital is zero or negative

---

## 4. Reading Results

### BacktestResults object

`run_backtest` and `run_tick_backtest` both return a `BacktestResults` object:

```python
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars},
    initial_capital=100_000.0,
)

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
results.strategy_name           # str
results.initial_capital         # float
results.final_value             # float
results.total_pnl               # float
results.total_fees              # float
results.net_pnl                 # float (property: final_value - initial_capital)
results.return_pct              # float
results.equity_curve            # List[float]
results.timestamps              # List[int], nanoseconds
results.trade_pnls              # List[float], per-trade PnL for all closed trades
results.benchmark_equity_curve  # List[float] or None, set by run_backtest benchmark args
```

### Computing metrics

Call `.compute()` to calculate all performance metrics and cache them on the object. Returns `self` for chaining:

```python
results.compute()
print(results.metrics)
```

The `metrics` property raises `RuntimeError` if accessed before calling `compute()`.

When a benchmark equity curve is present, `.compute()` also calculates benchmark-relative metrics automatically. See [Benchmark Comparison](#benchmark-comparison) below.

### Reading from the engine directly

If you used `BacktestEngine` directly:

```python
engine.get_total_pnl()         # float
engine.get_total_fees()        # float
engine.get_trade_pnls()        # List[float], PnL of every closed trade
engine.get_equity_curve()      # List[float]
engine.get_timestamps()        # List[int]
engine.get_portfolio_context() # PortfolioContext
```

---

## 5. Execution Simulation

### ExecutionConfig fields

```python
config = qc.ExecutionConfig()

config.maker_fee    = 0.001      # Fraction of notional, applied when order adds liquidity
config.taker_fee    = 0.002      # Fraction of notional, applied when order removes liquidity
config.slippage_pct = 0.0005     # Fraction of price, applied in direction of trade
config.latency_ns   = 1_000_000  # Nanoseconds delay between signal and fill
```

Fees and slippage are applied per fill. For a 100-share buy at $100 with `taker_fee=0.002` and `slippage_pct=0.0005`:

- Fill price: $100.05 (slippage pushes up)
- Fee: $100.05 × 100 × 0.002 = $20.01

### Order types

```python
qc.OrderType.GOOD_TILL_CANCEL    # Rests on the book until filled or canceled (default)
qc.OrderType.MARKET              # Fills at best available price, no resting
qc.OrderType.IMMEDIATE_OR_CANCEL # Fills what it can immediately, cancels the rest
qc.OrderType.FILL_OR_KILL        # Fills entirely or not at all
qc.OrderType.GOOD_FOR_DAY        # Canceled at end of session if unfilled
```

### ExecutionEngine (per-symbol access)

```python
ee = engine.get_execution_engine('AAPL')
# Returns None if the symbol was never loaded via add_data or add_tick_data

ee.get_position()          # float, shares held (negative = short)
ee.get_average_price()     # float, volume-weighted average entry price
ee.get_realized_pnl()      # float
ee.get_unrealized_pnl()    # float, mark-to-market on the open position
ee.get_total_pnl()         # float, realized + unrealized
ee.get_total_fees()        # float
ee.get_best_bid()          # float, best resting bid price in dollars
ee.get_best_ask()          # float, best resting ask price in dollars
ee.get_mid_price()         # float | None, (best_bid + best_ask) / 2
ee.get_closed_trade_pnls() # List[float], PnL of each completed round-trip
ee.reset()                 # clear all state (called automatically at run())
```

All fields return zero (or `None` for `get_mid_price()`) before `run()` is called.

---

## 6. Position Sizing

Position sizers control how many shares are ordered when a signal is generated. All sizers respect constraints set via `set_max_position_size`, `set_min_position_size`, and `set_max_leverage`.

### Default leverage behaviour

Each sizer has an internal `max_leverage_` cap that limits the notional value of any single order to `capital * max_leverage_`. The default differs by sizer:

- `FixedPercentage`, `VolatilityTargeting`, `FixedShares`: cap is **disabled by default** (`max_leverage_ = 0`). The engine's `RiskLimits` act as the sole ceiling. Call `set_max_leverage()` to impose an additional sizer-level cap.
- `RiskBased`, `KellyCriterion`, `EqualWeight`: cap defaults to **1x**. Call `set_max_leverage()` to allow leverage above 1x.

### FixedPercentage

Allocates a fixed fraction of current capital per position. Values above `1.0` imply leverage (e.g. `2.0` = 200% of capital per position) and are valid - the sizer does not cap leverage internally, so `RiskLimits.max_leverage` on the engine acts as the ceiling.

```python
sizer = qc.FixedPercentage(0.10)   # 10% per trade, no leverage
sizer = qc.FixedPercentage(1.0)    # 100% per trade (1x, single asset)
sizer = qc.FixedPercentage(2.0)    # 200% per trade - requires matching RiskLimits
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

`set_volatility_params` controls the `stop_loss_distance` injected into every `PositionSizingContext`. Without it the stop distance falls back to the engine's internal default and `RiskBased` will produce zero sizes. In practice, set `stop_distance` to a fixed ATR-based estimate before running.

### KellyCriterion

Sizes using the Kelly formula: `f* = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win`.

```python
sizer = qc.KellyCriterion(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
```

Kelly sizing can produce large positions. The sizer defaults to a 1x leverage cap - call `set_max_leverage` to allow leverage above 1x, or use a fractional Kelly to reduce size:

```python
# Allow up to 2x leverage
sizer.set_max_leverage(2.0)

# Or use fractional Kelly to reduce size at 1x
sizer = qc.KellyCriterion(win_rate=0.55, avg_win=0.02, avg_loss=0.01, fraction=0.25)
```

### EqualWeight

Divides capital equally across `n` positions.

```python
sizer = qc.EqualWeight(num_positions=5)  # 20% per position
```

### VolatilityTargeting

Scales leverage to hit a target annualized portfolio volatility. The sizer is uncapped by default - it will scale above 1x if the portfolio volatility is low relative to the target. Use `set_max_leverage()` to impose a ceiling, or rely on `RiskLimits.max_leverage` on the engine.

```python
sizer = qc.VolatilityTargeting(target_volatility=0.15)  # Target 15% annualized vol

# Cap leverage at 2x
sizer.set_max_leverage(2.0)
```

The engine provides a running volatility estimate after sufficient history is available; the sizer returns zero until then. Set `bars_per_year` on the engine to ensure correct annualization (see [Bars per year](#bars-per-year)).

### FixedShares

Always orders the same number of shares regardless of capital or price. Uncapped by default.

```python
sizer = qc.FixedShares(100)  # Always order 100 shares
```

### Applying constraints

All sizers support the same constraint API:

```python
sizer.set_max_position_size(500)   # Max 500 shares per position
sizer.set_min_position_size(10)    # Orders below this size are dropped entirely
sizer.set_max_leverage(2.0)        # Notional cannot exceed 2x capital (0 = disabled)
```

### Retrieving the active sizer

```python
sizer = engine.get_position_sizer()
```

### PositionSizingContext

`PositionSizingContext` is the data object the engine builds and passes to the sizer on every signal. You can construct one manually to test a sizer in isolation:

```python
ctx = qc.PositionSizingContext(
    signal_strength=1.0,
    current_capital=100_000.0,
    current_price=150.0,
    current_position=0.0,
    portfolio_volatility=0.02,
    stop_loss_distance=0.05
)
shares = sizer.calculate_size(ctx)
```

All fields are read-write.

### Python-side utilities: PositionCalculator

`PositionCalculator` is a standalone Python class for pre-trade sizing calculations outside the engine (useful in research notebooks). Methods return a `PositionSizeResult` with `.quantity`, `.notional_value`, `.percent_of_capital`, and `.reasoning` - not a plain float:

```python
from quantcore import PositionCalculator

calc = PositionCalculator(capital=100_000.0, max_position_pct=0.2)

result = calc.fixed_percentage(price=150.0, percentage=0.10)
result = calc.risk_based(price=150.0, stop_loss_price=145.0, risk_per_trade=0.01)
result = calc.kelly_criterion(price=150.0, win_rate=0.55, avg_win=0.02, avg_loss=0.01)
result = calc.equal_weight(price=150.0, num_positions=5)
result = calc.volatility_adjusted(price=150.0, volatility=0.20, target_volatility=0.15)
result = calc.leveraged(price=150.0, leverage=2.0, base_percentage=0.10)

print(result.quantity)            # shares to order
print(result.notional_value)      # $ exposure
print(result.percent_of_capital)  # fraction of capital used
print(result.reasoning)           # human-readable explanation
```

### PortfolioPositionSizer

`PortfolioPositionSizer` enforces portfolio-level exposure limits across multiple open positions. It is a research utility, not connected to the engine's internal sizing:

```python
from quantcore import PortfolioPositionSizer

sizer = PortfolioPositionSizer(
    capital=100_000.0,
    max_total_exposure=1.0,    # max gross notional / capital
    max_single_position=0.20,  # max single-asset notional / capital
)

# Update tracked notional after a fill
sizer.update_position('AAPL', notional_value=15_000.0)
sizer.update_position('MSFT', notional_value=12_000.0)
sizer.update_position('MSFT', notional_value=0.0)  # clear a position

print(sizer.get_total_exposure())    # float, current gross exposure as fraction of capital
print(sizer.get_available_capital()) # float, remaining notional budget in dollars

# Check before submitting an order
allowed, reason = sizer.can_add_position('GOOGL', notional_value=20_000.0)

# Size a new position respecting all constraints; returns None if no room
result = sizer.size_new_position('GOOGL', price=140.0, desired_percentage=0.10)
if result:
    print(result.quantity)
```

---

## 7. Risk Management

### RiskLimits fields

```python
limits = qc.RiskLimits()

limits.enabled            = True   # Set False to disable all checks
limits.max_position_pct   = 0.20   # Max position notional / capital (per symbol).
                                   # Values above 1.0 allow leverage on a single asset,
                                   # e.g. 2.0 means one asset can be sized up to 2x capital.
limits.max_leverage       = 2.0    # Max total notional / capital across all positions
limits.max_loss_pct       = 0.50   # Halt if drawdown from initial capital exceeds this
limits.max_order_value    = 50_000 # Max notional per single order (0 = no limit)
```

`limits.validate()` checks that all field values are within acceptable ranges and raises `ValueError` if not. It is called automatically when limits are attached to the engine, but you can call it manually to catch bad values early.

### Attaching limits to the engine

```python
engine.set_risk_limits(limits)
retrieved = engine.get_risk_limits()

print(retrieved.max_leverage)      # float
print(retrieved.max_loss_pct)      # float
print(retrieved.max_position_pct)  # float
```

### RiskManager (direct access)

```python
risk_mgr = engine.get_risk_manager()

# Check whether a hypothetical order would be approved
response = risk_mgr.check_order('AAPL', qc.Side.BUY, quantity=100, price=175.0)

if response.is_approved():
    print("Order approved")
else:
    print(f"Rejected: {response.result}, {response.reason}")
```

`RiskCheckResult` enum values:

```python
qc.RiskCheckResult.APPROVED
qc.RiskCheckResult.REJECTED_POSITION_LIMIT   # per-symbol notional limit breached
qc.RiskCheckResult.REJECTED_LEVERAGE_LIMIT   # total portfolio leverage limit breached
qc.RiskCheckResult.REJECTED_CAPITAL_LIMIT    # no capital available
qc.RiskCheckResult.REJECTED_LOSS_LIMIT       # max drawdown limit exceeded
qc.RiskCheckResult.REJECTED_ORDER_SIZE       # single-order notional limit breached
```

The full `RiskManager` API:

```python
risk_mgr.set_capital(initial, current)           # update the capital figures used for checks
risk_mgr.set_position(symbol, quantity, price)   # record a position with its last-known price
risk_mgr.get_position(symbol)                    # float, tracked quantity
risk_mgr.set_limits(limits)                      # replace the active RiskLimits
risk_mgr.get_limits()                            # RiskLimits
risk_mgr.update_position(symbol, side, quantity) # update quantity only (no price update)
risk_mgr.get_all_positions()                     # dict[str, float]
risk_mgr.calculate_total_exposure()              # float, sum of abs notional across all positions
risk_mgr.reset()                                 # clear all state
```

### Behavior on rejection

A rejected order does not execute. The strategy's `on_rejected(symbol, reason)` method is called so you can react - reduce targets, log, or halt. If you do not override `on_rejected` the rejection is silently discarded.

---

## 8. Analytics

All analytics functions live in `quantcore.analytics` and operate on NumPy arrays.

### Setup

```python
import numpy as np
from quantcore.analytics import (
    calculate_returns,
    calculate_all_metrics,
    calculate_benchmark_metrics,
    infer_periods_per_year,
    rolling_sharpe,
    rolling_volatility,
    monthly_returns,
    underwater_plot_data,
)

equity     = np.array(engine.get_equity_curve())
timestamps = np.array(engine.get_timestamps())
returns    = calculate_returns(equity)
```

### calculate_returns

Converts an equity curve to period returns:

```python
returns = calculate_returns(equity)
# np.ndarray, length len(equity) - 1
# returns[i] = (equity[i+1] - equity[i]) / equity[i]
```

### infer_periods_per_year

Estimates the number of equity snapshots per year from a timestamp array.
Measures the median inter-snapshot interval and divides one calendar year by it.
Useful when you need the correct annualisation factor for a custom calculation:

```python
ppy = infer_periods_per_year(timestamps)
# ~525_960 for 1-minute crypto snapshots (365.25 * 1440)
# ~362_880 for 1-minute equity snapshots (252 * 1440)
# ~365     for daily bars
# 252      fallback when fewer than 2 timestamps are provided
```

### calculate_all_metrics

Returns a `PerformanceMetrics` dataclass. Printing it gives a formatted summary:

```python
# Recommended: pass timestamps so that max_drawdown_duration is reported in
# calendar days and periods_per_year is inferred automatically.
metrics = calculate_all_metrics(equity, trade_pnls=trade_pnls, timestamps=timestamps)
print(metrics)
# Total Return:          1.32%
# Annualized Return:     5.89%
# Sharpe Ratio:          4.39
# Sortino Ratio:         0.70
# Max Drawdown:         -0.42%
# Max DD Duration:      35 days
# ...
```

**Signature:**

```python
calculate_all_metrics(
    equity_curve,
    trade_pnls=None,       # List[float], per-trade PnL for trade-level metrics
    timestamps=None,       # np.ndarray of int64 nanosecond timestamps
    risk_free_rate=0.0,    # annual rate, e.g. 0.02 = 2%
    periods_per_year=None, # int or None; when None, auto-inferred from timestamps
)                          # falls back to 252 when no timestamps provided
```

**`periods_per_year` auto-inference:** when `None` (the default) and `timestamps` are
provided, `infer_periods_per_year` is called automatically. This means annualised
return, Sharpe, Sortino, volatility, and Calmar are all correctly scaled regardless
of whether you are running on daily bars, minute-level tick data, or anything else.
Pass an explicit integer to override (e.g. `periods_per_year=252` to force the
daily-bars convention regardless of actual data density).

**`max_drawdown_duration`:** when `timestamps` are provided this field is the
calendar-day span from the equity peak to recovery (or end of data). When
`timestamps` are not provided it is `-1`, and `PerformanceMetrics.__str__` prints
`"n/a (no timestamps)"` rather than a misleading snapshot count.

### Benchmark Comparison

`calculate_benchmark_metrics` computes benchmark-relative performance metrics from
two period-return arrays. Pass `timestamps` to let it infer the correct annualisation
factor automatically — this is especially important for intraday and tick data where
the default of 252 would produce wildly wrong annualised figures.

```python
from quantcore.analytics import calculate_benchmark_metrics

bm_metrics = calculate_benchmark_metrics(
    strategy_returns,   # np.ndarray, output of calculate_returns on strategy equity
    benchmark_returns,  # np.ndarray, output of calculate_returns on benchmark equity
    timestamps=timestamps,  # recommended — auto-infers periods_per_year
)
print(bm_metrics)
# ============================================================
#   Benchmark Comparison
# ============================================================
#   Benchmark Total Return:        3.74%
#   Benchmark Annual Return:      52.34%
#   Active Return (ann.):        -54.30%
#
#   Alpha (ann.):                 -2.09%
#   Beta:                          0.00
#   Correlation:                   0.13
#   R-Squared:                     0.02
#
#   Tracking Error (ann.):        50.75%
#   Information Ratio:            -1.07
#
#   Up Capture:                    0.00%
#   Down Capture:                  2.13%
# ============================================================
```

**Signature:**

```python
calculate_benchmark_metrics(
    strategy_returns,             # np.ndarray of period returns
    benchmark_returns,            # np.ndarray of period returns
    periods_per_year=None,        # int or None; when None, auto-inferred from timestamps
    timestamps=None,              # np.ndarray of int64 nanosecond timestamps
)
```

Both return arrays must be the same type as `calculate_returns` output — period
returns, not equity curves. If the two backtests produced different-length equity
curves, align them on a common timestamp index before calling this function; the
function clips to the shorter of the two arrays as a fallback but alignment is
more correct.

**`BenchmarkMetrics` fields** (all percentages except dimensionless ratios):

```python
bm_metrics.benchmark_total_return       # float, total benchmark return (%)
bm_metrics.benchmark_annualized_return  # float, annualised benchmark return (%)
bm_metrics.active_return                # float, annualised strategy minus benchmark (%)
bm_metrics.alpha                        # float, CAPM alpha annualised (%)
bm_metrics.beta                         # float, OLS slope of strategy on benchmark returns
bm_metrics.correlation                  # float, Pearson correlation [-1, 1]
bm_metrics.r_squared                    # float, fraction of variance explained [0, 1]
bm_metrics.tracking_error               # float, annualised std of active returns (%)
bm_metrics.information_ratio            # float, active_return / tracking_error
bm_metrics.up_capture                   # float, strategy / benchmark return in up periods (%)
bm_metrics.down_capture                 # float, strategy / benchmark return in down periods (%)
```

**Annualisation and `periods_per_year`:** passing `timestamps` is strongly recommended.
Without it the function defaults to 252 (daily bars), which is correct for daily equity
data but wrong for everything else. On 1-minute crypto bars with a 30-day backtest,
using 252 compresses 30 days of data into ~182 implied years and collapses all annualised
figures toward zero. With the correct ~525,960 periods-per-year, the same 3.74% monthly
BTC gain annualises correctly to ~52%.

**Using benchmark metrics with `BacktestResults`:** when `benchmark_equity_curve` is set
on a `BacktestResults` object (either from `run_backtest` with benchmark arguments or by
assigning it manually), `.compute()` calculates benchmark metrics automatically and stores
them in `results.benchmark_metrics`. The timestamps from the strategy run are used for
`periods_per_year` inference, so no manual override is needed:

```python
# Via run_backtest — benchmark runs automatically
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars},
    initial_capital=100_000.0,
    benchmark_strategy=qc.BuyAndHold(),
).compute()

print(results)           # summary includes active return, alpha, beta, IR, capture ratios
print(results.metrics)          # full PerformanceMetrics
print(results.benchmark_metrics)  # BenchmarkMetrics, or None if no benchmark

# Or assign manually after the fact
results.benchmark_equity_curve = np.array(bm_engine.get_equity_curve())
results.compute()
print(results.benchmark_metrics)
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

total_return = calculate_total_return(equity)                              # float, %
ann_ret      = calculate_annualized_return(equity, periods_per_year=252)  # float, %
sharpe       = calculate_sharpe_ratio(returns)                            # float
sortino      = calculate_sortino_ratio(returns)                           # float
vol          = calculate_volatility(returns, periods_per_year=252)        # float, %

# timestamps is optional; when provided, duration is calendar days.
# When omitted, duration is -1 (indeterminate).
max_dd, dur  = calculate_max_drawdown(equity, timestamps=timestamps)      # (float %, int)

calmar       = calculate_calmar_ratio(ann_ret, max_dd)                    # float

trade_stats  = analyze_trades([100.0, -50.0, 200.0, -30.0, 80.0])
# dict with keys: total_trades, win_rate, profit_factor,
#                 avg_win, avg_loss, largest_win, largest_loss
```

### Rolling metrics

```python
# periods_per_year defaults to 252.
# For tick/minute data pass the correct value or use infer_periods_per_year:
ppy = infer_periods_per_year(timestamps)

roll_sharpe = rolling_sharpe(returns, window=60, periods_per_year=ppy)
# np.ndarray of length len(returns) - window + 1
# NaN where the rolling window is entirely flat (std == 0) — renders as a
# gap in charts rather than a misleading spike.

roll_vol = rolling_volatility(returns, window=60, periods_per_year=ppy)
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
# np.ndarray, same length as equity
# values are drawdown % from running peak, always <= 0
```

---

## 9. Visualizations

All plotting functions live in `quantcore.plotting` and return `matplotlib.figure.Figure`.

```python
from quantcore.plotting import (
    plot_equity_curve,
    plot_benchmark_comparison,
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
    timestamps=timestamps,          # optional, enables date axis
    title="My Strategy",
    figsize=(14, 7),
    show_drawdown=True,             # shade strategy drawdown periods in red
    benchmark_equity=bm_equity,     # optional, overlays benchmark line
    benchmark_label="Buy & Hold",   # legend label for the benchmark
)
fig.show()
```

When `benchmark_equity` is provided both curves are normalised to 100 at the start so
different initial capitals do not distort the overlay. The y-axis label changes to
"Normalised Value (100 = start)" accordingly.

### plot_benchmark_comparison

Dedicated three-panel benchmark comparison chart. Shows more detail than the overlay
in `plot_equity_curve` or `plot_full_tearsheet`.

```python
fig = plot_benchmark_comparison(
    equity_curve=equity,
    benchmark_equity=bm_equity,
    timestamps=timestamps,
    strategy_label="My Strategy",
    benchmark_label="Buy & Hold",
    title="Strategy vs Benchmark",
    figsize=(14, 10),
    periods_per_year=None,    # auto-inferred from timestamps when None
)
fig.show()
```

The three panels are: normalised equity curves overlaid (both start at 100), rolling
active return with green/red fill for out/underperformance periods, and side-by-side
underwater drawdown comparison. The rolling window is sized to approximately one
calendar quarter.

### plot_underwater

Dedicated drawdown chart:

```python
fig = plot_underwater(equity, timestamps=timestamps)
```

### plot_returns_distribution

Histogram of returns with Q-Q plot. Zero-return periods (flat equity between
trades) are automatically excluded — common with tick/minute data where most
snapshots show no change because there is no open position. The excluded
fraction is noted in an annotation when it exceeds 5%:

```python
fig = plot_returns_distribution(returns, title="Return Distribution")
```

### plot_rolling_metrics

Two-panel chart: rolling Sharpe (top) and rolling volatility (bottom).

The `window` parameter is in **periods**, not calendar days. When `window=0`
(the default), it is inferred automatically from `timestamps` to span
approximately 14 calendar days, which stays meaningful across daily, minute,
and tick data. `periods_per_year` is also inferred from `timestamps` when
provided, so the Sharpe and volatility scales are correctly annualised:

```python
fig = plot_rolling_metrics(
    returns,
    timestamps=timestamps,  # required for auto window + correct annualisation
    window=0,               # 0 = auto-infer from timestamps (~14 calendar days)
    title="Rolling Metrics"
)
```

NaN values from flat-equity windows render as gaps in the line rather than
spikes.

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

All charts in one figure (equity curve, drawdown, returns distribution,
rolling Sharpe, rolling volatility). Rolling window and annualisation factor
are both auto-inferred from `timestamps`. When `benchmark_equity` is supplied
the equity curve panel overlays both curves normalised to 100:

```python
import matplotlib.pyplot as plt

fig = plot_full_tearsheet(
    equity, returns,
    timestamps=timestamps,
    title="My Strategy",
    benchmark_equity=bm_equity,     # optional
    benchmark_label="Buy & Hold",   # legend label for the benchmark
)
plt.show(block=True)
```

### save_all_plots

Saves individual plot files to a directory. When `benchmark_equity` is provided
an additional `{strategy_name}_benchmark.png` file is written using
`plot_benchmark_comparison`:

```python
save_all_plots(
    equity_curve=equity,
    returns=returns,
    timestamps=timestamps,
    output_dir='plots',
    strategy_name='mean_reversion',
    benchmark_equity=bm_equity,     # optional
    benchmark_label="Buy & Hold",   # optional
)
# Writes: mean_reversion_equity.png, _underwater.png, _returns_dist.png,
#         _rolling.png, _tearsheet.png, _benchmark.png (when benchmark provided)
```

---

## 10. Built-in Strategies

These are available as C++ classes exposed via pybind11.

### BuyAndHold

Buys on the first bar or tick per symbol, holds until the end. Used as a benchmark.

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

# After run(), query how many signals were generated
count = strategy.get_signal_count()  # int, >= 0
```

### PairsTrading

Statistical arbitrage on two correlated assets. Monitors the log spread `log(price1 / price2)`, buys the underperformer and sells the outperformer when the spread diverges beyond a z-score threshold, exits when it reverts.

```python
strategy = qc.PairsTrading(
    symbol1='AAPL',
    symbol2='MSFT',
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5
)

# Check whether the strategy currently has an open spread position
currently_trading = strategy.in_trade()  # bool
```

Requires both symbols to be loaded via `add_data` or `add_tick_data`.

---

## 11. Multi-Asset Backtests

Pass multiple symbols via `add_data` or `add_tick_data`. The engine interleaves their events in timestamp order automatically.

```python
engine = qc.BacktestEngine(100_000.0)
engine.add_data('AAPL',  qc.load_csv_data('data/aapl.csv',  'AAPL'))
engine.add_data('GOOGL', qc.load_csv_data('data/googl.csv', 'GOOGL'))
engine.add_data('MSFT',  qc.load_csv_data('data/msft.csv',  'MSFT'))
engine.set_strategy(MyMultiAssetStrategy())
engine.run()
```

Inside the strategy, `event.get_symbol()` tells you which asset triggered the call:

```python
class MultiAssetStrategy(qc.Strategy):
    def on_data(self, event):
        if event.get_symbol() == 'AAPL':
            pass  # AAPL-specific logic
        elif event.get_symbol() == 'GOOGL':
            pass  # GOOGL-specific logic
```

Capital is shared across all positions. The position sizer and risk limits apply per-symbol.

### PortfolioContext

`engine.get_portfolio_context()` returns the full portfolio state after `run()`. The same object is accessible inside strategies via `self.get_portfolio()`:

```python
portfolio = engine.get_portfolio_context()

portfolio.get_initial_capital()        # float, capital passed to BacktestEngine
portfolio.get_cash()                   # float, uninvested cash
portfolio.get_portfolio_value()        # float, cash + mark-to-market value of all positions
portfolio.get_total_position_value()   # float, gross market value of all open positions
portfolio.get_leverage()               # float, total notional / portfolio value
portfolio.num_positions()              # int, number of symbols with a non-zero position
portfolio.has_position('AAPL')         # bool
portfolio.get_position('AAPL')         # float, shares held (negative = short)
portfolio.get_position_value('AAPL')   # float, shares × last price
portfolio.get_position_weight('AAPL')  # float, position value / portfolio value
portfolio.get_price('AAPL')            # float, last known price
portfolio.get_all_positions()          # dict[str, float], all non-zero positions
portfolio.get_all_prices()             # dict[str, float], last known price per symbol
```

---

## 12. Parameter Sweeps and Optimization

### GridSearchOptimizer (recommended)

`GridSearchOptimizer` is the high-level API for parameter optimization. It handles the grid, runs each combination, and returns results sorted by the chosen metric:

```python
from quantcore.walk_forward import GridSearchOptimizer

bars = qc.load_csv_data('data/aapl.csv', 'AAPL')
data = {'AAPL': bars}

param_grid = {
    'fast_period': [10, 20, 50],
    'slow_period': [100, 150, 200],
}

opt = GridSearchOptimizer(
    strategy_factory=qc.SMACrossover,  # must be a class or module-level function, not a lambda
    param_grid=param_grid,
    metric='sharpe_ratio',             # 'sharpe_ratio', 'total_return', 'max_drawdown', 'num_trades', or 'final_value'
    n_jobs=1,                          # set to -1 for all cores (see parallelism note below)
)

results = opt.optimize(data, initial_capital=100_000.0, verbose=True)

best = results[0]
print(f"Best params: {best.params}")
print(f"Sharpe: {best.sharpe_ratio:.2f}")
print(f"Return: {best.total_return_pct:.2f}%")
print(f"Max DD: {best.max_drawdown_pct:.2f}%")

# Get top 10
top10 = opt.get_top_n(10)

# Export to DataFrame
df = opt.get_results_dataframe()
print(df[['fast_period', 'slow_period', 'sharpe_ratio', 'total_return_pct']].head())
```

**`OptimizationResult` fields:**

```python
result.params           # dict, e.g. {'fast_period': 20, 'slow_period': 100}
result.sharpe_ratio     # float
result.total_return     # float, DECIMAL format (0.105 = 10.5%)
result.total_return_pct # float, percentage format (10.5)    - use this for display
result.max_drawdown     # float, DECIMAL format (-0.05 = -5%)
result.max_drawdown_pct # float, percentage format (-5.0)    - use this for display
result.num_trades       # int
result.final_value      # float
```

> **Note on decimal vs percentage format:** `total_return` and `max_drawdown` on `OptimizationResult` use **decimal format** (0.105 = 10.5%), unlike `PerformanceMetrics` from `calculate_all_metrics` which uses percentage format (10.5). Use the `_pct` properties for display, or the helpers from `walk_forward` if you need to convert programmatically:
>
> ```python
> from quantcore.walk_forward import pct_to_decimal, decimal_to_pct
>
> decimal_to_pct(0.105)   # → 10.5
> pct_to_decimal(10.5)    # → 0.105
> ```

### Parallelism

`n_jobs` controls worker processes. `n_jobs=-1` uses all available cores. Key constraints:

- `strategy_factory` must be **picklable** - pass a class or module-level function, not a lambda or closure. Lambdas will raise `PicklingError` at runtime.
- On Windows, guard your entry point with `if __name__ == '__main__':`.
- Process spawn overhead on Windows is ~500 ms per worker. Parallelism only pays off when each combo takes substantially longer than that (e.g. 100-year backtests at ~240 ms each give a clear speedup; 5-year backtests at ~11 ms each do not). On Linux (`fork`-based), the break-even point is much lower.

### Manual grid search

For cases where you need full control:

```python
import itertools
import numpy as np
from quantcore.analytics import calculate_sharpe_ratio, calculate_returns

fast_periods = [10, 20, 50]
slow_periods = [100, 150, 200]
results_grid = {}

for fast, slow in itertools.product(fast_periods, slow_periods):
    if fast >= slow:
        continue

    engine = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', bars)
    engine.set_strategy(qc.SMACrossover(fast_period=fast, slow_period=slow))
    engine.run()

    equity  = np.array(engine.get_equity_curve())
    returns = calculate_returns(equity)
    results_grid[(fast, slow)] = calculate_sharpe_ratio(returns)

best = max(results_grid, key=results_grid.get)
print(f"Best: SMA({best[0]}/{best[1]}) Sharpe={results_grid[best]:.2f}")
```

The engine is reentrant - `run()` resets all internal state before each run, so the same engine instance can be reused across multiple calls and will always produce identical results:

```python
engine = qc.BacktestEngine(100_000.0)
engine.add_data('AAPL', bars)
engine.set_strategy(qc.SMACrossover(20, 50))

fv1 = engine.run()
fv2 = engine.run()  # identical to fv1
```

### Parallel sweep with multiprocessing

```python
from multiprocessing import Pool

def run_single(params):
    fast, slow = params
    bars   = qc.load_csv_data('data/aapl.csv', 'AAPL')
    engine = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', bars)
    engine.set_strategy(qc.SMACrossover(fast_period=fast, slow_period=slow))
    engine.run()
    equity  = np.array(engine.get_equity_curve())
    returns = calculate_returns(equity)
    return (fast, slow, calculate_sharpe_ratio(returns))

if __name__ == '__main__':    # required on Windows
    param_grid = [(f, s) for f in [10, 20, 50] for s in [100, 150, 200] if f < s]
    with Pool() as pool:
        results = pool.map(run_single, param_grid)
    for fast, slow, sharpe in sorted(results, key=lambda x: -x[2]):
        print(f"SMA({fast}/{slow}): {sharpe:.2f}")
```

---

## 13. Walk-Forward Analysis

Walk-forward analysis validates a strategy by repeatedly optimizing on an in-sample window and evaluating the best parameters on the following out-of-sample window. This guards against overfitting to a single historical period.

`WalkForwardAnalyzer` supports both single-asset and multi-asset data.

### WalkForwardAnalyzer

```python
from quantcore.walk_forward import WalkForwardAnalyzer
 
bars_aapl = qc.load_csv_data('data/aapl.csv', 'AAPL')
bars_msft = qc.load_csv_data('data/msft.csv', 'MSFT')
data = {'AAPL': bars_aapl, 'MSFT': bars_msft}
 
param_grid = {
    'fast_period': [10, 20, 50],
    'slow_period': [50, 100, 200],
}
 
wfa = WalkForwardAnalyzer(
    strategy_factory=qc.SMACrossover,
    param_grid=param_grid,
    train_size=252,          # bars in each in-sample window
    test_size=63,            # bars in each out-of-sample window (also the step size)
    metric='sharpe_ratio',
    n_jobs=1,
)
 
result = wfa.analyze(data, initial_capital=100_000.0, verbose=True)
print(result.summary())
```

**`WalkForwardResult` fields:**

```python
result.in_sample_results        # List[OptimizationResult], best IS result per window
result.out_of_sample_results    # List[dict], OOS metrics per window
result.best_params_per_window   # List[dict], winning params per window
result.combined_equity_curve    # np.ndarray, chained OOS equity curve across all windows
result.overall_metrics          # dict: sharpe_ratio, total_return, max_drawdown, num_windows
```

OOS metrics in `out_of_sample_results` use **decimal format** for `total_return` and `max_drawdown`, consistent with `OptimizationResult`.

The combined equity curve is continuous: each OOS segment is rescaled to begin from the end value of the previous segment, giving a smooth compounded curve across all windows.

### Window construction for multi-asset data

The first symbol in the data dict is used as the reference timeline. `train_size` and `test_size` are measured in bars of that reference series. Bars for all other symbols are selected by timestamp inclusion within `[window_start, window_end)`, so minor gaps in a secondary symbol do not corrupt the windows of the reference symbol.

A window is skipped if any symbol ends up with zero bars in either its training or test period within that timestamp range. When all symbols are perfectly aligned (typical for daily data from a common source) this produces the same window count as a single-asset run.

### Plotting parameter stability

```python
fig = wfa.plot_stability(result)
fig.show()
```

This produces one subplot per parameter showing how the optimal value shifts across windows. Stable parameters are a sign of a robust strategy; highly variable ones may indicate overfitting.

### Manual walk-forward

```python
import numpy as np
from quantcore.analytics import calculate_sharpe_ratio, calculate_returns
 
all_bars   = qc.load_csv_data('data/aapl.csv', 'AAPL')
n_bars     = len(all_bars)
is_window  = 252
oos_window = 63
 
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
        eq = np.array(engine.get_equity_curve())
        sr = calculate_sharpe_ratio(calculate_returns(eq))
        if sr > best_sharpe:
            best_sharpe, best_params = sr, (fast, slow)
 
    # Out-of-sample evaluation
    engine = qc.BacktestEngine(100_000.0)
    engine.add_data('AAPL', oos_bars)
    engine.set_strategy(qc.SMACrossover(*best_params))
    engine.run()
    eq     = np.array(engine.get_equity_curve())
    oos_sr = calculate_sharpe_ratio(calculate_returns(eq))
    oos_sharpes.append(oos_sr)
 
    print(f"Period {start}–{start+is_window+oos_window}: "
          f"best params={best_params}, OOS Sharpe={oos_sr:.2f}")
 
print(f"\nMean OOS Sharpe: {np.mean(oos_sharpes):.2f}")
```
 
---

## 14. Monte Carlo Validation

`monte_carlo_validation` stress-tests a strategy by running it on many resampled versions of the price series and returning the distribution of outcomes. This measures how sensitive the strategy's performance is to the specific sequence of returns in the historical data.

`monte_carlo_validation` supports both single-asset and multi-asset data.

```python
from quantcore.walk_forward import monte_carlo_validation
 
bars_aapl = qc.load_csv_data('data/aapl.csv', 'AAPL')
bars_msft = qc.load_csv_data('data/msft.csv', 'MSFT')
data = {'AAPL': bars_aapl, 'MSFT': bars_msft}
 
mc_results = monte_carlo_validation(
    strategy_factory=qc.SMACrossover,
    params={'fast_period': 20, 'slow_period': 100},
    data=data,
    n_simulations=1000,
    initial_capital=100_000.0,
    method='bootstrap',   # 'bootstrap' (resample bars with replacement) or 'shuffle' (randomly reorder bars)
    n_jobs=1,
)
 
import numpy as np
sharpes   = mc_results['sharpe_ratios']  # np.ndarray
returns   = mc_results['returns']        # np.ndarray, DECIMAL format (0.105 = 10.5%)
drawdowns = mc_results['drawdowns']      # np.ndarray, DECIMAL format (-0.05 = -5%)
 
print(f"Median Sharpe:   {np.median(sharpes):.2f}")
print(f"5th pct Sharpe:  {np.percentile(sharpes, 5):.2f}")
print(f"Median Return:   {np.median(returns):.2%}")
print(f"Worst Drawdown:  {np.min(drawdowns):.2%}")
```

### Multi-asset resampling

For multi-asset data the same random return indices are applied to all symbols simultaneously. This synchronized resampling preserves the cross-asset correlation structure of the original data. Resampling each symbol independently would destroy those correlations and produce unrealistic scenarios that do not reflect how the assets actually co-move.

All symbols must have the same number of bars. If they differ, `monte_carlo_validation` raises a `ValueError` naming each symbol and its bar count, and suggests aligning the series to a common date range before running:

```python
# Raises ValueError if bar counts differ across symbols:
# "monte_carlo_validation requires all symbols to have the same number
#  of bars for synchronized resampling. Got: AAPL: 252, MSFT: 248.
#  Align your series to a common date range before running."
mc_results = monte_carlo_validation(
    strategy_factory=qc.SMACrossover,
    params={'fast_period': 20, 'slow_period': 100},
    data={'AAPL': bars_aapl, 'MSFT': bars_msft},  # must have equal length
    n_simulations=1000,
)
```

`strategy_factory` must be picklable when `n_jobs != 1`. The same rules apply as for `GridSearchOptimizer`: pass a class or module-level function, not a lambda, and guard with `if __name__ == '__main__':` on Windows.

## 15. Tick Data

### Loading

```python
# CSV
ticks = qc.load_tick_csv('data/aapl_ticks.csv', 'AAPL')

# Parquet - List[TickData]
ticks = qc.load_tick_parquet('data/aapl_ticks.parquet', symbol='AAPL', use_numpy=False)

# Parquet - (N, 4) numpy array [timestamp_ns, price, quantity, side]
arr = qc.load_tick_parquet('data/aapl_ticks.parquet', use_numpy=True)
```

### Running a tick backtest

```python
results = qc.run_tick_backtest(
    strategy=MyStrategy(),
    tick_data={'AAPL': ticks},
    initial_capital=100_000.0,
    mm_refresh_interval_ns=1_000_000_000,
    equity_snapshot_interval_ns=60_000_000_000,
).compute()

print(results)
print(results.metrics)
```

Or use `BacktestEngine` directly for full control:

```python
engine = qc.BacktestEngine(100_000.0)
engine.add_tick_data('AAPL', ticks)
engine.set_strategy(MyStrategy())
engine.set_mm_refresh_interval(1_000_000_000)
engine.set_equity_snapshot_interval(60_000_000_000)
engine.run()
```

### Aggregating ticks to bars

If your strategy is bar-based, aggregate before running:

```python
bars    = qc.aggregate_ticks_to_bars(ticks, bar_duration_ns=60_000_000_000)
results = qc.run_backtest(strategy=MyStrategy(), data={'AAPL': bars}, initial_capital=100_000.0)
```

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

### Performance notes

The market maker refresh interval is the main lever. Without throttling the MM refreshes on every tick, which dominates cost at high frequencies:

| MM refresh interval | Ticks/s | vs no throttle |
|---|---|---|
| No throttle | 340 K/s | 1.0x |
| 1s | 1.66 M/s | 4.9x |
| 10s | 2.78 M/s | 8.2x |

For large datasets use the numpy `add_tick_data` path (3–5x faster than `List[TickData]`). Set `equity_snapshot_interval_ns` to avoid a dense equity curve - the snapshot itself is cheap, but the vector can grow very large on million-tick datasets.

See [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for the full tick data benchmark results.

---

## 16. Trading Calendar

Without a trading calendar, the engine processes every bar in the series regardless of date. If your data contains bars dated on weekends, exchange holidays, or other non-trading days — common with synthetic data, research databases, or anything you have assembled manually — those bars generate signals and simulate fills against a market maker that will always fill. For daily strategies this inflates performance by roughly 10–14 extra trading days per year.

The `TradingCalendar` class filters a bar series to remove non-trading days before the data reaches the engine. It is backed by `pandas_market_calendars`, which covers 50+ exchanges including NYSE, NASDAQ, LSE, TSX, EUREX, and ASX, and handles the edge cases you would otherwise have to hardcode: Good Friday (calculated from Easter), Juneteenth (added to NYSE in 2022), observed holidays (e.g. when Christmas falls on a Sunday), and ad hoc closures.

Requires: `pip install pandas_market_calendars`

### Usage

**At load time** — the simplest option:

```python
bars = qc.load_csv_data('data/aapl.csv', 'AAPL', calendar='NYSE')
bars = qc.load_parquet_data('data/aapl.parquet', 'AAPL', calendar='NYSE')
```

**At run time** — applies the same calendar to every symbol in the backtest:

```python
results = qc.run_backtest(
    strategy=MyStrategy(),
    data={'AAPL': bars_aapl, 'MSFT': bars_msft},
    initial_capital=100_000.0,
    calendar='NYSE',
)
```

**Explicitly** — when you need more control:

```python
cal  = qc.TradingCalendar('NYSE')
bars = cal.filter_bars(qc.load_csv_data('data/aapl.csv', 'AAPL'))
```

The `calendar` parameter is not available on `run_tick_backtest`. Tick data loaded from an exchange feed already contains only actual trades — filtering by calendar would be wrong for crypto and other markets that trade on holidays.

### filter_bars parameters

```python
cal.filter_bars(
    bars,
    strict=False,        # when True, raises if more than max_skip_pct bars are removed
    max_skip_pct=0.20,   # threshold for strict mode (default 20%)
)
```

In the default mode (`strict=False`), removed bars emit a `UserWarning` showing how many were dropped and why. In strict mode, a `RuntimeError` is raised if the removed fraction exceeds `max_skip_pct`. Either way, if every bar in the series falls on a non-trading day, `filter_bars` raises `RuntimeError` — an empty series would crash the engine silently.

### is_trading_day

```python
cal = qc.TradingCalendar('NYSE')
cal.is_trading_day(timestamp_ns)  # bool
```

### Available exchanges

```python
qc.TradingCalendar.available_calendars()
# ['ASX', 'BMF', 'CFE', 'CME', 'EUREX', 'LSE', 'NASDAQ', 'NYSE', 'TSX', ...]
```

### When to use it

Use the calendar whenever your data comes from a source that does not already guarantee trading-day-only bars. Yahoo Finance daily data, CRSP, and most vendor feeds are clean. Synthetic data, hand-assembled CSVs, and data resampled from intraday sources without a session filter often are not.

Tick data does not need calendar filtering. A tick is a real trade — if there is a tick, the market was open.