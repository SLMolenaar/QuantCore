import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings

_current_dir = Path(__file__).parent

try:
    from . import _core
except ImportError as e:
    raise ImportError(
        f"Failed to import C++ extension.\n"
        f"Module should be at: {_current_dir / '_core.cp312-win_amd64.pyd'}\n"
        f"Did you build it? Run: python python/build_module.py"
    ) from e

# ============================================================================
# C++ EXPORTS
# ============================================================================

SignalType = _core.SignalType
Side       = _core.Side
OrderType  = _core.OrderType

BarData       = _core.BarData
CSVDataLoader = _core.CSVDataLoader

TickData       = _core.TickData
TickDataLoader = _core.TickDataLoader

MarketDataEvent = _core.MarketDataEvent
SignalEvent     = _core.SignalEvent
FillEvent       = _core.FillEvent

ExecutionConfig = _core.ExecutionConfig
ExecutionEngine = _core.ExecutionEngine

Strategy      = _core.Strategy
BuyAndHold    = _core.BuyAndHold
SMACrossover  = _core.SMACrossover
MeanReversion = _core.MeanReversion
PairsTrading  = _core.PairsTrading

BacktestEngine = _core.BacktestEngine

PositionSizingContext = _core.PositionSizingContext
PositionSizer         = _core.PositionSizer
FixedPercentage       = _core.FixedPercentage
RiskBased             = _core.RiskBased
KellyCriterion        = _core.KellyCriterion
EqualWeight           = _core.EqualWeight
VolatilityTargeting   = _core.VolatilityTargeting
FixedShares           = _core.FixedShares

RiskCheckResult   = _core.RiskCheckResult
RiskCheckResponse = _core.RiskCheckResponse
RiskLimits        = _core.RiskLimits
RiskManager       = _core.RiskManager

PortfolioContext = _core.PortfolioContext

hello   = _core.hello
version = _core.version

# ============================================================================
# PYTHON HELPERS
# ============================================================================

from .position_sizing import PositionCalculator, PortfolioPositionSizer
from .parquet_loader import ParquetDataLoader
from .tick_parquet_loader import TickParquetLoader
from .corporate_actions import CorporateActionsAdjuster, SplitEvent, DividendEvent
from .calendar import TradingCalendar
from .walk_forward import (
    BacktestConfig,
    GridSearchOptimizer,
    WalkForwardAnalyzer,
    WalkForwardResult,
    OptimizationResult,
    ParameterGrid,
    monte_carlo_validation,
    pct_to_decimal,
    decimal_to_pct,
)


def load_parquet_data(
        filepath: str,
        symbol: str = "",
        use_numpy: bool = True,
        calendar: Optional[str] = None,
):
    """
    Load OHLCV data from a Parquet file.

    Accepts any column naming convention that resolves to
    timestamp, open, high, low, close, volume. Timestamps may be
    datetime64 or integer (seconds, milliseconds, microseconds, nanoseconds).

    Args:
        filepath:  Path to the .parquet file.
        symbol:    Symbol name to assign when the file has no 'symbol' column.
        use_numpy: When True (default), reads via the fast numpy path and
                   returns a numpy (N, 6) array suitable for
                   BacktestEngine.add_data(symbol, array).
                   When False, returns List[BarData] (same as load_csv_data).
        calendar:  Optional exchange name (e.g. "NYSE"). When provided, bars
                   that fall on non-trading days are removed before returning.
                   Forces use_numpy=False internally since filtering requires
                   BarData objects. Requires pandas_market_calendars.

    Requires: pyarrow  (pip install pyarrow)
    """
    if calendar:
        bars = ParquetDataLoader.load(filepath, symbol)
        return TradingCalendar(calendar).filter_bars(bars)
    if use_numpy:
        return ParquetDataLoader.load_numpy(filepath, symbol)
    return ParquetDataLoader.load(filepath, symbol)


def load_csv_data(
        filepath: str,
        symbol: str = "",
        has_header: bool = True,
        calendar: Optional[str] = None,
) -> List[BarData]:
    """
    Load OHLCV data from a CSV file.

    Accepts 6-column (timestamp, O, H, L, C, V) or 7-column
    (symbol, timestamp, O, H, L, C, V) layouts. Timestamps may be
    in seconds, milliseconds, or nanoseconds.

    Args:
        filepath:   Path to the .csv file.
        symbol:     Symbol name to assign when the file has no symbol column.
        has_header: Whether the file has a header row (default True).
        calendar:   Optional exchange name (e.g. "NYSE"). When provided, bars
                    that fall on non-trading days are removed before returning.
                    Requires pandas_market_calendars.
    """
    bars = CSVDataLoader.load(filepath, symbol, has_header)
    if calendar:
        bars = TradingCalendar(calendar).filter_bars(bars)
    return bars


def load_tick_csv(filepath: str, symbol: str = "", has_header: bool = True) -> List[TickData]:
    """
    Load tick data from a CSV file.

    Accepted column layouts (auto-detected by column count):
      3 cols: timestamp, price, quantity
      4 cols: timestamp, price, quantity, side
      5 cols: symbol, timestamp, price, quantity, side

    side values: B/b/buy/BUY or S/s/sell/SELL
    Timestamps in seconds, milliseconds, microseconds, or nanoseconds.
    """
    return TickDataLoader.load(filepath, symbol, has_header)


def load_tick_parquet(
        filepath: str,
        symbol: str = "",
        use_numpy: bool = True,
):
    """
    Load tick data from a Parquet file.

    Args:
        filepath:  Path to the .parquet file.
        symbol:    Symbol name to assign when no 'symbol' column is present.
        use_numpy: When True (default), returns a (N, 4) float64 numpy array
                   [timestamp_ns, price, quantity, side] suitable for
                   BacktestEngine.add_tick_data(symbol, array).
                   When False, returns List[TickData].

    Requires: pyarrow  (pip install pyarrow)
    """
    if use_numpy:
        return TickParquetLoader.load_numpy(filepath)
    return TickParquetLoader.load(filepath, symbol)


def aggregate_ticks_to_bars(ticks: List[TickData], bar_duration_ns: int) -> List[BarData]:
    """
    Aggregate a list of TickData objects into OHLCV bars.

    Args:
        ticks:           List of TickData objects (must be sorted by timestamp).
        bar_duration_ns: Bar width in nanoseconds.
                         Common values:
                           1_000_000_000      -- 1 second
                           60_000_000_000     -- 1 minute
                           3_600_000_000_000  -- 1 hour
    """
    return TickDataLoader.aggregate_to_bars(ticks, bar_duration_ns)


def create_backtest(
        initial_capital: float = 100000.0,
        data: Optional[Dict[str, List[BarData]]] = None,
        strategy: Optional[Strategy] = None,
) -> BacktestEngine:
    """
    Convenience constructor for a BacktestEngine.

    Example:
        >>> bars   = load_csv_data("data/AAPL.csv", "AAPL")
        >>> strat  = SMACrossover(fast_period=50, slow_period=200)
        >>> engine = create_backtest(100_000, {"AAPL": bars}, strat)
        >>> final  = engine.run()
    """
    engine = BacktestEngine(initial_capital)
    if data:
        for symbol, bars in data.items():
            engine.add_data(symbol, bars)
    if strategy:
        engine.set_strategy(strategy)
    return engine


def run_backtest(
        strategy: Strategy,
        data: Dict[str, List[BarData]],
        initial_capital: float = 100000.0,
        calendar: Optional[str] = None,
        benchmark: Optional[Dict[str, List[BarData]]] = None,
        benchmark_strategy: Optional[Strategy] = None,
) -> "BacktestResults":
    """
    Run a complete backtest and return a BacktestResults object.

    This is the recommended entry point for one-shot backtests. For more
    control over execution config, position sizing, and risk limits, use
    BacktestEngine directly.

    Args:
        strategy:           Strategy instance.
        data:               Dict mapping symbol to List[BarData].
        initial_capital:    Starting capital.
        calendar:           Optional exchange name (e.g. "NYSE"). When provided,
                            each symbol's bar series is filtered to remove bars
                            that fall on non-trading days before the backtest runs.
                            Requires pandas_market_calendars.
        benchmark:          Optional data dict for the benchmark backtest. When
                            provided a second backtest is run automatically and the
                            resulting equity curve is stored on BacktestResults for
                            use with compute() and the plotting functions.
                            Defaults to running BuyAndHold on the same `data` dict
                            when benchmark_strategy is provided without benchmark.
        benchmark_strategy: Strategy used for the benchmark backtest. Defaults to
                            BuyAndHold() when benchmark data is given but no
                            strategy is specified.
    """
    if calendar:
        cal  = TradingCalendar(calendar)
        data = {sym: cal.filter_bars(bars) for sym, bars in data.items()}

    engine      = create_backtest(initial_capital, data, strategy)
    final_value = engine.run()

    benchmark_equity_curve = None
    if benchmark is not None or benchmark_strategy is not None:
        bm_data     = benchmark if benchmark is not None else data
        bm_strategy = benchmark_strategy if benchmark_strategy is not None else BuyAndHold()
        bm_engine   = create_backtest(initial_capital, bm_data, bm_strategy)
        bm_engine.run()
        benchmark_equity_curve = bm_engine.get_equity_curve()

    return BacktestResults({
        'strategy':                strategy.get_name(),
        'initial_capital':         initial_capital,
        'final_value':             final_value,
        'total_pnl':               engine.get_total_pnl(),
        'total_fees':              engine.get_total_fees(),
        'return_pct':              (final_value / initial_capital - 1.0) * 100.0,
        'equity_curve':            engine.get_equity_curve(),
        'timestamps':              engine.get_timestamps(),
        'trade_pnls':              engine.get_trade_pnls(),
        'benchmark_equity_curve':  benchmark_equity_curve,
    })


def run_tick_backtest(
        strategy: Strategy,
        tick_data: Dict[str, List[TickData]],
        initial_capital: float = 100000.0,
        mm_refresh_interval_ns: int = 1_000_000_000,
        equity_snapshot_interval_ns: int = 60_000_000_000,
) -> "BacktestResults":
    """
    Run a backtest on tick data and return a BacktestResults object.

    Sensible defaults are applied for tick mode:
      - Market maker refresh interval: 1 second (avoids refreshing on every tick).
      - Equity snapshot interval: 1 minute (keeps the equity curve manageable).

    Both can be overridden. Pass 0 for either interval to revert to
    event-by-event behaviour (not recommended for large tick datasets).

    Args:
        strategy:                    Strategy instance.
        tick_data:                   Dict mapping symbol to List[TickData].
        initial_capital:             Starting capital.
        mm_refresh_interval_ns:      Minimum nanoseconds between MM quote refreshes.
        equity_snapshot_interval_ns: Minimum nanoseconds between equity snapshots.
    """
    engine = BacktestEngine(initial_capital)

    for symbol, ticks in tick_data.items():
        engine.add_tick_data(symbol, ticks)

    engine.set_strategy(strategy)
    engine.set_mm_refresh_interval(mm_refresh_interval_ns)
    engine.set_equity_snapshot_interval(equity_snapshot_interval_ns)

    final_value = engine.run()

    return BacktestResults({
        'strategy':        strategy.get_name(),
        'initial_capital': initial_capital,
        'final_value':     final_value,
        'total_pnl':       engine.get_total_pnl(),
        'total_fees':      engine.get_total_fees(),
        'return_pct':      (final_value / initial_capital - 1.0) * 100.0,
        'equity_curve':    engine.get_equity_curve(),
        'timestamps':      engine.get_timestamps(),
        'trade_pnls':      engine.get_trade_pnls(),
    })


class BacktestResults:
    """Container for backtest results with lazy metric computation."""

    def __init__(self, results: Dict):
        self.strategy_name           = results.get('strategy', 'Unknown')
        self.initial_capital         = results.get('initial_capital', 0.0)
        self.final_value             = results.get('final_value', 0.0)
        self.total_pnl               = results.get('total_pnl', 0.0)
        self.total_fees              = results.get('total_fees', 0.0)
        self.return_pct              = results.get('return_pct', 0.0)
        self.equity_curve            = results.get('equity_curve', [])
        self.timestamps              = results.get('timestamps', [])
        self.trade_pnls              = results.get('trade_pnls', [])
        self.benchmark_equity_curve  = results.get('benchmark_equity_curve', None)
        self._metrics                = None
        self._benchmark_metrics      = None

    @property
    def net_pnl(self) -> float:
        return self.final_value - self.initial_capital

    def compute(self) -> "BacktestResults":
        """
        Compute and cache performance metrics. Returns self for chaining.

        When benchmark_equity_curve is set (either from run_backtest with a
        benchmark argument or by assigning it manually), benchmark-relative
        metrics are computed and stored in benchmark_metrics.
        """
        import numpy as np
        from .analytics import (
            calculate_all_metrics, calculate_returns, calculate_benchmark_metrics,
        )

        equity_arr = np.array(self.equity_curve)
        ts_arr     = np.array(self.timestamps, dtype=np.int64) if self.timestamps else None

        self._metrics = calculate_all_metrics(
            equity_arr,
            trade_pnls=self.trade_pnls if self.trade_pnls else None,
            timestamps=ts_arr,
        )

        if self.benchmark_equity_curve is not None:
            strat_returns = calculate_returns(equity_arr)
            bm_returns    = calculate_returns(np.array(self.benchmark_equity_curve))
            self._benchmark_metrics = calculate_benchmark_metrics(
                strat_returns, bm_returns, timestamps=ts_arr
            )

        return self

    @property
    def metrics(self):
        if self._metrics is None:
            raise RuntimeError("Call .compute() first.")
        return self._metrics

    @property
    def benchmark_metrics(self):
        if self._metrics is None:
            raise RuntimeError("Call .compute() first.")
        return self._benchmark_metrics

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Backtest Results - {self.strategy_name}",
            "=" * 60,
            f"Initial Capital:  ${self.initial_capital:>15,.2f}",
            f"Final Value:      ${self.final_value:>15,.2f}",
            f"Total PnL:        ${self.total_pnl:>15,.2f}",
            f"Total Fees:       ${self.total_fees:>15,.2f}",
            f"Net PnL:          ${self.net_pnl:>15,.2f}",
            f"Return:           {self.return_pct:>15.2f}%",
            "=" * 60,
            ]

        if self._metrics is None:
            # compute() has not been called yet — hint at what's available
            if self.benchmark_equity_curve is not None:
                lines += [
                    "",
                    "  Call .compute() to calculate performance and benchmark metrics.",
                    "=" * 60,
                    ]
            else:
                lines += [
                    "",
                    "  Call .compute() to calculate performance metrics.",
                    "=" * 60,
                    ]
        elif self._benchmark_metrics is not None:
            bm = self._benchmark_metrics
            lines += [
                "",
                "  vs Benchmark",
                "-" * 60,
                f"  Active Return (ann.):    {bm.active_return:>10.2f}%",
                f"  Alpha (ann.):            {bm.alpha:>10.2f}%",
                f"  Beta:                    {bm.beta:>10.2f}",
                f"  Information Ratio:       {bm.information_ratio:>10.2f}",
                f"  Up Capture:              {bm.up_capture:>10.2f}%",
                f"  Down Capture:            {bm.down_capture:>10.2f}%",
                "=" * 60,
                ]

        return "\n".join(lines)


# ============================================================================
# VERSION
# ============================================================================

__version__ = version()
__all__ = [
    # Enums
    'SignalType', 'Side', 'OrderType',
    # Bar data
    'BarData', 'CSVDataLoader', 'load_csv_data',
    # Tick data
    'TickData', 'TickDataLoader',
    'load_tick_csv', 'load_tick_parquet', 'aggregate_ticks_to_bars',
    'TickParquetLoader',
    # Events
    'MarketDataEvent', 'SignalEvent', 'FillEvent',
    # Execution
    'ExecutionConfig', 'ExecutionEngine',
    # Strategy
    'Strategy', 'BuyAndHold', 'SMACrossover', 'MeanReversion', 'PairsTrading',
    # Backtest
    'BacktestEngine', 'BacktestResults',
    'create_backtest', 'run_backtest', 'run_tick_backtest',
    # C++ position sizing
    'PositionSizingContext', 'PositionSizer',
    'FixedPercentage', 'RiskBased', 'KellyCriterion',
    'EqualWeight', 'VolatilityTargeting', 'FixedShares',
    # Python position sizing utilities
    'PositionCalculator', 'PortfolioPositionSizer',
    # Risk management
    'RiskCheckResult', 'RiskCheckResponse', 'RiskLimits', 'RiskManager',
    # Portfolio context
    'PortfolioContext',
    # Optimization and walk-forward
    'BacktestConfig',
    'GridSearchOptimizer', 'WalkForwardAnalyzer',
    'WalkForwardResult', 'OptimizationResult', 'ParameterGrid',
    'monte_carlo_validation',
    'pct_to_decimal', 'decimal_to_pct',
    # Trading calendar
    'TradingCalendar',
    # Data utilities
    'ParquetDataLoader', 'load_parquet_data',
    'CorporateActionsAdjuster', 'SplitEvent', 'DividendEvent',
    # C++ utilities
    'hello', 'version',
]