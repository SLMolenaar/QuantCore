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
# PositionSizer is the C++ abstract base class used by BacktestEngine.
PositionSizer       = _core.PositionSizer
FixedPercentage     = _core.FixedPercentage
RiskBased           = _core.RiskBased
KellyCriterion      = _core.KellyCriterion
EqualWeight         = _core.EqualWeight
VolatilityTargeting = _core.VolatilityTargeting
FixedShares         = _core.FixedShares

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

# PositionCalculator and PortfolioPositionSizer are standalone Python utilities
# for pre-trade sizing calculations. They are distinct from the C++ PositionSizer
# hierarchy that BacktestEngine uses internally.
from .position_sizing import PositionCalculator, PortfolioPositionSizer
from .parquet_loader import ParquetDataLoader
from .tick_parquet_loader import TickParquetLoader
from .corporate_actions import CorporateActionsAdjuster, SplitEvent, DividendEvent
from .calendar import TradingCalendar


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
    # Calendar filtering requires BarData objects, not a raw numpy array.
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
) -> "BacktestResults":
    """
    Run a complete backtest and return a BacktestResults object.

    This is the recommended entry point for one-shot backtests. For more
    control over execution config, position sizing, and risk limits, use
    BacktestEngine directly.

    Args:
        strategy:        Strategy instance.
        data:            Dict mapping symbol to List[BarData].
        initial_capital: Starting capital.
        calendar:        Optional exchange name (e.g. "NYSE"). When provided,
                         each symbol's bar series is filtered to remove bars
                         that fall on non-trading days before the backtest runs.
                         Requires pandas_market_calendars.
    """
    if calendar:
        cal  = TradingCalendar(calendar)
        data = {sym: cal.filter_bars(bars) for sym, bars in data.items()}

    engine      = create_backtest(initial_capital, data, strategy)
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
        self.strategy_name   = results.get('strategy', 'Unknown')
        self.initial_capital = results.get('initial_capital', 0.0)
        self.final_value     = results.get('final_value', 0.0)
        self.total_pnl       = results.get('total_pnl', 0.0)
        self.total_fees      = results.get('total_fees', 0.0)
        self.return_pct      = results.get('return_pct', 0.0)
        self.equity_curve    = results.get('equity_curve', [])
        self.timestamps      = results.get('timestamps', [])
        self.trade_pnls      = results.get('trade_pnls', [])
        self._metrics        = None

    @property
    def net_pnl(self) -> float:
        return self.final_value - self.initial_capital

    def compute(self) -> "BacktestResults":
        """Compute and cache performance metrics. Returns self for chaining."""
        from .analytics import calculate_all_metrics
        import numpy as np
        self._metrics = calculate_all_metrics(
            np.array(self.equity_curve),
            trade_pnls=self.trade_pnls if self.trade_pnls else None,
            timestamps=np.array(self.timestamps, dtype=np.int64) if self.timestamps else None,
        )
        return self

    @property
    def metrics(self):
        if self._metrics is None:
            raise RuntimeError("Call .compute() first.")
        return self._metrics

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
    # C++ position sizing (used by BacktestEngine)
    'PositionSizingContext', 'PositionSizer',
    'FixedPercentage', 'RiskBased', 'KellyCriterion',
    'EqualWeight', 'VolatilityTargeting', 'FixedShares',
    # Python position sizing utilities (standalone helpers)
    'PositionCalculator', 'PortfolioPositionSizer',
    # Risk management
    'RiskCheckResult', 'RiskCheckResponse', 'RiskLimits', 'RiskManager',
    # Portfolio context
    'PortfolioContext',
    # Trading calendar
    'TradingCalendar',
    # Utilities
    'hello', 'version',
    'ParquetDataLoader', 'load_parquet_data',
    'CorporateActionsAdjuster', 'SplitEvent', 'DividendEvent',
]