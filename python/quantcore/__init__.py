import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings

_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from . import _core
except ImportError:
    try:
        import _core
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

MarketDataEvent = _core.MarketDataEvent
SignalEvent     = _core.SignalEvent
FillEvent       = _core.FillEvent

ExecutionConfig = _core.ExecutionConfig
ExecutionEngine = _core.ExecutionEngine

Strategy     = _core.Strategy
BuyAndHold   = _core.BuyAndHold
SMACrossover = _core.SMACrossover
MeanReversion = _core.MeanReversion
PairsTrading  = _core.PairsTrading

BacktestEngine = _core.BacktestEngine

PositionSizingContext = _core.PositionSizingContext
# PositionSizer is the C++ abstract base class used by BacktestEngine.
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

# PositionCalculator and PortfolioPositionSizer are standalone Python utilities
# for pre-trade sizing calculations. They are distinct from the C++ PositionSizer
# hierarchy that BacktestEngine uses internally.
from .position_sizing import PositionCalculator, PortfolioPositionSizer


def load_csv_data(filepath: str, symbol: str = "", has_header: bool = True) -> List[BarData]:
    """
    Load OHLCV data from a CSV file.

    Accepts 6-column (timestamp, O, H, L, C, V) or 7-column
    (symbol, timestamp, O, H, L, C, V) layouts. Timestamps may be
    in seconds, milliseconds, or nanoseconds.
    """
    return CSVDataLoader.load(filepath, symbol, has_header)


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
) -> Dict:
    """
    Run a complete backtest and return a results dictionary containing:
    strategy, initial_capital, final_value, total_pnl, total_fees,
    return_pct, equity_curve, timestamps.
    """
    engine     = create_backtest(initial_capital, data, strategy)
    final_value = engine.run()

    return {
        'strategy':        strategy.get_name(),
        'initial_capital': initial_capital,
        'final_value':     final_value,
        'total_pnl':       engine.get_total_pnl(),
        'total_fees':      engine.get_total_fees(),
        'return_pct':      (final_value / initial_capital - 1.0) * 100.0,
        'equity_curve':    engine.get_equity_curve(),
        'timestamps':      engine.get_timestamps(),
    }


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
        self._metrics        = None

    @property
    def net_pnl(self) -> float:
        return self.final_value - self.initial_capital

    def compute(self) -> "BacktestResults":
        """Compute and cache performance metrics. Returns self for chaining."""
        from .analytics import calculate_all_metrics
        import numpy as np
        self._metrics = calculate_all_metrics(np.array(self.equity_curve))
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
    # Data
    'BarData', 'CSVDataLoader', 'load_csv_data',
    # Events
    'MarketDataEvent', 'SignalEvent', 'FillEvent',
    # Execution
    'ExecutionConfig', 'ExecutionEngine',
    # Strategy
    'Strategy', 'BuyAndHold', 'SMACrossover', 'MeanReversion', 'PairsTrading',
    # Backtest
    'BacktestEngine', 'BacktestResults', 'create_backtest', 'run_backtest',
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
    # Utilities
    'hello', 'version',
]
