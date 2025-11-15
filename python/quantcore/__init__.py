import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings

# add current directory to path for C++ module import
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# import the C++ extension
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
# EXPORTS FROM C++ MODULE
# ============================================================================

# Enums
SignalType = _core.SignalType
Side = _core.Side
OrderType = _core.OrderType

# Data structures
BarData = _core.BarData
CSVDataLoader = _core.CSVDataLoader

# Events
MarketDataEvent = _core.MarketDataEvent
SignalEvent = _core.SignalEvent
FillEvent = _core.FillEvent

# Execution
ExecutionConfig = _core.ExecutionConfig
ExecutionEngine = _core.ExecutionEngine

# Strategy
Strategy = _core.Strategy
BuyAndHold = _core.BuyAndHold
SMACrossover = _core.SMACrossover
MeanReversion = _core.MeanReversion

# Backtest engine
BacktestEngine = _core.BacktestEngine

# Position Sizing
PositionSizingContext = _core.PositionSizingContext
PositionSizer = _core.PositionSizer
FixedPercentage = _core.FixedPercentage
RiskBased = _core.RiskBased
KellyCriterion = _core.KellyCriterion
EqualWeight = _core.EqualWeight
VolatilityTargeting = _core.VolatilityTargeting
FixedShares = _core.FixedShares

# Risk Management
RiskCheckResult = _core.RiskCheckResult
RiskCheckResponse = _core.RiskCheckResponse
RiskLimits = _core.RiskLimits
RiskManager = _core.RiskManager

# Utilities
hello = _core.hello
version = _core.version


# ============================================================================
# PYTHON HELPERS
# ============================================================================

def load_csv_data(filepath: str, symbol: str = "", has_header: bool = True) -> List[BarData]:
    """
    Load OHLCV data from CSV file

    CSV Format:
        6 columns: timestamp,open,high,low,close,volume
        7 columns: symbol,timestamp,open,high,low,close,volume

    Timestamps can be seconds, milliseconds, or nanoseconds
    """
    return CSVDataLoader.load(filepath, symbol, has_header)


def create_backtest(
        initial_capital: float = 100000.0,
        data: Optional[Dict[str, List[BarData]]] = None,
        strategy: Optional[Strategy] = None
) -> BacktestEngine:
    """
    Create a backtest engine with data and strategy

    Example:
        >>> bars = load_csv_data("data/AAPL.csv", "AAPL")
        >>> strategy = SMACrossover(fast_period=50, slow_period=200)
        >>> engine = create_backtest(100000, {"AAPL": bars}, strategy)
        >>> final_value = engine.run()
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
        initial_capital: float = 100000.0
) -> Dict[str, float]:
    """
    Run a complete backtest and return results

    Returns dict with:
        - initial_capital, final_value, total_pnl, total_fees
        - return_pct, equity_curve, timestamps
    """
    engine = create_backtest(initial_capital, data, strategy)
    final_value = engine.run()
    total_pnl = engine.get_total_pnl()
    total_fees = engine.get_total_fees()
    equity_curve = engine.get_equity_curve()
    timestamps = engine.get_timestamps()

    return {
        'strategy': strategy.get_name(),
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'return_pct': ((final_value / initial_capital) - 1.0) * 100.0,
        'equity_curve': equity_curve,
        'timestamps': timestamps
    }


class BacktestResults:
    """Container for backtest results"""

    def __init__(self, engine: BacktestEngine, initial_capital: float, strategy: Strategy):
        self.engine = engine
        self.initial_capital = initial_capital
        self.strategy_name = strategy.get_name()
        self.final_value = None
        self.total_pnl = None
        self.total_fees = None
        self.execution_engines = {}

    def compute(self):
        """Compute results after backtest run"""
        self.final_value = self.engine.run()
        self.total_pnl = self.engine.get_total_pnl()
        self.total_fees = self.engine.get_total_fees()
        return self

    @property
    def return_pct(self) -> float:
        """Return percentage"""
        if self.final_value is None:
            return 0.0
        return ((self.final_value / self.initial_capital) - 1.0) * 100.0

    @property
    def net_pnl(self) -> float:
        """Net PnL after fees"""
        if self.total_pnl is None or self.total_fees is None:
            return 0.0
        return self.total_pnl - self.total_fees

    def get_position(self, symbol: str) -> float:
        """Get final position for a symbol"""
        ee = self.engine.get_execution_engine(symbol)
        return ee.get_position() if ee else 0.0

    def __repr__(self) -> str:
        if self.final_value is None:
            return f"<BacktestResults (not computed)>"
        return (
            f"<BacktestResults\n"
            f"  Strategy: {self.strategy_name}\n"
            f"  Initial: ${self.initial_capital:,.2f}\n"
            f"  Final: ${self.final_value:,.2f}\n"
            f"  PnL: ${self.total_pnl:,.2f}\n"
            f"  Fees: ${self.total_fees:,.2f}\n"
            f"  Return: {self.return_pct:.2f}%>"
        )

    def summary(self) -> str:
        """Get formatted summary"""
        if self.final_value is None:
            return "Backtest not yet computed. Call .compute() first."

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
            "=" * 60
        ]
        return "\n".join(lines)


# ============================================================================
# VERSION AND METADATA
# ============================================================================

__version__ = version()
__all__ = [
    # Enums
    'SignalType',
    'Side',
    'OrderType',
    # Data
    'BarData',
    'CSVDataLoader',
    'load_csv_data',
    # Events
    'MarketDataEvent',
    'SignalEvent',
    'FillEvent',
    # Execution
    'ExecutionConfig',
    'ExecutionEngine',
    # Strategy
    'Strategy',
    'BuyAndHold',
    'SMACrossover',
    'MeanReversion',
    # Backtest
    'BacktestEngine',
    'BacktestResults',
    'create_backtest',
    'run_backtest',
    # Utils
    'hello',
    'version',
    # Position Sizing
    'PositionSizingContext',
    'PositionSizer',
    'FixedPercentage',
    'RiskBased',
    'KellyCriterion',
    'EqualWeight',
    'VolatilityTargeting',
    'FixedShares',
]