"""
Walk-forward analysis and parameter optimization.

Tools for validating strategies and finding optimal parameters:
- Walk-forward analysis to avoid overfitting
- Grid search optimization
- Rolling window backtests
- Monte Carlo validation

Parallelism notes
-----------------
Both GridSearchOptimizer and WalkForwardAnalyzer support n_jobs > 1 via
ProcessPoolExecutor. Key constraints:

  * Use processes, not threads. Each backtest is CPU-bound C++ work and
    Python's GIL means threads give no parallelism benefit.

  * strategy_factory must be a picklable callable — a class or module-level
    function. Lambdas and closures will raise a PicklingError at runtime.
    Example: pass SMACrossover, not lambda **k: SMACrossover(**k).

  * BarData objects (C++ pybind11) are not picklable. Data is serialised to
    plain tuples before dispatch and reconstructed inside each worker.

  * BacktestConfig stores only Python primitives so it is always picklable,
    regardless of the underlying quantcore C++ object pickling behaviour.

  * Process spawn overhead is non-trivial (~0.5 s per worker on first import).
    Parallelism only activates when the number of tasks exceeds
    MIN_PARALLEL_TASKS (default 4). Below that threshold the sequential
    path is used regardless of n_jobs.

  * On Windows, guard your entry point with if __name__ == '__main__'.

Return Format Convention
------------------------
This differs from PerformanceMetrics which uses PERCENTAGE format (10.5, -5.0).
Use the helper functions to convert if needed:
  - pct_to_decimal(10.5) -> 0.105
  - decimal_to_pct(0.105) -> 10.5

Multi-asset support
-------------------
WalkForwardAnalyzer and monte_carlo_validation both support multi-asset data.

WalkForwardAnalyzer slices all symbols over the same timestamp range, using
the first symbol's bar series as the reference timeline. Bars for all other
symbols are selected by timestamp inclusion within [window_start, window_end),
so minor gaps in one symbol do not corrupt the windows of others.

monte_carlo_validation resamples the same log-return indices across all
symbols simultaneously (synchronized block bootstrap), preserving
cross-asset correlation structure. Resampling each symbol independently
would destroy those correlations and produce unrealistic scenarios.

BacktestConfig
--------------
BacktestConfig controls how the engine is configured inside every worker
backtest. It is picklable (plain Python primitives only) and can be passed
to WalkForwardAnalyzer, GridSearchOptimizer, and monte_carlo_validation so
that optimization and stress-testing run under identical conditions to the
final production backtest.

Supported sizer_type values: 'FixedPercentage', 'RiskBased', 'KellyCriterion',
'EqualWeight', 'VolatilityTargeting', 'FixedShares'.

Example:
    config = BacktestConfig(
        sizer_type='VolatilityTargeting',
        sizer_args=(0.20,),
        bars_per_year=24 * 365,
        max_position_pct=0.40,
        max_leverage=1.5,
    )
    wfa = WalkForwardAnalyzer(
        strategy_factory=MyStrategy,
        param_grid={...},
        backtest_config=config,
    )
"""

import sys
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


MIN_PARALLEL_TASKS = 4


def pct_to_decimal(pct: float) -> float:
    """Convert percentage to decimal (10.5 -> 0.105)"""
    return pct / 100.0


def decimal_to_pct(decimal: float) -> float:
    """Convert decimal to percentage (0.105 -> 10.5)"""
    return decimal * 100.0


# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Engine configuration forwarded into every worker backtest.

    All fields are plain Python primitives so BacktestConfig is always
    picklable, regardless of the pickling behaviour of the underlying
    quantcore C++ objects.

    Fields
    ------
    sizer_type : str
        Name of the PositionSizer subclass to use.
        One of: 'FixedPercentage', 'RiskBased', 'KellyCriterion',
                'EqualWeight', 'VolatilityTargeting', 'FixedShares'.
    sizer_args : tuple
        Positional arguments forwarded to the sizer constructor.
        Examples:
          FixedPercentage(0.10)     -> sizer_type='FixedPercentage',
                                       sizer_args=(0.10,)
          VolatilityTargeting(0.20) -> sizer_type='VolatilityTargeting',
                                       sizer_args=(0.20,)
    bars_per_year : int
        Used for volatility annualisation inside the engine.
        252 for daily equity, 8760 for 1h crypto, etc.
    maker_fee : float
        Fraction of notional charged when an order adds liquidity.
    taker_fee : float
        Fraction of notional charged when an order removes liquidity.
    latency_ns : int
        Order submission latency in nanoseconds.
    slippage_pct : float
        One-way slippage as a fraction of price.
    max_position_pct : float
        Maximum single-asset notional as a fraction of capital.
    max_leverage : float
        Maximum total notional / capital across all positions.
    max_loss_pct : float
        Halt trading if drawdown from initial capital exceeds this fraction.
    max_order_value : float
        Maximum notional per single order. 0.0 = disabled.
    risk_enabled : bool
        Set False to disable all risk checks (useful for unit tests).
    """
    sizer_type:       str   = 'FixedPercentage'
    sizer_args:       tuple = (0.10,)
    bars_per_year:    int   = 252
    maker_fee:        float = 0.0001
    taker_fee:        float = 0.0002
    latency_ns:       int   = 1_000_000
    slippage_pct:     float = 0.0
    max_position_pct: float = 0.20
    max_leverage:     float = 2.0
    max_loss_pct:     float = 0.50
    max_order_value:  float = 0.0
    risk_enabled:     bool  = True


def _build_engine(data: Dict[str, list], capital: float, config: BacktestConfig):
    """
    Construct and fully configure a BacktestEngine from a BacktestConfig.

    Centralises all engine setup so that every worker, WalkForwardAnalyzer,
    GridSearchOptimizer, and monte_carlo_validation all run under identical
    conditions when given the same config.

    Returns the configured engine before run() is called.
    """
    import quantcore as qc

    exec_config              = qc.ExecutionConfig()
    exec_config.maker_fee    = config.maker_fee
    exec_config.taker_fee    = config.taker_fee
    exec_config.latency_ns   = config.latency_ns
    exec_config.slippage_pct = config.slippage_pct

    engine = qc.BacktestEngine(capital, exec_config)

    for symbol, bars in data.items():
        engine.add_data(symbol, bars)

    sizer_classes = {
        'FixedPercentage':    qc.FixedPercentage,
        'RiskBased':          qc.RiskBased,
        'KellyCriterion':     qc.KellyCriterion,
        'EqualWeight':        qc.EqualWeight,
        'VolatilityTargeting': qc.VolatilityTargeting,
        'FixedShares':        qc.FixedShares,
    }
    cls = sizer_classes.get(config.sizer_type)
    if cls is None:
        raise ValueError(
            f"Unknown sizer_type '{config.sizer_type}'. "
            f"Valid values: {list(sizer_classes)}"
        )
    engine.set_position_sizer(cls(*config.sizer_args))
    engine.set_bars_per_year(config.bars_per_year)

    limits = qc.RiskLimits()
    limits.max_position_pct = config.max_position_pct
    limits.max_leverage     = config.max_leverage
    limits.max_loss_pct     = config.max_loss_pct
    limits.max_order_value  = config.max_order_value
    limits.enabled          = config.risk_enabled
    engine.set_risk_limits(limits)

    return engine


def _run_configured_backtest(
        strategy,
        data:    Dict[str, list],
        capital: float,
        config:  BacktestConfig,
):
    """
    Build, configure, run a BacktestEngine and return a BacktestResults object.

    Used by all workers when a BacktestConfig is provided, replacing the
    default qc.run_backtest() call which uses hardwired engine defaults.
    """
    import quantcore as qc

    engine = _build_engine(data, capital, config)
    engine.set_strategy(strategy)
    engine.run()

    return qc.BacktestResults({
        'strategy':        strategy.get_name(),
        'initial_capital': capital,
        'final_value':     engine.get_total_pnl() + capital,
        'total_pnl':       engine.get_total_pnl(),
        'total_fees':      engine.get_total_fees(),
        'return_pct':      engine.get_total_pnl() / capital * 100.0,
        'equity_curve':    engine.get_equity_curve(),
        'timestamps':      engine.get_timestamps(),
        'trade_pnls':      engine.get_trade_pnls(),
    })


# ============================================================================
# SERIALISATION HELPERS
# ============================================================================

def _serialize_data(data: Dict[str, list]) -> Dict[str, list]:
    """
    Convert Dict[str, List[BarData]] to a picklable format.

    BarData objects are C++ pybind11 wrappers and cannot be pickled directly.
    Each bar is extracted into a plain tuple for cross-process transport.
    """
    return {
        symbol: [
            (bar.symbol, bar.timestamp_ns,
             bar.open, bar.high, bar.low, bar.close, bar.volume)
            for bar in bars
        ]
        for symbol, bars in data.items()
    }


def _deserialize_data(serialized: Dict[str, list]) -> Dict[str, list]:
    """Reconstruct BarData objects from serialised tuples inside a worker."""
    import quantcore as qc
    return {
        symbol: [qc.BarData(sym, ts, o, h, l, c, v) for sym, ts, o, h, l, c, v in bars]
        for symbol, bars in serialized.items()
    }


def _serialize_data_plain(data: Dict[str, list]) -> Dict[str, list]:
    """Re-serialise already-reconstructed BarData objects for nested workers."""
    return {
        symbol: [
            (bar.symbol, bar.timestamp_ns,
             bar.open, bar.high, bar.low, bar.close, bar.volume)
            for bar in bars
        ]
        for symbol, bars in data.items()
    }


# ============================================================================
# WINDOW CONSTRUCTION
# ============================================================================

def _slice_by_timestamp(
        data:     Dict[str, list],
        start_ns: int,
        end_ns:   int,
) -> Dict[str, list]:
    """
    Return bars for every symbol whose timestamp satisfies start_ns <= ts < end_ns.

    Using half-open intervals means adjacent windows never share a bar and
    every bar belongs to exactly one window.
    """
    return {
        symbol: [b for b in bars if start_ns <= b.timestamp_ns < end_ns]
        for symbol, bars in data.items()
    }


def _build_windows(
        data:       Dict[str, list],
        train_size: int,
        test_size:  int,
) -> List[Tuple[Dict[str, list], Dict[str, list]]]:
    """
    Build (train_data, test_data) window pairs for all symbols.

    The reference timeline is taken from the first symbol's bar series.
    Bars for all symbols are selected by timestamp, so minor gaps in one
    symbol do not affect the windows of the others.

    Each pair covers [train_start, test_end) of real time. Windows advance
    by test_size bars (of the reference symbol) at a time.

    Returns an empty list if there is not enough data for even one window.
    """
    ref_symbol = list(data.keys())[0]
    ref_bars   = data[ref_symbol]
    total      = len(ref_bars)

    if total < train_size + test_size:
        return []

    windows = []
    pos = 0
    while pos + train_size + test_size <= total:
        train_start_ns = ref_bars[pos].timestamp_ns
        train_end_ns   = ref_bars[pos + train_size].timestamp_ns  # exclusive

        last_test_bar  = ref_bars[pos + train_size + test_size - 1]
        # Use the bar after the last test bar as the exclusive upper bound.
        # If we're at the end of the series, add 1 ns to include the final bar.
        if pos + train_size + test_size < total:
            test_end_ns = ref_bars[pos + train_size + test_size].timestamp_ns
        else:
            test_end_ns = last_test_bar.timestamp_ns + 1

        train_slice = _slice_by_timestamp(data, train_start_ns, train_end_ns)
        test_slice  = _slice_by_timestamp(data, train_end_ns,   test_end_ns)

        # Skip windows where any symbol has no bars in either period.
        if any(len(v) == 0 for v in train_slice.values()):
            pos += test_size
            continue
        if any(len(v) == 0 for v in test_slice.values()):
            pos += test_size
            continue

        windows.append((train_slice, test_slice))
        pos += test_size

    return windows


# ============================================================================
# MODULE-LEVEL WORKER FUNCTIONS
# Must be top-level for ProcessPoolExecutor pickling.
# ============================================================================

def _backtest_worker(args: tuple) -> Optional[dict]:
    """
    Run a single backtest for one parameter combination.

    Args tuple:
        (strategy_factory, params, serialized_data, initial_capital, config)

    config is a BacktestConfig instance or None. When None, qc.run_backtest
    is used (library defaults: FixedPercentage(0.10), bars_per_year=252).

    Returns a plain dict (picklable) or None on failure.
    Failures are logged to stderr with the params and exception message.

    NOTE: numeric return values are in DECIMAL format (0.105 = 10.5%).
    """
    strategy_factory, params, serialized_data, initial_capital, config = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics

        data     = _deserialize_data(serialized_data)
        strategy = strategy_factory(**params)

        if config is not None:
            results = _run_configured_backtest(strategy, data, initial_capital, config)
        else:
            results = qc.run_backtest(strategy, data, initial_capital)

        equity_curve = np.array(results.equity_curve)
        if len(equity_curve) == 0:
            return None

        metrics = calculate_all_metrics(
            equity_curve,
            trade_pnls=results.trade_pnls if results.trade_pnls else None,
        )

        return {
            'params':       params,
            'sharpe_ratio': metrics.sharpe_ratio,
            'total_return': metrics.total_return / 100.0,  # pct -> decimal
            'max_drawdown': metrics.max_drawdown / 100.0,  # pct -> decimal
            'num_trades':   metrics.total_trades,
            'final_value':  results.final_value,
        }
    except Exception as exc:
        print(
            f"[walk_forward] _backtest_worker failed for params={params}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _window_worker(args: tuple) -> Optional[dict]:
    """
    Process a single walk-forward window: optimise on in-sample data, then
    evaluate the best parameters on the out-of-sample period.

    Args tuple:
        (strategy_factory, param_grid_dict, train_raw, test_raw,
         initial_capital, metric, config)

    config is a BacktestConfig instance or None.

    Returns a dict with keys: best_params, in_sample, out_of_sample.
    out_of_sample includes 'equity_curve' (list) and 'trade_pnls' (list)
    for stitching the combined curve and trade log in the caller.
    Failures are logged to stderr.
    """
    strategy_factory, param_grid_dict, train_raw, test_raw, \
        initial_capital, metric, config = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics
        from quantcore.walk_forward import (
            ParameterGrid, _deserialize_data, _serialize_data_plain,
            _backtest_worker, _run_configured_backtest,
        )

        train_data = _deserialize_data(train_raw)
        test_data  = _deserialize_data(test_raw)

        # In-sample optimisation (sequential inside the worker).
        grid        = ParameterGrid(param_grid_dict)
        best_result = None

        for params in grid:
            if 'fast_period' in params and 'slow_period' in params:
                if params['fast_period'] >= params['slow_period']:
                    continue

            result = _backtest_worker(
                (strategy_factory, params,
                 _serialize_data_plain(train_data), initial_capital, config)
            )
            if result is None:
                continue
            if best_result is None or result[metric] > best_result[metric]:
                best_result = result

        if best_result is None:
            return None

        best_params = best_result['params']

        # Out-of-sample evaluation with the best in-sample parameters.
        strategy = strategy_factory(**best_params)
        if config is not None:
            oos_results = _run_configured_backtest(
                strategy, test_data, initial_capital, config
            )
        else:
            oos_results = qc.run_backtest(strategy, test_data, initial_capital)

        equity_curve = np.array(oos_results.equity_curve)
        if len(equity_curve) > 0:
            m = calculate_all_metrics(
                equity_curve,
                trade_pnls=oos_results.trade_pnls if oos_results.trade_pnls else None,
            )
            oos = {
                'sharpe_ratio': m.sharpe_ratio,
                'total_return': m.total_return / 100.0,  # pct -> decimal
                'max_drawdown': m.max_drawdown / 100.0,  # pct -> decimal
                'num_trades':   m.total_trades,
                'final_value':  oos_results.final_value,
                'equity_curve': equity_curve.tolist(),
                'trade_pnls':   oos_results.trade_pnls or [],
            }
        else:
            oos = {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'num_trades':   0,
                'final_value':  initial_capital,
                'equity_curve': [],
                'trade_pnls':   [],
            }

        return {
            'best_params':   best_params,
            'in_sample':     best_result,
            'out_of_sample': oos,
        }

    except Exception as exc:
        print(
            f"[walk_forward] _window_worker failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OptimizationResult:
    """
    Single parameter combination result.

    NOTE: total_return and max_drawdown are in DECIMAL format:
      - total_return=0.105 means 10.5% return
      - max_drawdown=-0.05 means -5% drawdown
    """
    params:       Dict[str, Any]
    sharpe_ratio: float
    total_return: float  # Decimal format (0.105 = 10.5%)
    max_drawdown: float  # Decimal format (-0.05 = -5%)
    num_trades:   int
    final_value:  float

    def __repr__(self):
        return (f"OptimizationResult(sharpe={self.sharpe_ratio:.2f}, "
                f"return={self.total_return:.2%})")

    @property
    def total_return_pct(self) -> float:
        """Return as percentage (10.5 instead of 0.105)"""
        return self.total_return * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        """Drawdown as percentage (-5.0 instead of -0.05)"""
        return self.max_drawdown * 100.0


@dataclass
class WalkForwardResult:
    """
    Results from a walk-forward analysis run.

    Fields
    ------
    in_sample_results
        Best in-sample OptimizationResult per window.
    out_of_sample_results
        Per-window OOS metrics dict. Keys: sharpe_ratio, total_return
        (decimal), max_drawdown (decimal), num_trades, final_value.
    best_params_per_window
        The parameter combination selected in each window.
    combined_equity_curve
        Chained OOS equity curve across all windows. Each segment is
        rescaled to begin from the end of the previous one.
    combined_trade_pnls
        All trade PnLs from the OOS periods, concatenated in window order.
        Pass to calculate_all_metrics(ec, trade_pnls=wfa_result.combined_trade_pnls)
        to get accurate win rate, profit factor, and avg trade metrics on
        the combined equity curve.
    overall_metrics
        Aggregate stats: sharpe_ratio, total_return, max_drawdown, num_windows.
    """
    in_sample_results:      List[OptimizationResult]
    out_of_sample_results:  List[Dict[str, Any]]
    best_params_per_window: List[Dict[str, Any]]
    combined_equity_curve:  np.ndarray
    combined_trade_pnls:    List[float]
    overall_metrics:        Dict[str, float]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  Walk-Forward Analysis Results",
            "=" * 70,
            "",
            f"Number of Windows:    {len(self.in_sample_results)}",
            f"Overall Sharpe Ratio: {self.overall_metrics['sharpe_ratio']:.2f}",
            f"Overall Return:       {self.overall_metrics['total_return']:.2%}",
            f"Overall Max Drawdown: {self.overall_metrics['max_drawdown']:.2%}",
            "",
            "Per-Window Performance:",
            "-" * 70,
            ]
        for i, oos in enumerate(self.out_of_sample_results):
            lines.append(f"Window {i + 1}:")
            lines.append(f"  Best Params: {self.best_params_per_window[i]}")
            lines.append(f"  OOS Sharpe:  {oos['sharpe_ratio']:.2f}")
            lines.append(f"  OOS Return:  {oos['total_return']:.2%}")
            lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# PARAMETER GRID
# ============================================================================

class ParameterGrid:
    """Generate the Cartesian product of a parameter space."""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid   = param_grid
        self.param_names  = list(param_grid.keys())
        self.param_values = [param_grid[name] for name in self.param_names]

    def __iter__(self):
        for values in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __len__(self):
        return int(np.prod([len(v) for v in self.param_values]))


# ============================================================================
# GRID SEARCH OPTIMIZER
# ============================================================================

class GridSearchOptimizer:
    """
    Exhaustive grid search over a parameter space with optional parallelism.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable (a class or module-level function, not a lambda).
    param_grid : dict
        Maps parameter names to lists of candidate values.
    metric : str
        Optimisation objective: 'sharpe_ratio', 'total_return', or 'calmar_ratio'.
    n_jobs : int
        Number of worker processes. -1 uses all cores. 1 is sequential (default).
    backtest_config : BacktestConfig, optional
        Engine configuration applied to every worker backtest. When None,
        the library defaults are used (FixedPercentage(0.10), bars_per_year=252).
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid:       Dict[str, List[Any]],
            metric:           str                      = 'sharpe_ratio',
            n_jobs:           int                      = 1,
            backtest_config:  Optional[BacktestConfig] = None,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid       = ParameterGrid(param_grid)
        self.metric           = metric
        self.n_jobs           = n_jobs
        self.backtest_config  = backtest_config
        self.results:         List[OptimizationResult] = []

    def optimize(
            self,
            data:            Dict[str, list],
            initial_capital: float = 100000.0,
            verbose:         bool  = True,
    ) -> List[OptimizationResult]:
        """Run grid search and return results sorted by the chosen metric."""
        serialized = _serialize_data(data)

        tasks = []
        for params in self.param_grid:
            if 'fast_period' in params and 'slow_period' in params:
                if params['fast_period'] >= params['slow_period']:
                    continue
            tasks.append(
                (self.strategy_factory, params, serialized,
                 initial_capital, self.backtest_config)
            )

        use_parallel = self.n_jobs != 1 and len(tasks) >= MIN_PARALLEL_TASKS
        raw_results: List[Optional[dict]] = []

        if use_parallel:
            workers = None if self.n_jobs == -1 else self.n_jobs
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_backtest_worker, t): t for t in tasks}
                for future in as_completed(futures):
                    raw_results.append(future.result())
        else:
            for i, task in enumerate(tasks):
                if verbose:
                    print(f"\r  Optimising {i + 1}/{len(tasks)}...", end="", flush=True)
                raw_results.append(_backtest_worker(task))
            if verbose and tasks:
                print()

        self.results = [
            OptimizationResult(**r)
            for r in raw_results
            if r is not None
        ]
        self.results.sort(key=lambda x: getattr(x, self.metric), reverse=True)
        return self.results

    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        return self.results[:n]

    def get_results_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = r.params.copy()
            row.update({
                'sharpe_ratio':    r.sharpe_ratio,
                'total_return':    r.total_return,       # Decimal format
                'total_return_pct': r.total_return_pct,  # Percentage for convenience
                'max_drawdown':    r.max_drawdown,       # Decimal format
                'max_drawdown_pct': r.max_drawdown_pct,  # Percentage for convenience
                'num_trades':      r.num_trades,
            })
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# WALK-FORWARD ANALYSER
# ============================================================================

class WalkForwardAnalyzer:
    """
    Rolling walk-forward analysis for out-of-sample strategy validation.

    Supports both single-asset and multi-asset data.

    For each window:
      1. Optimise parameters on the in-sample training period.
      2. Evaluate the best parameters on the following out-of-sample period.
      3. Advance the window by test_size bars and repeat.

    The out-of-sample equity curves from each window are chained into a single
    continuous combined_equity_curve. Each segment is rescaled to begin from
    the ending value of the previous one, so the combined curve reflects
    compounded performance across all windows.

    Trade PnLs from all OOS periods are collected into combined_trade_pnls on
    WalkForwardResult, enabling accurate trade-level metrics (win rate, profit
    factor, avg win/loss) on the combined curve via calculate_all_metrics.

    Window construction
    -------------------
    Windows are built using the first symbol's bar series as the reference
    timeline. train_size and test_size are measured in bars of that reference
    series. Bars for all other symbols are selected by timestamp inclusion
    within [window_start, window_end), so minor gaps in secondary symbols
    do not corrupt other windows.

    A window is skipped if any symbol has zero bars in either the training
    or test period within that timestamp range.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable when n_jobs != 1.
    param_grid : dict
    train_size : int
        Number of bars (in the reference symbol) in each training window.
    test_size : int
        Number of bars (in the reference symbol) in each test window.
        Also the step size between consecutive windows.
    metric : str
        In-sample optimisation metric.
    n_jobs : int
        Worker processes for parallel window execution. -1 = all cores.
    backtest_config : BacktestConfig, optional
        Engine configuration applied to every worker backtest. When None,
        the library defaults are used (FixedPercentage(0.10), bars_per_year=252).
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid:       Dict[str, List[Any]],
            train_size:       int                      = 252,
            test_size:        int                      = 63,
            metric:           str                      = 'sharpe_ratio',
            n_jobs:           int                      = 1,
            backtest_config:  Optional[BacktestConfig] = None,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid       = param_grid
        self.train_size       = train_size
        self.test_size        = test_size
        self.metric           = metric
        self.n_jobs           = n_jobs
        self.backtest_config  = backtest_config

    def analyze(
            self,
            data:            Dict[str, list],
            initial_capital: float = 100000.0,
            verbose:         bool  = False,
    ) -> WalkForwardResult:
        """Run walk-forward analysis and return aggregated results."""
        ref_symbol = list(data.keys())[0]
        total      = len(data[ref_symbol])

        if total < self.train_size + self.test_size:
            raise ValueError(
                f"Not enough data: need {self.train_size + self.test_size} bars "
                f"in '{ref_symbol}', have {total}"
            )

        windows = _build_windows(data, self.train_size, self.test_size)

        if not windows:
            raise ValueError(
                "No valid walk-forward windows could be built. "
                "Check that all symbols have sufficient overlapping data."
            )

        tasks = [
            (self.strategy_factory, self.param_grid,
             _serialize_data(train_slice), _serialize_data(test_slice),
             initial_capital, self.metric, self.backtest_config)
            for train_slice, test_slice in windows
        ]

        use_parallel = self.n_jobs != 1 and len(tasks) >= MIN_PARALLEL_TASKS
        raw_results: List[Optional[dict]] = []

        if use_parallel:
            workers = None if self.n_jobs == -1 else self.n_jobs
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Preserve window order by indexing futures.
                futures = {executor.submit(_window_worker, t): i
                           for i, t in enumerate(tasks)}
                ordered = [None] * len(tasks)
                for future, idx in futures.items():
                    ordered[idx] = future.result()
                raw_results = ordered
        else:
            for i, task in enumerate(tasks):
                if verbose:
                    print(f"  Walk-forward window {i + 1}/{len(tasks)}...")
                raw_results.append(_window_worker(task))

        in_sample_results      = []
        out_of_sample_results  = []
        best_params_per_window = []
        equity_segments        = []
        all_trade_pnls         = []

        for r in raw_results:
            if r is None:
                continue

            in_sample_results.append(OptimizationResult(**r['in_sample']))
            best_params_per_window.append(r['best_params'])

            oos = r['out_of_sample']
            # Extract fields that should not live in the stored metrics dict.
            ec_list    = oos.pop('equity_curve', [])
            trade_pnls = oos.pop('trade_pnls',   [])

            out_of_sample_results.append(oos)

            if ec_list:
                equity_segments.append(np.array(ec_list))
            all_trade_pnls.extend(trade_pnls)

        if not out_of_sample_results:
            raise ValueError(
                "All walk-forward windows failed. Check data size and parameter grid."
            )

        # Chain OOS equity curves: rescale each segment to start from the
        # last value of the previous one so the curve is continuous.
        if equity_segments:
            chained = [equity_segments[0]]
            for seg in equity_segments[1:]:
                if len(seg) == 0 or seg[0] == 0:
                    continue
                scale_factor = chained[-1][-1] / seg[0]
                chained.append(seg * scale_factor)
            combined_equity_curve = np.concatenate(chained)
        else:
            combined_equity_curve = np.array([])

        overall_metrics = {
            'sharpe_ratio': float(np.mean([r['sharpe_ratio'] for r in out_of_sample_results])),
            'total_return': float(np.mean([r['total_return'] for r in out_of_sample_results])),
            'max_drawdown': float(np.mean([r['max_drawdown'] for r in out_of_sample_results])),
            'num_windows':  len(out_of_sample_results),
        }

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            best_params_per_window=best_params_per_window,
            combined_equity_curve=combined_equity_curve,
            combined_trade_pnls=all_trade_pnls,
            overall_metrics=overall_metrics,
        )

    def plot_stability(self, result: WalkForwardResult):
        """Plot how the optimal parameters change across windows."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return None

        if not result.best_params_per_window:
            return None

        param_names = list(result.best_params_per_window[0].keys())
        n_params    = len(param_names)
        fig, axes   = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
        if n_params == 1:
            axes = [axes]

        windows = range(1, len(result.best_params_per_window) + 1)
        for i, name in enumerate(param_names):
            values = [p[name] for p in result.best_params_per_window]
            axes[i].plot(windows, values, marker='o', linewidth=2)
            axes[i].set_xlabel('Window')
            axes[i].set_ylabel(name)
            axes[i].set_title(f'{name} Stability Across Windows')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ============================================================================
# MONTE CARLO VALIDATION
# ============================================================================

def monte_carlo_validation(
        strategy_factory:     Callable,
        params:               Dict[str, Any],
        data:                 Dict[str, list],
        n_simulations:        int                      = 1000,
        initial_capital:      float                    = 100000.0,
        method:               str                      = 'bootstrap',
        n_jobs:               int                      = 1,
        backtest_config:      Optional[BacktestConfig] = None,
        return_equity_curves: bool                     = False,
) -> Dict[str, Any]:
    """
    Assess strategy robustness via repeated resampling of the price series.

    Supports both single-asset and multi-asset data. For multi-asset data,
    the same random indices are applied to all symbols simultaneously
    (synchronized resampling), preserving cross-asset correlation structure.
    Resampling each symbol independently would destroy those correlations
    and produce unrealistic scenarios.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable when n_jobs != 1 and return_equity_curves is False.
    params : dict
        Fixed strategy parameters.
    data : dict
        Dict mapping symbol to List[BarData]. All symbols must have the same
        number of bars; bar counts are validated before running.
    n_simulations : int
        Number of resampled scenarios to run.
    initial_capital : float
    method : str
        'bootstrap': resample log-return indices with replacement.
        'shuffle':   randomly permute log-return indices.
    n_jobs : int
        Worker processes. -1 = all cores. Ignored when return_equity_curves
        is True (equity curves are always collected in-process).
    backtest_config : BacktestConfig, optional
        Engine configuration applied to every simulation. When None, the
        library defaults are used (FixedPercentage(0.10), bars_per_year=252).
    return_equity_curves : bool
        When True, the returned dict includes an 'equity_curves' key
        containing a list of np.ndarray, one per successful simulation.
        Simulations always run in-process when this is True (equity curve
        arrays are not efficient to pickle across process boundaries).
        Defaults to False to avoid storing large arrays unnecessarily.

    Returns
    -------
    dict with:
        sharpe_ratios  : np.ndarray
        returns        : np.ndarray  (decimal format, 0.105 = 10.5%)
        drawdowns      : np.ndarray  (decimal format, -0.05 = -5%)
        equity_curves  : List[np.ndarray]
                         Only present when return_equity_curves=True.

    Raises
    ------
    ValueError
        If symbols have different bar counts or method is not recognised.
    """
    symbols    = list(data.keys())
    bar_counts = {sym: len(data[sym]) for sym in symbols}

    if len(set(bar_counts.values())) > 1:
        detail = ", ".join(f"{sym}: {n}" for sym, n in bar_counts.items())
        raise ValueError(
            f"monte_carlo_validation requires all symbols to have the same number "
            f"of bars for synchronized resampling. Got: {detail}. "
            f"Align your series to a common date range before running."
        )

    if method not in ('bootstrap', 'shuffle'):
        raise ValueError(f"Unknown method: '{method}'. Use 'bootstrap' or 'shuffle'.")

    n_bars = bar_counts[symbols[0]]

    def _make_sample(seed: int) -> Dict[str, list]:
        """
        Resample all symbols using the same random index sequence, preserving
        cross-asset correlation structure.
        """
        from quantcore._core import BarData
        rng = np.random.default_rng(seed)

        if method == 'bootstrap':
            indices = rng.integers(0, n_bars - 1, size=n_bars - 1)
        else:  # shuffle
            indices = rng.permutation(n_bars - 1)

        sampled: Dict[str, list] = {}
        for sym in symbols:
            bars   = data[sym]
            closes = np.array([b.close for b in bars])

            log_returns         = np.diff(np.log(closes))
            sampled_log_returns = log_returns[indices]
            new_closes          = closes[0] * np.exp(
                np.concatenate([[0.0], np.cumsum(sampled_log_returns)])
            )

            new_bars = []
            for i, bar in enumerate(bars):
                scale     = new_closes[i] / bar.close if bar.close != 0.0 else 1.0
                new_open  = bar.open  * scale
                new_high  = bar.high  * scale
                new_low   = bar.low   * scale
                new_close = new_closes[i]
                # Clamp to guarantee BarData validation constraints.
                new_high = max(new_high, new_open, new_close)
                new_low  = min(new_low,  new_open, new_close)
                new_bars.append(BarData(
                    bar.symbol, bar.timestamp_ns,
                    new_open, new_high, new_low, new_close, bar.volume,
                ))
            sampled[sym] = new_bars

        return sampled

    # When equity curves are requested, run in-process: numpy arrays are not
    # efficient to pickle across process boundaries at scale.
    if return_equity_curves:
        raw_results = _mc_collect_with_curves(
            strategy_factory, params, initial_capital,
            backtest_config, _make_sample, n_simulations,
        )
    else:
        tasks = [
            (strategy_factory, params,
             _serialize_data(_make_sample(seed)), initial_capital, backtest_config)
            for seed in range(n_simulations)
        ]

        use_parallel = n_jobs != 1 and n_simulations >= MIN_PARALLEL_TASKS
        raw_results_worker: List[Optional[dict]] = []

        if use_parallel:
            workers = None if n_jobs == -1 else n_jobs
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for r in executor.map(_backtest_worker, tasks, chunksize=10):
                    raw_results_worker.append(r)
        else:
            for task in tasks:
                raw_results_worker.append(_backtest_worker(task))

        raw_results = [r for r in raw_results_worker if r is not None]

    if not raw_results:
        empty: np.ndarray = np.array([])
        out: Dict[str, Any] = {
            'sharpe_ratios': empty,
            'returns':       empty,
            'drawdowns':     empty,
        }
        if return_equity_curves:
            out['equity_curves'] = []
        return out

    out = {
        'sharpe_ratios': np.array([r['sharpe_ratio'] for r in raw_results]),
        'returns':       np.array([r['total_return'] for r in raw_results]),
        'drawdowns':     np.array([r['max_drawdown'] for r in raw_results]),
    }
    if return_equity_curves:
        out['equity_curves'] = [r['equity_curve'] for r in raw_results]

    return out


def _mc_collect_with_curves(
        strategy_factory: Callable,
        params:           Dict[str, Any],
        initial_capital:  float,
        config:           Optional[BacktestConfig],
        make_sample_fn:   Callable[[int], Dict[str, list]],
        n_simulations:    int,
) -> List[dict]:
    """
    Run Monte Carlo simulations in-process and return results including equity curves.

    Called only when return_equity_curves=True. Running in-process avoids
    pickling numpy arrays across process boundaries, which is inefficient at
    the scale of fan charts (typically 50–200 simulations).
    """
    import quantcore as qc
    from quantcore.analytics import calculate_all_metrics

    results = []
    for seed in range(n_simulations):
        try:
            sim_data = make_sample_fn(seed)
            strategy = strategy_factory(**params)

            if config is not None:
                bt = _run_configured_backtest(strategy, sim_data, initial_capital, config)
            else:
                bt = qc.run_backtest(strategy, sim_data, initial_capital)

            ec = np.array(bt.equity_curve)
            if len(ec) == 0:
                continue

            m = calculate_all_metrics(
                ec,
                trade_pnls=bt.trade_pnls if bt.trade_pnls else None,
            )
            results.append({
                'sharpe_ratio': m.sharpe_ratio,
                'total_return': m.total_return / 100.0,
                'max_drawdown': m.max_drawdown / 100.0,
                'equity_curve': ec,
            })
        except Exception as exc:
            print(
                f"[walk_forward] _mc_collect_with_curves failed for seed={seed}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return results