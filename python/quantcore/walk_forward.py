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
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
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
# MODULE-LEVEL WORKER FUNCTIONS
# Must be top-level for ProcessPoolExecutor pickling.
# ============================================================================

def _backtest_worker(args: tuple) -> Optional[dict]:
    """
    Run a single backtest for one parameter combination.

    Arguments are packed into a tuple to satisfy ProcessPoolExecutor's
    single-argument constraint:
        (strategy_factory, params, serialized_data, initial_capital)

    Returns a plain dict (picklable) or None on failure.

    NOTE: Returns are in DECIMAL format (0.105 = 10.5%)
    """
    strategy_factory, params, serialized_data, initial_capital = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics

        data     = _deserialize_data(serialized_data)
        strategy = strategy_factory(**params)
        results  = qc.run_backtest(strategy, data, initial_capital)

        equity_curve = np.array(results.equity_curve)
        if len(equity_curve) == 0:
            return None

        metrics = calculate_all_metrics(equity_curve)

        return {
            'params':       params,
            'sharpe_ratio': metrics.sharpe_ratio,
            'total_return': metrics.total_return / 100.0,  # pct -> decimal
            'max_drawdown': metrics.max_drawdown / 100.0,  # pct -> decimal
            'num_trades':   metrics.total_trades,
            'final_value':  results.final_value,
        }
    except Exception:
        return None


def _window_worker(args: tuple) -> Optional[dict]:
    """
    Process a single walk-forward window: optimise on in-sample data, then
    evaluate the best parameters on the out-of-sample period.

    Args tuple:
        (strategy_factory, param_grid_dict, train_raw, test_raw,
         initial_capital, metric)

    Returns a dict with keys: best_params, in_sample, out_of_sample.
    out_of_sample includes an 'equity_curve' list for stitching the combined
    equity curve in the caller.
    """
    strategy_factory, param_grid_dict, train_raw, test_raw, initial_capital, metric = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics
        from quantcore.walk_forward import (
            ParameterGrid, _deserialize_data, _serialize_data_plain, _backtest_worker
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
                 _serialize_data_plain(train_data), initial_capital)
            )
            if result is None:
                continue
            if best_result is None or result[metric] > best_result[metric]:
                best_result = result

        if best_result is None:
            return None

        best_params = best_result['params']

        # Out-of-sample evaluation with the best in-sample parameters.
        strategy    = strategy_factory(**best_params)
        oos_results = qc.run_backtest(strategy, test_data, initial_capital)

        equity_curve = np.array(oos_results.equity_curve)
        if len(equity_curve) > 0:
            m   = calculate_all_metrics(equity_curve)
            oos = {
                'sharpe_ratio': m.sharpe_ratio,
                'total_return': m.total_return / 100.0,  # pct -> decimal
                'max_drawdown': m.max_drawdown / 100.0,  # pct -> decimal
                'num_trades':   m.total_trades,
                'final_value':  oos_results.final_value,
                'equity_curve': equity_curve.tolist(),
            }
        else:
            oos = {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'num_trades':   0,
                'final_value':  initial_capital,
                'equity_curve': [],
            }

        return {
            'best_params':    best_params,
            'in_sample':      best_result,
            'out_of_sample':  oos,
        }

    except Exception:
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
    """Results from a walk-forward analysis run."""
    in_sample_results:      List[OptimizationResult]
    out_of_sample_results:  List[Dict[str, Any]]
    best_params_per_window: List[Dict[str, Any]]
    # Chained out-of-sample equity curves spanning all windows. Each segment
    # is rescaled to start from the ending value of the previous segment so
    # that the curve is continuous and comparable to a single-run equity curve.
    combined_equity_curve:  np.ndarray
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
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid:       Dict[str, List[Any]],
            metric:           str = 'sharpe_ratio',
            n_jobs:           int = 1,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid       = ParameterGrid(param_grid)
        self.metric           = metric
        self.n_jobs           = n_jobs
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
            tasks.append((self.strategy_factory, params, serialized, initial_capital))

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
                'sharpe_ratio': r.sharpe_ratio,
                'total_return': r.total_return,  # Decimal format
                'total_return_pct': r.total_return_pct,  # Percentage for convenience
                'max_drawdown': r.max_drawdown,  # Decimal format
                'max_drawdown_pct': r.max_drawdown_pct,  # Percentage for convenience
                'num_trades':   r.num_trades,
            })
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# WALK-FORWARD ANALYSER
# ============================================================================

class WalkForwardAnalyzer:
    """
    Rolling walk-forward analysis for out-of-sample strategy validation.

    For each window:
      1. Optimise parameters on the in-sample training period.
      2. Evaluate the best parameters on the following out-of-sample period.
      3. Advance the window by test_size bars and repeat.

    The out-of-sample equity curves from each window are chained into a single
    continuous combined_equity_curve. Each segment is rescaled to begin from
    the ending value of the previous one, so the combined curve reflects
    compounded performance across all windows.

    Note: Only single-asset data is supported. If a multi-symbol dict is
    passed, only the first symbol is used and a warning is raised.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable when n_jobs != 1.
    param_grid : dict
    train_size : int
        Number of bars in each training window.
    test_size : int
        Number of bars in each test window (also the step size).
    metric : str
        In-sample optimisation metric.
    n_jobs : int
        Worker processes for parallel window execution. -1 = all cores.
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid:       Dict[str, List[Any]],
            train_size:       int = 252,
            test_size:        int = 63,
            metric:           str = 'sharpe_ratio',
            n_jobs:           int = 1,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid       = param_grid
        self.train_size       = train_size
        self.test_size        = test_size
        self.metric           = metric
        self.n_jobs           = n_jobs

    def analyze(
            self,
            data:            Dict[str, list],
            initial_capital: float = 100000.0,
            verbose:         bool  = False,
    ) -> WalkForwardResult:
        """Run walk-forward analysis and return aggregated results."""
        # Warn loudly rather than silently dropping symbols — callers should
        # know their data is being partially ignored.
        if len(data) > 1:
            ignored = list(data.keys())[1:]
            warnings.warn(
                f"WalkForwardAnalyzer only supports single-asset data. "
                f"Only '{list(data.keys())[0]}' will be used; "
                f"ignoring: {ignored}",
                stacklevel=2,
            )

        symbol = list(data.keys())[0]
        bars   = data[symbol]
        total  = len(bars)

        if total < self.train_size + self.test_size:
            raise ValueError(
                f"Not enough data: need {self.train_size + self.test_size}, have {total}"
            )

        windows = []
        pos     = 0
        while pos + self.train_size + self.test_size <= total:
            train_bars = {symbol: bars[pos : pos + self.train_size]}
            test_bars  = {symbol: bars[pos + self.train_size : pos + self.train_size + self.test_size]}
            windows.append((_serialize_data(train_bars), _serialize_data(test_bars)))
            pos += self.test_size

        if not windows:
            raise ValueError("No valid walk-forward windows.")

        tasks = [
            (self.strategy_factory, self.param_grid,
             train_raw, test_raw, initial_capital, self.metric)
            for train_raw, test_raw in windows
        ]

        use_parallel = self.n_jobs != 1 and len(tasks) >= MIN_PARALLEL_TASKS
        raw_results: List[Optional[dict]] = []

        if use_parallel:
            workers = None if self.n_jobs == -1 else self.n_jobs
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_window_worker, t): i for i, t in enumerate(tasks)}
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

        for r in raw_results:
            if r is None:
                continue

            in_sample_results.append(OptimizationResult(**r['in_sample']))
            best_params_per_window.append(r['best_params'])

            oos = r['out_of_sample']
            # Separate the equity curve from the metrics dict before storing.
            ec_list = oos.pop('equity_curve', [])
            out_of_sample_results.append(oos)

            if ec_list:
                equity_segments.append(np.array(ec_list))

        if not out_of_sample_results:
            raise ValueError(
                "All walk-forward windows failed. Check data size and parameter grid."
            )

        # Chain out-of-sample equity curves. Each segment is rescaled so that
        # its first value matches the last value of the previous segment,
        # producing a smooth compounded equity curve.
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
            'total_return': float(np.mean([r['total_return']  for r in out_of_sample_results])),
            'max_drawdown': float(np.mean([r['max_drawdown']  for r in out_of_sample_results])),
            'num_windows':  len(out_of_sample_results),
        }

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            best_params_per_window=best_params_per_window,
            combined_equity_curve=combined_equity_curve,
            overall_metrics=overall_metrics,
        )

    def plot_stability(self, result: WalkForwardResult):
        """Plot how the optimal parameters change across windows."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        if not result.best_params_per_window:
            return

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
        strategy_factory: Callable,
        params:           Dict[str, Any],
        data:             Dict[str, list],
        n_simulations:    int  = 1000,
        initial_capital:  float = 100000.0,
        method:           str  = 'bootstrap',
        n_jobs:           int  = 1,
) -> Dict[str, np.ndarray]:
    """
    Assess strategy robustness via repeated resampling of the price series.

    Note: Only single-asset data is supported. If a multi-symbol dict is
    passed, only the first symbol is used and a warning is raised.

    Parameters
    ----------
    method : str
        'bootstrap' (sample with replacement) or 'shuffle' (permute returns).
    n_jobs : int
        Worker processes. -1 = all cores.

    Returns
    -------
    dict with arrays: sharpe_ratios, returns, drawdowns.
    NOTE: returns and drawdowns are in DECIMAL format (0.105 = 10.5%)
    """
    # Warn loudly rather than silently dropping symbols — callers should
    # know their data is being partially ignored.
    if len(data) > 1:
        ignored = list(data.keys())[1:]
        warnings.warn(
            f"monte_carlo_validation only supports single-asset data. "
            f"Only '{list(data.keys())[0]}' will be used; "
            f"ignoring: {ignored}",
            stacklevel=2,
        )

    symbol = list(data.keys())[0]
    bars   = data[symbol]

    def _make_sample(seed: int) -> Dict[str, list]:
        from quantcore._core import BarData
        rng = np.random.default_rng(seed)
        closes = np.array([bar.close for bar in bars])

        # compute log returns from the price series
        log_returns = np.diff(np.log(closes))

        if method == 'bootstrap':
            sampled_returns = rng.choice(log_returns, size=len(log_returns), replace=True)
        elif method == 'shuffle':
            sampled_returns = rng.permutation(log_returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reconstruct a synthetic price series anchored at the first close
        new_closes = closes[0] * np.exp(np.concatenate([[0], np.cumsum(sampled_returns)]))

        # rebuild bars scaling OHLC proportionally to preserve intra-bar structure
        new_bars = []
        for i, bar in enumerate(bars):
            scale = new_closes[i] / bar.close
            new_open  = bar.open  * scale
            new_high  = bar.high  * scale
            new_low   = bar.low   * scale
            new_close = new_closes[i]

            # clamp to guarantee BarData validation constraints are satisfied
            new_high  = max(new_high, new_open, new_close)
            new_low   = min(new_low,  new_open, new_close)

            new_bars.append(BarData(
                bar.symbol,
                bar.timestamp_ns,
                new_open,
                new_high,
                new_low,
                new_close,
                bar.volume,
            ))

        return {symbol: new_bars}
    tasks = [
        (strategy_factory, params,
         _serialize_data(_make_sample(seed)), initial_capital)
        for seed in range(n_simulations)
    ]

    use_parallel = n_jobs != 1 and n_simulations >= MIN_PARALLEL_TASKS
    raw_results: List[Optional[dict]] = []

    if use_parallel:
        workers = None if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_backtest_worker, tasks, chunksize=10):
                raw_results.append(result)
    else:
        for task in tasks:
            raw_results.append(_backtest_worker(task))

    valid = [r for r in raw_results if r is not None]

    return {
        'sharpe_ratios': np.array([r['sharpe_ratio'] for r in valid]),
        'returns':        np.array([r['total_return']  for r in valid]),  # Decimal format
        'drawdowns':      np.array([r['max_drawdown']  for r in valid]),  # Decimal format
    }