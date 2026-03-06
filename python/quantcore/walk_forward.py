"""
Walk-forward analysis and parameter optimization

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

  * BarData objects (C++ pybind11) are not picklable. Data is serialized to
    plain tuples before dispatch and reconstructed inside each worker.

  * Process spawn overhead is non-trivial (~0.5s per worker on first import).
    Parallelism only activates when the number of tasks exceeds
    MIN_PARALLEL_TASKS (default 4). Below that threshold the sequential
    path is used regardless of n_jobs.

  * On Windows, guard your entry point with if __name__ == '__main__'.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


# Minimum number of tasks before parallelism is worth the spawn overhead.
MIN_PARALLEL_TASKS = 4


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def _serialize_data(data: Dict[str, list]) -> Dict[str, list]:
    """
    Convert Dict[str, List[BarData]] to a picklable format.

    BarData objects are C++ pybind11 wrappers and cannot be pickled directly.
    We extract each field into a plain tuple so the dict can cross process
    boundaries cleanly.
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
    """Reconstruct BarData objects from serialized tuples inside a worker."""
    import quantcore as qc
    return {
        symbol: [qc.BarData(sym, ts, o, h, l, c, v) for sym, ts, o, h, l, c, v in bars]
        for symbol, bars in serialized.items()
    }


# ============================================================================
# MODULE-LEVEL WORKER FUNCTIONS
# (Must be top-level for ProcessPoolExecutor pickling.)
# ============================================================================

def _backtest_worker(args: tuple) -> Optional[dict]:
    """
    Run a single backtest for one parameter combination.

    Args come as a tuple to satisfy ProcessPoolExecutor's single-argument
    constraint: (strategy_factory, params, serialized_data, initial_capital).

    Returns a plain dict (picklable) or None on failure.
    """
    strategy_factory, params, serialized_data, initial_capital = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics

        data = _deserialize_data(serialized_data)
        strategy = strategy_factory(**params)
        results = qc.run_backtest(strategy, data, initial_capital)

        equity_curve = np.array(results.get('equity_curve', []))
        if len(equity_curve) == 0:
            return None

        metrics = calculate_all_metrics(equity_curve)

        return {
            'params': params,
            'sharpe_ratio': metrics.sharpe_ratio,
            'total_return': metrics.total_return / 100.0,
            'max_drawdown': metrics.max_drawdown / 100.0,
            'num_trades': metrics.total_trades,
            'final_value': results['final_value'],
        }
    except Exception:
        return None


def _window_worker(args: tuple) -> Optional[dict]:
    """
    Process a single walk-forward window: optimize on train, test on OOS.

    Args tuple: (strategy_factory, param_grid_dict, train_raw, test_raw,
                 initial_capital, metric)

    Returns a plain dict with keys: best_params, in_sample, out_of_sample.
    """
    strategy_factory, param_grid_dict, train_raw, test_raw, initial_capital, metric = args

    try:
        import numpy as np
        import quantcore as qc
        from quantcore.analytics import calculate_all_metrics
        from quantcore.walk_forward import (
            ParameterGrid, _deserialize_data, _backtest_worker
        )

        train_data = _deserialize_data(train_raw)
        test_data  = _deserialize_data(test_raw)

        # --- in-sample optimization (sequential inside the worker) ---
        grid = ParameterGrid(param_grid_dict)
        best_result = None

        for params in grid:
            # skip invalid combos
            if 'fast_period' in params and 'slow_period' in params:
                if params['fast_period'] >= params['slow_period']:
                    continue

            result = _backtest_worker(
                (strategy_factory, params, _serialize_data_plain(train_data), initial_capital)
            )
            if result is None:
                continue

            if best_result is None or result[metric] > best_result[metric]:
                best_result = result

        if best_result is None:
            return None

        best_params = best_result['params']

        # --- out-of-sample test with best params ---
        strategy = strategy_factory(**best_params)
        oos_results = qc.run_backtest(strategy, test_data, initial_capital)

        equity_curve = np.array(oos_results.get('equity_curve', []))
        if len(equity_curve) > 0:
            m = calculate_all_metrics(equity_curve)
            oos = {
                'sharpe_ratio': m.sharpe_ratio,
                'total_return': m.total_return / 100.0,
                'max_drawdown': m.max_drawdown / 100.0,
                'num_trades': m.total_trades,
                'final_value': oos_results['final_value'],
            }
        else:
            oos = {
                'sharpe_ratio': 0.0, 'total_return': 0.0,
                'max_drawdown': 0.0, 'num_trades': 0,
                'final_value': initial_capital,
            }

        return {
            'best_params': best_params,
            'in_sample': best_result,
            'out_of_sample': oos,
        }

    except Exception:
        return None


def _serialize_data_plain(data: Dict[str, list]) -> Dict[str, list]:
    """Serialize already-reconstructed BarData back to tuples for nested workers."""
    return {
        symbol: [
            (bar.symbol, bar.timestamp_ns,
             bar.open, bar.high, bar.low, bar.close, bar.volume)
            for bar in bars
        ]
        for symbol, bars in data.items()
    }


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OptimizationResult:
    """Single parameter combination result"""
    params: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    num_trades: int
    final_value: float

    def __repr__(self):
        return (f"OptimizationResult(sharpe={self.sharpe_ratio:.2f}, "
                f"return={self.total_return:.2%})")


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    in_sample_results: List[OptimizationResult]
    out_of_sample_results: List[Dict[str, Any]]
    best_params_per_window: List[Dict[str, Any]]
    combined_equity_curve: np.ndarray
    overall_metrics: Dict[str, float]

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
    """Generate all parameter combinations for grid search"""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())
        self.param_values = [param_grid[name] for name in self.param_names]

    def __iter__(self):
        for values in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __len__(self):
        return int(np.prod([len(vals) for vals in self.param_values]))


# ============================================================================
# GRID SEARCH OPTIMIZER
# ============================================================================

class GridSearchOptimizer:
    """
    Grid search over parameter space with optional parallelism.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable (a class or module-level function, not a lambda).
    param_grid : dict
        Maps parameter names to lists of values to try.
    metric : str
        Metric to optimise: 'sharpe_ratio', 'total_return', or 'calmar_ratio'.
    n_jobs : int
        Number of worker processes. -1 uses all available cores. 1 is
        sequential (default). Parallelism only activates when the number of
        valid parameter combinations exceeds MIN_PARALLEL_TASKS.
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid: Dict[str, List[Any]],
            metric: str = 'sharpe_ratio',
            n_jobs: int = 1,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = ParameterGrid(param_grid)
        self.metric = metric
        self.n_jobs = n_jobs
        self.results: List[OptimizationResult] = []

    def optimize(
            self,
            data: Dict[str, list],
            initial_capital: float = 100000.0,
            verbose: bool = True,
    ) -> List[OptimizationResult]:
        """
        Run grid search and return results sorted by the chosen metric.

        Parameters
        ----------
        data : dict
            Symbol → List[BarData].
        initial_capital : float
        verbose : bool
            Print progress when running sequentially.
        """
        serialized = _serialize_data(data)

        # Build valid task list upfront so we know whether parallelism pays off.
        tasks = []
        for params in self.param_grid:
            if 'fast_period' in params and 'slow_period' in params:
                if params['fast_period'] >= params['slow_period']:
                    continue
            tasks.append((self.strategy_factory, params, serialized, initial_capital))

        use_parallel = (
                self.n_jobs != 1
                and len(tasks) >= MIN_PARALLEL_TASKS
        )

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
                    print(f"\r  Optimizing {i + 1}/{len(tasks)}...", end="", flush=True)
                raw_results.append(_backtest_worker(task))
            if verbose and tasks:
                print()  # newline after progress

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
            row['sharpe_ratio'] = r.sharpe_ratio
            row['total_return']  = r.total_return
            row['max_drawdown']  = r.max_drawdown
            row['num_trades']    = r.num_trades
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# WALK-FORWARD ANALYZER
# ============================================================================

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Splits data into rolling train/test windows:
      1. Optimize on in-sample window to find best params.
      2. Test with those params on the following out-of-sample window.
      3. Advance the window by test_size and repeat.

    Parameters
    ----------
    strategy_factory : callable
        Must be picklable when n_jobs != 1.
    param_grid : dict
    train_size : int
        Number of bars in each training window.
    test_size : int
        Number of bars in each test window. The window advances by this amount.
    metric : str
        In-sample optimisation metric.
    n_jobs : int
        Number of worker processes for parallel window processing.
        Each window runs its own sequential grid search internally.
        -1 uses all available cores.

    Note: anchored walk-forward (expanding training window) is not implemented.
    """

    def __init__(
            self,
            strategy_factory: Callable,
            param_grid: Dict[str, List[Any]],
            train_size: int = 252,
            test_size: int = 63,
            metric: str = 'sharpe_ratio',
            n_jobs: int = 1,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid  # keep as raw dict for serialization
        self.train_size = train_size
        self.test_size  = test_size
        self.metric     = metric
        self.n_jobs     = n_jobs

    def analyze(
            self,
            data: Dict[str, list],
            initial_capital: float = 100000.0,
            verbose: bool = False,
    ) -> WalkForwardResult:
        """Run walk-forward analysis and return aggregated results."""
        symbol = list(data.keys())[0]
        bars   = data[symbol]
        total  = len(bars)

        if total < self.train_size + self.test_size:
            raise ValueError(
                f"Not enough data: need {self.train_size + self.test_size}, "
                f"have {total}"
            )

        # Build window slices.
        windows = []
        pos = 0
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

        use_parallel = (
                self.n_jobs != 1
                and len(tasks) >= MIN_PARALLEL_TASKS
        )

        raw_results: List[Optional[dict]] = []

        if use_parallel:
            workers = None if self.n_jobs == -1 else self.n_jobs
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_window_worker, t): i
                           for i, t in enumerate(tasks)}
                # preserve window order
                ordered = [None] * len(tasks)
                for future, idx in futures.items():
                    ordered[idx] = future.result()
                raw_results = ordered
        else:
            for i, task in enumerate(tasks):
                if verbose:
                    print(f"  Walk-forward window {i + 1}/{len(tasks)}...")
                raw_results.append(_window_worker(task))

        # Unpack results, skipping any failed windows.
        in_sample_results    = []
        out_of_sample_results = []
        best_params_per_window = []

        for r in raw_results:
            if r is None:
                continue
            isr = r['in_sample']
            in_sample_results.append(OptimizationResult(**isr))
            out_of_sample_results.append(r['out_of_sample'])
            best_params_per_window.append(r['best_params'])

        if not out_of_sample_results:
            raise ValueError(
                "All walk-forward windows failed. Check data size and parameter grid."
            )

        overall_metrics = {
            'sharpe_ratio': float(np.mean([r['sharpe_ratio'] for r in out_of_sample_results])),
            'total_return':  float(np.mean([r['total_return']  for r in out_of_sample_results])),
            'max_drawdown':  float(np.mean([r['max_drawdown']  for r in out_of_sample_results])),
            'num_windows':   len(out_of_sample_results),
        }

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            best_params_per_window=best_params_per_window,
            combined_equity_curve=np.array([]),
            overall_metrics=overall_metrics,
        )

    def plot_stability(self, result: WalkForwardResult):
        """Plot how optimal parameters change across windows."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        if not result.best_params_per_window:
            return

        param_names = list(result.best_params_per_window[0].keys())
        n_params    = len(param_names)

        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
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
        params: Dict[str, Any],
        data: Dict[str, list],
        n_simulations: int = 1000,
        initial_capital: float = 100000.0,
        method: str = 'bootstrap',
        n_jobs: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo validation of strategy robustness via data resampling.

    Parameters
    ----------
    method : str
        'bootstrap' (sample with replacement) or 'shuffle' (permute returns).
    n_jobs : int
        Worker processes. -1 = all cores.
    """
    symbol = list(data.keys())[0]
    bars   = data[symbol]

    def _make_sample(seed: int) -> Dict[str, list]:
        rng = np.random.default_rng(seed)
        if method == 'bootstrap':
            indices = sorted(rng.choice(len(bars), size=len(bars), replace=True))
            return {symbol: [bars[i] for i in indices]}
        elif method == 'shuffle':
            indices = rng.permutation(len(bars))
            return {symbol: [bars[i] for i in indices]}
        else:
            raise ValueError(f"Unknown method: {method}")

    tasks = [
        (strategy_factory, params, _serialize_data(_make_sample(seed)), initial_capital)
        for seed in range(n_simulations)
    ]

    use_parallel = (
            n_jobs != 1
            and n_simulations >= MIN_PARALLEL_TASKS
    )

    raw_results: List[Optional[dict]] = []

    if use_parallel:
        workers = None if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_backtest_worker, tasks, chunksize=10):
                raw_results.append(result)
    else:
        for task in tasks:
            raw_results.append(_backtest_worker(task))

    sharpes   = [r['sharpe_ratio'] for r in raw_results if r is not None]
    returns   = [r['total_return']  for r in raw_results if r is not None]
    drawdowns = [r['max_drawdown']  for r in raw_results if r is not None]

    return {
        'sharpe_ratios': np.array(sharpes),
        'returns':        np.array(returns),
        'drawdowns':      np.array(drawdowns),
    }