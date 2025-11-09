"""
Walk-forward analysis and parameter optimization

This module provides tools for robust strategy validation:
- Walk-forward analysis to prevent overfitting
- Parameter grid search optimization
- Rolling window backtests
- Performance stability analysis

Walk-forward analysis is critical for production strategies as it simulates
the real-world process of periodically reoptimizing parameters on new data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import itertools


@dataclass
class OptimizationResult:
    """Result from a single parameter combination"""
    params: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    num_trades: int
    final_value: float

    def __repr__(self):
        return f"OptimizationResult(sharpe={self.sharpe_ratio:.2f}, return={self.total_return:.2%})"


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis"""
    in_sample_results: List[OptimizationResult]
    out_of_sample_results: List[Dict[str, Any]]
    best_params_per_window: List[Dict[str, Any]]
    combined_equity_curve: np.ndarray
    overall_metrics: Dict[str, float]

    def summary(self) -> str:
        """Generate summary report"""
        lines = [
            "=" * 70,
            "  Walk-Forward Analysis Results",
            "=" * 70,
            "",
            f"Number of Windows: {len(self.in_sample_results)}",
            f"Overall Sharpe Ratio: {self.overall_metrics['sharpe_ratio']:.2f}",
            f"Overall Return: {self.overall_metrics['total_return']:.2%}",
            f"Overall Max Drawdown: {self.overall_metrics['max_drawdown']:.2%}",
            "",
            "Per-Window Performance:",
            "-" * 70,
        ]

        for i, oos in enumerate(self.out_of_sample_results):
            lines.append(f"Window {i+1}:")
            lines.append(f"  Best Params: {self.best_params_per_window[i]}")
            lines.append(f"  OOS Sharpe: {oos['sharpe_ratio']:.2f}")
            lines.append(f"  OOS Return: {oos['total_return']:.2%}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class ParameterGrid:
    """Generate parameter combinations for grid search"""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize parameter grid

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       e.g., {'fast_period': [10, 20, 30], 'slow_period': [50, 100, 200]}
        """
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())
        self.param_values = [param_grid[name] for name in self.param_names]

    def __iter__(self):
        """Iterate over all parameter combinations"""
        for values in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __len__(self):
        """Total number of combinations"""
        return np.prod([len(vals) for vals in self.param_values])


class GridSearchOptimizer:
    """
    Grid search parameter optimization

    Tests all combinations of parameters and ranks by chosen metric.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        n_jobs: int = 1
    ):
        """
        Initialize grid search optimizer

        Args:
            strategy_factory: Function that creates strategy given params
            param_grid: Dictionary of parameter ranges
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
            n_jobs: Number of parallel jobs (1 = sequential)
        """
        self.strategy_factory = strategy_factory
        self.param_grid = ParameterGrid(param_grid)
        self.metric = metric
        self.n_jobs = n_jobs
        self.results: List[OptimizationResult] = []

    def optimize(
        self,
        data: Dict[str, List],
        initial_capital: float = 100000.0,
        verbose: bool = True
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization

        Args:
            data: Market data for backtesting
            initial_capital: Starting capital
            verbose: Print progress

        Returns:
            List of results sorted by metric
        """
        from . import run_backtest
        from .analytics import calculate_all_metrics, calculate_returns

        total_combos = len(self.param_grid)
        self.results = []

        for i, params in enumerate(self.param_grid):
            if 'fast_period' in params and 'slow_period' in params:
                if params['fast_period'] >= params['slow_period']:
                    continue

            try:
                strategy = self.strategy_factory(**params)
                backtest_results = run_backtest(
                    strategy=strategy,
                    data=data,
                    initial_capital=initial_capital
                )

                if not isinstance(backtest_results, dict):
                    continue

                equity_curve = backtest_results.get('equity_curve')
                if equity_curve is None or len(equity_curve) == 0:
                    continue

                equity_curve = np.array(equity_curve)
                returns = calculate_returns(equity_curve)
                metrics = calculate_all_metrics(equity_curve)

                result = OptimizationResult(
                    params=params,
                    sharpe_ratio=metrics.sharpe_ratio,
                    total_return=metrics.total_return / 100.0,
                    max_drawdown=metrics.max_drawdown / 100.0,
                    num_trades=metrics.total_trades,
                    final_value=backtest_results['final_value']
                )

                self.results.append(result)

            except Exception:
                continue

        self.results.sort(key=lambda x: getattr(x, self.metric), reverse=True)

        return self.results

    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """Get top N results by metric"""
        return self.results[:n]

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        for result in self.results:
            row = result.params.copy()
            row['sharpe_ratio'] = result.sharpe_ratio
            row['total_return'] = result.total_return
            row['max_drawdown'] = result.max_drawdown
            row['num_trades'] = result.num_trades
            data.append(row)

        return pd.DataFrame(data)


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation

    Splits data into multiple train/test windows:
    - Train on in-sample data to find best parameters
    - Test on out-of-sample data with those parameters
    - Repeat for each window

    This simulates realistic parameter reoptimization over time.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        param_grid: Dict[str, List[Any]],
        train_size: int = 252,
        test_size: int = 63,
        anchored: bool = False,
        metric: str = 'sharpe_ratio'
    ):
        """
        Initialize walk-forward analyzer

        Args:
            strategy_factory: Function creating strategy from params
            param_grid: Parameter ranges to test
            train_size: Number of periods for training
            test_size: Number of periods for testing
            anchored: If True, training window grows; if False, it slides
            metric: Metric to optimize on
        """
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.train_size = train_size
        self.test_size = test_size
        self.anchored = anchored
        self.metric = metric

    def analyze(
        self,
        data: Dict[str, List],
        initial_capital: float = 100000.0,
        verbose: bool = False
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis

        Args:
            data: Market data (dict mapping symbols to bar lists)
            initial_capital: Starting capital
            verbose: Print progress

        Returns:
            WalkForwardResult with all analysis
        """
        from . import run_backtest
        from .analytics import calculate_all_metrics

        symbol = list(data.keys())[0]
        bars = data[symbol]
        total_periods = len(bars)

        if total_periods < self.train_size + self.test_size:
            raise ValueError(f"Not enough data: need {self.train_size + self.test_size}, have {total_periods}")

        in_sample_results = []
        out_of_sample_results = []
        best_params_per_window = []

        train_start = 0
        window_num = 0

        while train_start + self.train_size + self.test_size <= total_periods:
            window_num += 1
            train_end = train_start + self.train_size
            test_end = train_end + self.test_size

            train_data = {symbol: bars[train_start:train_end]}
            test_data = {symbol: bars[train_end:test_end]}

            optimizer = GridSearchOptimizer(
                strategy_factory=self.strategy_factory,
                param_grid=self.param_grid,
                metric=self.metric,
                n_jobs=1
            )

            train_results = optimizer.optimize(
                data=train_data,
                initial_capital=initial_capital,
                verbose=False
            )

            if not train_results:
                if not self.anchored:
                    train_start = train_end
                continue

            best_result = train_results[0]
            best_params = best_result.params

            strategy = self.strategy_factory(**best_params)
            test_backtest = run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=initial_capital
            )

            equity_curve = np.array(test_backtest.get('equity_curve', []))
            if len(equity_curve) > 0:
                test_metrics = calculate_all_metrics(equity_curve)

                oos_result = {
                    'sharpe_ratio': test_metrics.sharpe_ratio,
                    'total_return': test_metrics.total_return / 100.0,
                    'max_drawdown': test_metrics.max_drawdown / 100.0,
                    'num_trades': test_metrics.total_trades,
                    'final_value': test_backtest['final_value']
                }
            else:
                oos_result = {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'num_trades': 0,
                    'final_value': initial_capital
                }

            in_sample_results.append(best_result)
            out_of_sample_results.append(oos_result)
            best_params_per_window.append(best_params)

            if not self.anchored:
                train_start = train_end

        if not out_of_sample_results:
            raise ValueError("No valid walk-forward windows. Check data size and parameter grid.")

        overall_sharpe = np.mean([r['sharpe_ratio'] for r in out_of_sample_results])
        overall_return = np.mean([r['total_return'] for r in out_of_sample_results])
        overall_dd = np.mean([r['max_drawdown'] for r in out_of_sample_results])

        overall_metrics = {
            'sharpe_ratio': overall_sharpe,
            'total_return': overall_return,
            'max_drawdown': overall_dd,
            'num_windows': len(out_of_sample_results)
        }

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            best_params_per_window=best_params_per_window,
            combined_equity_curve=np.array([]),
            overall_metrics=overall_metrics
        )

    def plot_stability(self, result: WalkForwardResult):
        """
        Plot parameter stability over windows

        Shows how optimal parameters change over time
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        param_names = list(result.best_params_per_window[0].keys())
        n_params = len(param_names)

        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        if n_params == 1:
            axes = [axes]

        windows = range(1, len(result.best_params_per_window) + 1)

        for i, param_name in enumerate(param_names):
            values = [params[param_name] for params in result.best_params_per_window]
            axes[i].plot(windows, values, marker='o', linewidth=2)
            axes[i].set_xlabel('Window')
            axes[i].set_ylabel(param_name)
            axes[i].set_title(f'{param_name} Stability')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def monte_carlo_validation(
    strategy_factory: Callable,
    params: Dict[str, Any],
    data: Dict[str, List],
    n_simulations: int = 1000,
    initial_capital: float = 100000.0,
    method: str = 'bootstrap'
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo validation of strategy robustness

    Args:
        strategy_factory: Function to create strategy
        params: Strategy parameters
        data: Market data
        n_simulations: Number of Monte Carlo runs
        initial_capital: Starting capital
        method: 'bootstrap' or 'shuffle'

    Returns:
        Dictionary with arrays of simulated metrics
    """
    from . import run_backtest
    from .analytics import calculate_all_metrics

    sharpes = []
    returns = []
    drawdowns = []

    symbol = list(data.keys())[0]
    bars = data[symbol]

    for i in range(n_simulations):
        if method == 'bootstrap':
            indices = np.random.choice(len(bars), size=len(bars), replace=True)
            sampled_bars = [bars[idx] for idx in sorted(indices)]
        elif method == 'shuffle':
            sampled_bars = bars.copy()
            np.random.shuffle(sampled_bars)
        else:
            raise ValueError(f"Unknown method: {method}")

        sim_data = {symbol: sampled_bars}

        try:
            strategy = strategy_factory(**params)
            results = run_backtest(strategy, sim_data, initial_capital)

            equity_curve = np.array(results.get('equity_curve', []))
            if len(equity_curve) > 0:
                metrics = calculate_all_metrics(equity_curve)
                sharpes.append(metrics.sharpe_ratio)
                returns.append(metrics.total_return / 100.0)
                drawdowns.append(metrics.max_drawdown / 100.0)
        except:
            continue

    return {
        'sharpe_ratios': np.array(sharpes),
        'returns': np.array(returns),
        'drawdowns': np.array(drawdowns)
    }