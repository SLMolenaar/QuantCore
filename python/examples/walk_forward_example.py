"""
Walk-Forward Analysis Example

Demonstrates robust parameter optimization and validation techniques.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import quantcore as qc
from quantcore.walk_forward import (
    GridSearchOptimizer,
    WalkForwardAnalyzer,
    monte_carlo_validation
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def generate_synthetic_data(n_bars=500, trend=True):
    """Generate synthetic price data for testing"""
    np.random.seed(42)

    base_price = 100.0
    if trend:
        trend_component = np.linspace(0, 20, n_bars)
        noise = np.random.normal(0, 2, n_bars)
        prices = base_price + trend_component + np.cumsum(noise)
    else:
        prices = base_price + np.cumsum(np.random.normal(0, 1, n_bars))

    bars = []
    for i, price in enumerate(prices):
        high = price * 1.01
        low = price * 0.99

        bar = qc.BarData(
            "SYNTH",
            i * 86400 * 1000000000,
            price,
            high,
            low,
            price,
            1000000.0
        )
        bars.append(bar)

    return bars


def example_grid_search():
    """Example 1: Grid Search Parameter Optimization"""
    print_section("Example 1: Grid Search Optimization")

    bars = generate_synthetic_data(n_bars=500)
    data = {"SYNTH": bars}

    def strategy_factory(fast_period, slow_period):
        return qc.SMACrossover(fast_period=fast_period, slow_period=slow_period)

    param_grid = {
        'fast_period': [10, 20, 30, 40],
        'slow_period': [100, 150, 200]
    }

    print(f"Testing {len(param_grid['fast_period']) * len(param_grid['slow_period'])} parameter combinations...")

    optimizer = GridSearchOptimizer(
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        metric='sharpe_ratio'
    )

    results = optimizer.optimize(data=data, initial_capital=100000.0)

    if not results:
        print("⚠ No valid results found.")
        return []

    print(f"✓ Complete. Tested {len(results)} combinations.\n")
    print("Top 3 Results:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result.params}")
        print(f"     Sharpe: {result.sharpe_ratio:.2f} | Return: {result.total_return:.1%} | MaxDD: {result.max_drawdown:.1%}")

    return results


def example_walk_forward():
    """Example 2: Walk-Forward Analysis"""
    print_section("Example 2: Walk-Forward Analysis")

    bars = generate_synthetic_data(n_bars=1000)
    data = {"SYNTH": bars}

    def strategy_factory(fast_period, slow_period):
        return qc.SMACrossover(fast_period=fast_period, slow_period=slow_period)

    param_grid = {
        'fast_period': [10, 20, 30],
        'slow_period': [50, 100, 150]
    }

    print(f"Running walk-forward analysis on {len(bars)} bars...")
    print("Train: 200 bars | Test: 50 bars | Mode: Rolling\n")

    analyzer = WalkForwardAnalyzer(
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        train_size=200,
        test_size=50,
        anchored=False,
        metric='sharpe_ratio'
    )

    result = analyzer.analyze(data=data, initial_capital=100000.0)

    print("Results:")
    print(f"  Windows: {len(result.best_params_per_window)}")
    print(f"  Avg OOS Sharpe: {result.overall_metrics['sharpe_ratio']:.2f}")
    print(f"  Avg OOS Return: {result.overall_metrics['total_return']:.1%}")

    print("\nParameter Stability:")
    for i, params in enumerate(result.best_params_per_window, 1):
        print(f"  Window {i}: {params}")

    return result


def example_monte_carlo():
    """Example 3: Monte Carlo Validation"""
    print_section("Example 3: Monte Carlo Validation")

    bars = generate_synthetic_data(n_bars=300)
    data = {"SYNTH": bars}

    def strategy_factory(fast_period, slow_period):
        return qc.SMACrossover(fast_period=fast_period, slow_period=slow_period)

    params = {'fast_period': 20, 'slow_period': 100}

    print(f"Testing {params} with 500 Monte Carlo simulations...")

    mc_results = monte_carlo_validation(
        strategy_factory=strategy_factory,
        params=params,
        data=data,
        n_simulations=500,
        method='bootstrap'
    )

    sharpes = mc_results['sharpe_ratios']
    returns = mc_results['returns']

    print("\nResults:")
    print(f"  Sharpe - Mean: {np.mean(sharpes):.2f} | Median: {np.median(sharpes):.2f} | Std: {np.std(sharpes):.2f}")
    print(f"  Return - Mean: {np.mean(returns):.1%} | Median: {np.median(returns):.1%}")
    print(f"  Positive: {np.sum(returns > 0) / len(returns):.0%}")

    if np.percentile(sharpes, 5) > 0:
        print("\n✓ Strategy shows robustness (95% CI positive)")
    else:
        print("\n⚠ Strategy may be overfitted (some negative runs)")

    return mc_results


def example_comparison():
    """Example 4: Compare Multiple Strategies"""
    print_section("Example 4: Strategy Comparison")

    bars = generate_synthetic_data(n_bars=600)
    data = {"SYNTH": bars}

    strategies = [
        ("SMA Fast", lambda: qc.SMACrossover(10, 30)),
        ("SMA Medium", lambda: qc.SMACrossover(20, 50)),
        ("Mean Reversion", lambda: qc.MeanReversion(20, 1.5, 0.5))
    ]

    print("Comparing strategies on same data...\n")

    results = []
    for name, factory in strategies:
        strategy = factory()
        backtest = qc.run_backtest(strategy, data, 100000.0)

        equity = np.array(backtest['equity_curve'])
        from quantcore.analytics import calculate_all_metrics
        metrics = calculate_all_metrics(equity)

        results.append({
            'name': name,
            'sharpe': metrics.sharpe_ratio,
            'return': metrics.total_return / 100.0,
            'drawdown': metrics.max_drawdown / 100.0
        })

    print(f"{'Strategy':<20} {'Sharpe':<10} {'Return':<10} {'MaxDD':<10}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['name']:<20} {r['sharpe']:<10.2f} {r['return']:<10.1%} {r['drawdown']:<10.1%}")

    return results


def main():
    print("=" * 70)
    print("  QuantCore Walk-Forward Analysis Examples")
    print("=" * 70)
    print("\nProduction-grade strategy validation:")
    print("  1. Grid search parameter optimization")
    print("  2. Walk-forward analysis")
    print("  3. Monte Carlo robustness testing")
    print("  4. Strategy comparison")

    try:
        example_grid_search()
        example_walk_forward()
        example_monte_carlo()
        example_comparison()

        print("\n" + "=" * 70)
        print("  All Examples Complete")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Grid search finds optimal parameters")
        print("  • Walk-forward prevents overfitting")
        print("  • Monte Carlo tests robustness")
        print("  • Always validate out-of-sample")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()