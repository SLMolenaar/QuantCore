"""
QuantCore Example Usage

Demonstrates the full workflow:
1. Load market data
2. Run backtests with different strategies
3. Calculate performance metrics
4. Generate visualizations
5. Compare strategies
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import quantcore as qc
except ImportError:
    print("ERROR: Could not import quantcore module.")
    print("Build the C++ extension first:")
    print("  python python/build_module.py")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import analytics and plotting
try:
    from quantcore.analytics import calculate_all_metrics, calculate_returns
    from quantcore.plotting import (
        plot_equity_curve,
        plot_underwater,
        plot_returns_distribution,
        plot_rolling_metrics,
        plot_full_tearsheet
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    print("WARNING: Analytics/plotting modules not found")
    print("Place analytics.py and plotting.py in python/quantcore/")
    PLOTTING_AVAILABLE = False

# Find project root (where data/ directory is)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Verify data directory exists
if not DATA_DIR.exists():
    print(f"WARNING: Data directory not found at {DATA_DIR}")
    print("Looking for data files in script directory...")
    DATA_DIR = SCRIPT_DIR


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def build_equity_curve(initial_capital, final_value, num_periods):
    """
    Build synthetic equity curve for visualization
    In production, you'd track this during the backtest
    """
    return_pct = (final_value / initial_capital - 1.0)

    # Simple linear growth for demo
    curve = np.array([
        initial_capital * (1 + (i / num_periods) * return_pct)
        for i in range(num_periods)
    ])

    # Add some realistic volatility
    noise = np.random.normal(0, 0.002, num_periods)
    curve = curve * (1 + np.cumsum(noise))

    return curve


def run_buy_and_hold():
    """Example 1: Buy and Hold Strategy"""
    print_section("Example 1: Buy and Hold Strategy")

    data_file = DATA_DIR / "test_buy_and_hold.csv"
    print(f"Loading data from: {data_file}")

    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    bars = qc.load_csv_data(str(data_file), "AAPL")
    print(f"Loaded {len(bars)} bars")

    strategy = qc.BuyAndHold()

    print("\nRunning backtest...")
    results = qc.run_backtest(
        strategy=strategy,
        data={"AAPL": bars},
        initial_capital=100000.0
    )

    print("\nResults:")
    print(f"  Strategy:        {results['strategy']}")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Value:     ${results['final_value']:,.2f}")
    print(f"  Total PnL:       ${results['total_pnl']:,.2f}")
    print(f"  Total Fees:      ${results['total_fees']:,.2f}")
    print(f"  Return:          {results['return_pct']:.2f}%")

    return results, bars


def run_sma_crossover():
    """Example 2: SMA Crossover Strategy"""
    print_section("Example 2: SMA Crossover Strategy")

    data_file = DATA_DIR / "test_sma_crossover.csv"
    print(f"Loading data from: {data_file}")

    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    bars = qc.load_csv_data(str(data_file), "AAPL")
    print(f"Loaded {len(bars)} bars")

    strategy = qc.SMACrossover(fast_period=50, slow_period=200)

    engine = qc.BacktestEngine(initial_capital=100000.0)
    engine.add_data("AAPL", bars)
    engine.set_strategy(strategy)

    print("\nRunning backtest...")
    final_value = engine.run()

    total_pnl = engine.get_total_pnl()
    total_fees = engine.get_total_fees()
    return_pct = ((final_value / 100000.0) - 1.0) * 100.0

    print("\nResults:")
    print(f"  Initial Capital: $100,000.00")
    print(f"  Final Value:     ${final_value:,.2f}")
    print(f"  Total PnL:       ${total_pnl:,.2f}")
    print(f"  Total Fees:      ${total_fees:,.2f}")
    print(f"  Return:          {return_pct:.2f}%")

    ee = engine.get_execution_engine("AAPL")
    if ee:
        print("\n  Position Details:")
        print(f"    Final Position:   {ee.get_position():.0f} shares")
        print(f"    Average Price:    ${ee.get_average_price():.2f}")
        print(f"    Realized PnL:     ${ee.get_realized_pnl():,.2f}")
        print(f"    Unrealized PnL:   ${ee.get_unrealized_pnl():,.2f}")

    return {
        'strategy': strategy.get_name(),
        'final_value': final_value,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'return_pct': return_pct
    }, bars


def run_mean_reversion():
    """Example 3: Mean Reversion Strategy"""
    print_section("Example 3: Mean Reversion Strategy")

    data_file = DATA_DIR / "test_mean_reversion.csv"
    print(f"Loading data from: {data_file}")

    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    bars = qc.load_csv_data(str(data_file), "AAPL")
    print(f"Loaded {len(bars)} bars")

    strategy = qc.MeanReversion(
        lookback=20,
        entry_threshold=1.5,
        exit_threshold=0.5
    )

    print("\nRunning backtest...")
    results = qc.run_backtest(
        strategy=strategy,
        data={"AAPL": bars},
        initial_capital=100000.0
    )

    print("\nResults:")
    print(f"  Return:           {results['return_pct']:.2f}%")
    print(f"  Total PnL:        ${results['total_pnl']:,.2f}")
    print(f"  Total Fees:       ${results['total_fees']:,.2f}")
    print(f"  Signals Generated: {strategy.get_signal_count()}")

    return results, bars


def calculate_metrics_example(results, bars):
    """Example 4: Calculate Performance Metrics"""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping metrics calculation (analytics module not available)")
        return None

    print_section("Example 4: Performance Metrics")

    initial_capital = results.get('initial_capital', 100000.0)
    final_value = results['final_value']

    # Build equity curve
    equity_curve = build_equity_curve(initial_capital, final_value, len(bars))

    print("Calculating comprehensive metrics...")
    metrics = calculate_all_metrics(
        equity_curve=equity_curve,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    print("\n" + str(metrics))

    return equity_curve, metrics


def visualize_results(equity_curve, bars, strategy_name):
    """Example 5: Visualize Backtest Results"""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping visualization (plotting module not available)")
        return

    print_section("Example 5: Visualization")

    # Extract timestamps
    timestamps = np.array([bar.timestamp_ns for bar in bars])

    print("Generating and displaying plots...\n")

    # 1. Equity curve
    print("  Plot 1: Equity curve with drawdown shading")
    try:
        fig = plot_equity_curve(
            equity_curve,
            timestamps,
            title=f"{strategy_name} - Equity Curve",
            show_drawdown=True
        )
        plt.savefig(f"plots/{strategy_name}_equity.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # 2. Underwater plot
    print("\n  Plot 2: Underwater plot (drawdown over time)")
    try:
        fig = plot_underwater(equity_curve, timestamps)
        plt.savefig(f"plots/{strategy_name}_underwater.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # 3. Returns distribution
    print("\n  Plot 3: Returns distribution with Q-Q plot")
    try:
        returns = calculate_returns(equity_curve)
        fig = plot_returns_distribution(returns)
        plt.savefig(f"plots/{strategy_name}_returns.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        print(f"    Hint: Install scipy with: pip install scipy")

    # 4. Rolling metrics
    print("\n  Plot 4: Rolling Sharpe and Volatility (60-period window)")
    try:
        returns = calculate_returns(equity_curve)
        fig = plot_rolling_metrics(returns, timestamps, window=60)
        plt.savefig(f"plots/{strategy_name}_rolling.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # 5. Full tearsheet
    print("\n  Plot 5: Full performance tearsheet (comprehensive view)")
    try:
        returns = calculate_returns(equity_curve)
        fig = plot_full_tearsheet(
            equity_curve,
            returns,
            timestamps,
            title=f"{strategy_name} Performance Tearsheet"
        )
        plt.savefig(f"plots/{strategy_name}_tearsheet.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("    ✓ Success")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print(f"\n  Plots saved to plots/ directory")


def compare_strategies(results_list):
    """Example 6: Strategy Comparison"""
    print_section("Example 6: Strategy Comparison")

    comparison = pd.DataFrame([
        {
            'Strategy': r['strategy'],
            'Return (%)': r['return_pct'],
            'Total PnL ($)': r['total_pnl'],
            'Fees ($)': r['total_fees']
        }
        for r in results_list
    ])

    print("\n" + comparison.to_string(index=False))

    # Find best strategy
    best_idx = comparison['Return (%)'].idxmax()
    best = comparison.iloc[best_idx]

    print(f"\nBest Performer: {best['Strategy']} ({best['Return (%)']:.2f}%)")

    if PLOTTING_AVAILABLE:
        print("\nGenerating comparison visualization...")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(comparison))
        colors = ['#2E86AB' if r > 0 else '#A23B72' for r in comparison['Return (%)']]
        bars = ax.bar(x, comparison['Return (%)'], 0.6, color=colors, alpha=0.8)

        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['Strategy'], rotation=15, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig('plots/strategy_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("  Comparison chart saved and displayed")


def main():
    """Main execution"""
    print("=" * 70)
    print("  QuantCore - Example Usage")
    print("=" * 70)
    print(f"\nVersion: {qc.version()}")
    print(qc.hello())
    print(f"\nData directory: {DATA_DIR}")
    print(f"Working directory: {Path.cwd()}")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    print(f"Plots directory: {plots_dir.absolute()}")

    results_list = []

    # Example 1: Buy and Hold
    try:
        bh_results, bh_bars = run_buy_and_hold()
        results_list.append(bh_results)
    except Exception as e:
        print(f"\nERROR in Buy and Hold: {e}")

    # Example 2: SMA Crossover
    try:
        sma_results, sma_bars = run_sma_crossover()
        results_list.append(sma_results)

        # Calculate metrics and visualize for SMA
        if PLOTTING_AVAILABLE:
            equity_curve, metrics = calculate_metrics_example(sma_results, sma_bars)
            if equity_curve is not None:
                visualize_results(equity_curve, sma_bars, "SMACrossover")

    except Exception as e:
        print(f"\nERROR in SMA Crossover: {e}")

    # Example 3: Mean Reversion
    try:
        mr_results, mr_bars = run_mean_reversion()
        results_list.append(mr_results)
    except Exception as e:
        print(f"\nERROR in Mean Reversion: {e}")

    # Example 6: Compare all strategies
    if len(results_list) > 1:
        compare_strategies(results_list)

    # Summary
    print_section("Summary")
    print("\nCompleted demonstrations:")
    print("  ✓ Buy and Hold baseline")
    print("  ✓ SMA Crossover trend following")
    print("  ✓ Mean Reversion statistical arbitrage")

    if PLOTTING_AVAILABLE:
        print("  ✓ Performance metrics calculation")
        print("  ✓ Visualization generation")
        print("  ✓ Strategy comparison")
    else:
        print("\n  ! Add analytics.py and plotting.py to python/quantcore/ for full features")

    print("\n" + "=" * 70)
    print("  Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Check plots/ directory for visualizations")
    print("  - Run example_backtest.ipynb for interactive analysis")
    print("  - Implement your own strategies in C++ or Python")
    print()


if __name__ == "__main__":
    main()