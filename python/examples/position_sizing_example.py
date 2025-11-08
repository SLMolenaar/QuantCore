"""
Position Sizing Strategies - Comprehensive Example

Demonstrates how position sizing dramatically impacts strategy performance.
Shows all available position sizing methods and their trade-offs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

@dataclass
class SizingResult:
    name: str
    final_value: float
    total_pnl: float
    total_fees: float
    return_pct: float

def run_backtest_with_sizer(bars, sizer, initial_capital=100000.0):
    """Run backtest with specific position sizer"""
    strategy = qc.SMACrossover(fast_period=50, slow_period=200)
    engine = qc.BacktestEngine(initial_capital)

    engine.add_data("AAPL", bars)
    engine.set_strategy(strategy)
    engine.set_position_sizer(sizer)

    final_value = engine.run()
    total_pnl = engine.get_total_pnl()
    total_fees = engine.get_total_fees()
    return_pct = ((final_value / initial_capital) - 1.0) * 100.0

    return SizingResult(
        name=sizer.get_name(),
        final_value=final_value,
        total_pnl=total_pnl,
        total_fees=total_fees,
        return_pct=return_pct
    )

def find_data_file(filename):
    """Find data file in various locations"""
    search_paths = [
        Path.cwd() / "data" / filename,
        Path(__file__).parent.parent.parent / "data" / filename,
        Path("data") / filename,
        Path("../data") / filename,
        Path("../../data") / filename,
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None

def compare_position_sizing_methods():
    """Compare all position sizing methods"""
    print("=" * 70)
    print("  Position Sizing Comparison")
    print("=" * 70)

    data_file = find_data_file("test_sma_crossover.csv")

    if data_file is None:
        print(f"\nERROR: Could not find test_sma_crossover.csv")
        print(f"Current directory: {Path.cwd()}")
        print(f"\nSearched in:")
        print(f"  - {Path.cwd() / 'data'}")
        print(f"  - {Path(__file__).parent.parent.parent / 'data'}")
        print("\nPlease ensure data files exist or run data generation script!")
        return None

    print(f"\nFound data: {data_file.absolute()}")

    bars = qc.load_csv_data(str(data_file), "AAPL")
    print(f"\nLoaded {len(bars)} bars")
    print("\nRunning backtests with different position sizing methods...")

    results = []

    # 1. Fixed Percentage methods
    print("\n1. Fixed Percentage (10%)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.FixedPercentage(0.10)))
    print("✓")

    print("2. Fixed Percentage (20%)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.FixedPercentage(0.20)))
    print("✓")

    # 2. Risk-Based methods
    print("3. Risk-Based (1% risk)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.RiskBased(0.01)))
    print("✓")

    print("4. Risk-Based (2% risk)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.RiskBased(0.02)))
    print("✓")

    # 3. Kelly Criterion
    print("5. Kelly Criterion (50% WR, 1.5:1)...", end=" ")
    results.append(run_backtest_with_sizer(
        bars,
        qc.KellyCriterion(win_rate=0.50, avg_win=1.5, avg_loss=1.0, fraction=0.5)
    ))
    print("✓")

    # 4. Equal Weight
    print("6. Equal Weight (5 positions)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.EqualWeight(5)))
    print("✓")

    # 5. Volatility Targeting
    print("7. Volatility Targeting (15%)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.VolatilityTargeting(0.15)))
    print("✓")

    # 6. Fixed Shares
    print("8. Fixed Shares (100)...", end=" ")
    results.append(run_backtest_with_sizer(bars, qc.FixedShares(100)))
    print("✓")

    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)

    df = pd.DataFrame([
        {
            'Method': r.name,
            'Final Value': f'${r.final_value:,.2f}',
            'PnL': f'${r.total_pnl:,.2f}',
            'Fees': f'${r.total_fees:,.2f}',
            'Return (%)': f'{r.return_pct:.2f}%'
        }
        for r in results
    ])

    print("\n" + df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  Performance Ranking")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x.return_pct, reverse=True)

    print(f"\n{'Rank':<6} {'Method':<40} {'Return':<12}")
    print("-" * 58)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6} {result.name:<40} {result.return_pct:>10.2f}%")

    best = sorted_results[0]
    worst = sorted_results[-1]

    print("\n" + "=" * 70)
    print("  Key Insights")
    print("=" * 70)

    print(f"\nBest Performer:")
    print(f"  Method: {best.name}")
    print(f"  Return: {best.return_pct:.2f}%")
    print(f"  Final:  ${best.final_value:,.2f}")

    print(f"\nWorst Performer:")
    print(f"  Method: {worst.name}")
    print(f"  Return: {worst.return_pct:.2f}%")
    print(f"  Final:  ${worst.final_value:,.2f}")

    spread = best.return_pct - worst.return_pct
    print(f"\nPerformance Spread: {spread:.2f}%")
    print("\n✓ Position sizing significantly impacts returns!")

    return results

def demonstrate_constraints():
    """Show how constraints work"""
    print("\n" + "=" * 70)
    print("  Position Sizing Constraints")
    print("=" * 70)

    print("\nDemonstrating various constraints on position sizing:\n")

    # Create context
    ctx = qc.PositionSizingContext(
        signal_strength=1.0,
        current_capital=100000.0,
        current_price=100.0,
        current_position=0.0,
        portfolio_volatility=0.02,
        stop_loss_distance=0.05
    )

    # Without constraints
    sizer = qc.FixedPercentage(0.15)
    size = sizer.calculate_size(ctx)

    print("1. Without Constraints:")
    print(f"   Capital: ${ctx.current_capital:,.2f}")
    print(f"   Price: ${ctx.current_price:.2f}")
    print(f"   Allocation: 15%")
    print(f"   Result: {size:.0f} shares (${size * ctx.current_price:,.2f})")

    # Max position constraint
    print("\n2. With Max Position Constraint (100 shares):")
    sizer.set_max_position_size(100)
    size = sizer.calculate_size(ctx)
    print(f"   Result: {size:.0f} shares (capped at maximum)")

    # Min position constraint
    print("\n3. With Min Position Constraint (200 shares):")
    sizer = qc.FixedPercentage(0.15)
    sizer.set_min_position_size(200)
    size = sizer.calculate_size(ctx)
    print(f"   Result: {size:.0f} shares (below minimum → 0)")

    # Leverage constraint
    print("\n4. With Leverage Constraint (1.5x):")
    sizer = qc.FixedPercentage(0.50)
    sizer.set_max_leverage(1.5)
    size = sizer.calculate_size(ctx)
    print(f"   50% allocation requested")
    print(f"   Result: {size:.0f} shares (leverage limited)")

    print("\n✓ Constraints are crucial for risk management!")

def demonstrate_signal_strength():
    """Show impact of signal strength"""
    print("\n" + "=" * 70)
    print("  Signal Strength Impact")
    print("=" * 70)

    print("\nHow signal strength scales position sizing:\n")

    sizer = qc.FixedPercentage(0.10)

    print("Base: 10% of $100,000 = $10,000")
    print("Price: $100/share\n")

    strengths = [0.25, 0.50, 0.75, 1.00]

    print(f"{'Signal Strength':<20} {'Shares':<15} {'Position ($)':<15}")
    print("-" * 50)

    for strength in strengths:
        ctx = qc.PositionSizingContext(
            signal_strength=strength,
            current_capital=100000.0,
            current_price=100.0
        )
        size = sizer.calculate_size(ctx)
        print(f"{strength:<20.2f} {size:<15.0f} ${size * 100.0:<15,.2f}")

    print("\n✓ Signal strength enables confidence-weighted sizing!")

def visualize_comparison(results: List[SizingResult]):
    """Create visualization of position sizing comparison"""
    print("\n" + "=" * 70)
    print("  Generating Visualization")
    print("=" * 70)

    methods = [r.name for r in results]
    returns = [r.return_pct for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax1.barh(methods, returns, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Return (%)', fontsize=12)
    ax1.set_title('Position Sizing Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f' {width:.1f}%',
                ha='left' if width > 0 else 'right',
                va='center',
                fontweight='bold')

    # Pie chart of top performers
    sorted_results = sorted(results, key=lambda x: x.return_pct, reverse=True)[:5]
    labels = [r.name.split('(')[0].strip() for r in sorted_results]
    sizes = [abs(r.return_pct) for r in sorted_results]

    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Top 5 Methods by Return', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = Path('plots/position_sizing_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.show()

def main():
    print("=" * 70)
    print("  QuantCore Position Sizing Examples")
    print("=" * 70)
    print(f"\nVersion: {qc.version()}")

    try:
        results = compare_position_sizing_methods()

        if results is None:
            print("\nSkipping remaining examples (no data available)")
            return 1

        if results:
            demonstrate_constraints()
            demonstrate_signal_strength()

            try:
                visualize_comparison(results)
            except Exception as e:
                print(f"\nVisualization skipped: {e}")

        print("\n" + "=" * 70)
        print("  Examples Complete!")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("  1. Position sizing has a MAJOR impact on returns")
        print("  2. Different methods suit different strategies")
        print("  3. Constraints prevent over-leverage")
        print("  4. Signal strength enables nuanced sizing")
        print("\n  → Choose your position sizing wisely!\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())