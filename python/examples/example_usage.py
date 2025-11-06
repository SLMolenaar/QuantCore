"""
Example: Using QuantCore Python bindings

This script demonstrates how to use the QuantCore backtesting engine from Python.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import quantcore
sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc

# Add parent directory to path to import quantcore
sys.path.insert(0, str(Path(__file__).parent))

try:
    import quantcore as qc
except ImportError:
    print("Error: Could not import quantcore module.")
    print("Make sure you've built the C++ extension:")
    print("  python python/build_module.py")
    sys.exit(1)


def example_1_simple_backtest():
    """Example 1: Simple buy and hold backtest"""
    print("\n" + "=" * 60)
    print("Example 1: Buy and Hold Strategy")
    print("=" * 60)

    # Load data
    print("Loading data from CSV...")
    bars = qc.load_csv_data("../data/test_buy_and_hold.csv", "AAPL")
    print(f"Loaded {len(bars)} bars")

    # Create strategy
    strategy = qc.BuyAndHold()

    # Create and run backtest
    print("Running backtest...")
    results = qc.run_backtest(
        strategy=strategy,
        data={"AAPL": bars},
        initial_capital=100000.0
    )

    # Print results
    print(f"\nResults:")
    print(f"  Strategy: {results['strategy']}")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Value: ${results['final_value']:,.2f}")
    print(f"  Total PnL: ${results['total_pnl']:,.2f}")
    print(f"  Total Fees: ${results['total_fees']:,.2f}")
    print(f"  Return: {results['return_pct']:.2f}%")


def example_2_sma_crossover():
    """Example 2: SMA Crossover strategy"""
    print("\n" + "=" * 60)
    print("Example 2: SMA Crossover Strategy")
    print("=" * 60)

    # Load data
    print("Loading data from CSV...")
    bars = qc.load_csv_data("../data/test_sma_crossover.csv", "AAPL")
    print(f"Loaded {len(bars)} bars")

    # Create strategy with custom parameters
    strategy = qc.SMACrossover(fast_period=50, slow_period=200)

    # Create backtest engine manually for more control
    engine = qc.BacktestEngine(initial_capital=100000.0)
    engine.add_data("AAPL", bars)
    engine.set_strategy(strategy)

    print("Running backtest...")
    final_value = engine.run()

    # Get detailed results
    print(f"\nResults:")
    print(f"  Initial Capital: $100,000.00")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total PnL: ${engine.get_total_pnl():,.2f}")
    print(f"  Total Fees: ${engine.get_total_fees():,.2f}")
    print(f"  Return: {((final_value / 100000.0 - 1.0) * 100.0):.2f}%")

    # Get execution engine details
    ee = engine.get_execution_engine("AAPL")
    if ee:
        print(f"\nPosition Details:")
        print(f"  Final Position: {ee.get_position()}")
        print(f"  Average Price: ${ee.get_average_price():.2f}")
        print(f"  Realized PnL: ${ee.get_realized_pnl():.2f}")
        print(f"  Unrealized PnL: ${ee.get_unrealized_pnl():.2f}")


def example_3_mean_reversion():
    """Example 3: Mean Reversion strategy"""
    print("\n" + "=" * 60)
    print("Example 3: Mean Reversion Strategy")
    print("=" * 60)

    # Load data
    print("Loading data from CSV...")
    bars = qc.load_csv_data("../data/test_mean_reversion.csv", "AAPL")
    print(f"Loaded {len(bars)} bars")

    # Create mean reversion strategy
    strategy = qc.MeanReversion(
        lookback=20,
        entry_threshold=1.5,
        exit_threshold=0.5
    )

    # Run backtest
    print("Running backtest...")
    results = qc.run_backtest(strategy, {"AAPL": bars}, 100000.0)

    # Print results
    print(f"\nResults:")
    print(f"  Return: {results['return_pct']:.2f}%")
    print(f"  Total PnL: ${results['total_pnl']:,.2f}")
    print(f"  Total Fees: ${results['total_fees']:,.2f}")
    print(f"  Signals Generated: {strategy.get_signal_count()}")


def example_4_custom_strategy():
    """Example 4: Custom Python strategy"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Python Strategy")
    print("=" * 60)

    print("\nIMPORTANT: Custom Python strategies are currently limited.")
    print("\nThe Strategy base class has protected methods that cannot be")
    print("directly accessed from Python without modifying the C++ code.")
    print("\nTo enable custom Python strategies, you need to:")
    print("  1. Make generate_signal() and get_position() public in strategy.h")
    print("  2. OR add friend class declarations")
    print("  3. OR create public wrapper methods")
    print("\nFor now, use the built-in C++ strategies:")
    print("  - BuyAndHold")
    print("  - SMACrossover")
    print("  - MeanReversion")
    print("\nOr implement new strategies directly in C++ for best performance.")

    # Example structure (won't work without C++ changes):
    print("\n# Would need C++ modifications:")
    print("""
    class SimpleThreshold(qc.Strategy):
        def __init__(self):
            super().__init__("SimpleThreshold")
            self.prices = {}
        
        def on_data(self, event):
            symbol = event.get_symbol()
            price = event.get_close()
            
            # Track price history
            if symbol not in self.prices:
                self.prices[symbol] = []
            self.prices[symbol].append(price)
            
            # Strategy logic here...
            # Would call self.generate_signal(...) if it was accessible
    """)


def example_5_bar_data_inspection():
    """Example 5: Inspecting bar data"""
    print("\n" + "=" * 60)
    print("Example 5: Bar Data Inspection")
    print("=" * 60)

    # Load data
    bars = qc.load_csv_data("../data/test_buy_and_hold.csv", "AAPL")

    print(f"Total bars: {len(bars)}")
    print(f"\nFirst bar:")
    bar = bars[0]
    print(f"  Symbol: {bar.symbol}")
    print(f"  Timestamp: {bar.timestamp_ns}")
    print(f"  Open: ${bar.open:.2f}")
    print(f"  High: ${bar.high:.2f}")
    print(f"  Low: ${bar.low:.2f}")
    print(f"  Close: ${bar.close:.2f}")
    print(f"  Volume: {bar.volume:,.0f}")
    print(f"  Typical Price: ${bar.typical_price():.2f}")
    print(f"  Range: ${bar.range():.2f}")
    print(f"  Bullish: {bar.is_bullish()}")
    print(f"  Bearish: {bar.is_bearish()}")

    # Calculate some statistics
    closes = [b.close for b in bars]
    print(f"\nPrice Statistics:")
    print(f"  Min: ${min(closes):.2f}")
    print(f"  Max: ${max(closes):.2f}")
    print(f"  Average: ${sum(closes) / len(closes):.2f}")


def example_6_execution_config():
    """Example 6: Custom execution configuration"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Execution Configuration")
    print("=" * 60)

    # Create custom execution config
    config = qc.ExecutionConfig()
    config.maker_fee = 0.0001  # 1 bps
    config.taker_fee = 0.0002  # 2 bps
    config.latency_ns = 1000000  # 1ms
    config.slippage_pct = 0.0001  # 1 bps

    print("Custom execution configuration:")
    print(f"  Maker fee: {config.maker_fee * 100:.3f}%")
    print(f"  Taker fee: {config.taker_fee * 100:.3f}%")
    print(f"  Latency: {config.latency_ns / 1_000_000:.1f}ms")
    print(f"  Slippage: {config.slippage_pct * 100:.3f}%")

    print("\nNote: Currently ExecutionEngine needs to be created manually with config.")
    print("This feature is planned for future releases.")


def main():
    """Run all examples"""
    print("=" * 60)
    print("QuantCore Python Bindings - Examples")
    print("=" * 60)
    print(f"Version: {qc.version()}")
    print(qc.hello())

    try:
        example_1_simple_backtest()
    except Exception as e:
        print(f"Error in example 1: {e}")

    try:
        example_2_sma_crossover()
    except Exception as e:
        print(f"Error in example 2: {e}")

    try:
        example_3_mean_reversion()
    except Exception as e:
        print(f"Error in example 3: {e}")

    example_4_custom_strategy()

    try:
        example_5_bar_data_inspection()
    except Exception as e:
        print(f"Error in example 5: {e}")

    example_6_execution_config()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()