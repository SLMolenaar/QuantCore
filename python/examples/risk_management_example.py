"""
Pre-Trade Risk Management Examples

Demonstrates how risk limits prevent dangerous trades before execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import quantcore as qc


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def generate_volatile_data(n_bars=300):
    """Generate synthetic data with high volatility"""
    np.random.seed(42)

    base_price = 100.0
    prices = [base_price]

    for _ in range(n_bars - 1):
        change = np.random.normal(0, 3)
        new_price = max(prices[-1] + change, 10.0)
        prices.append(new_price)

    bars = []
    for i, price in enumerate(prices):
        bar = qc.BarData(
            "VOL",
            i * 86400 * 1000000000,
            price,
            price * 1.02,
            price * 0.98,
            price,
            1000000.0
        )
        bars.append(bar)

    return bars


def example_position_limit():
    """Example 1: Position Size Limit"""
    print_section("Example 1: Position Size Limit Protection")

    limits = qc.RiskLimits()
    limits.max_position_pct = 0.10  # Max 10% per position
    limits.max_leverage = 2.0
    limits.enabled = True

    risk_mgr = qc.RiskManager(limits)
    risk_mgr.set_capital(100000.0, 100000.0)

    print("Risk Limits:")
    print(f"  Max Position: {limits.max_position_pct:.0%}")
    print(f"  Max Leverage: {limits.max_leverage}x")
    print(f"  Current Capital: $100,000\n")

    test_orders = [
        ("Small order (5%)", 50, 100.0),
        ("Medium order (10%)", 100, 100.0),
        ("Large order (20%)", 200, 100.0),
    ]

    for desc, qty, price in test_orders:
        check = risk_mgr.check_order("AAPL", qc.Side.BUY, qty, price)
        status = "✓ APPROVED" if check.is_approved() else f"✗ REJECTED: {check.reason}"
        print(f"{desc:<25} ${qty * price:>8,.0f}  {status}")

    print("\nKey Insight: Orders exceeding 10% position limit are rejected")


def example_leverage_limit():
    """Example 2: Leverage Limit"""
    print_section("Example 2: Leverage Limit Protection")

    limits = qc.RiskLimits()
    limits.max_position_pct = 0.50
    limits.max_leverage = 2.0
    limits.enabled = True

    risk_mgr = qc.RiskManager(limits)
    risk_mgr.set_capital(100000.0, 100000.0)

    print("Scenario: Building positions in multiple symbols")
    print(f"Max Leverage: {limits.max_leverage}x")
    print(f"Capital: $100,000\n")

    positions = [
        ("AAPL", 400, 100.0),
        ("GOOGL", 300, 150.0),
        ("MSFT", 500, 200.0),  # This will exceed leverage
    ]

    total_exposure = 0
    for symbol, qty, price in positions:
        check = risk_mgr.check_order(symbol, qc.Side.BUY, qty, price)

        if check.is_approved():
            risk_mgr.set_position(symbol, qty)
            total_exposure += qty * price
            leverage = total_exposure / 100000.0
            print(f"✓ {symbol:<6} {qty:>4} @ ${price:<6.0f} = ${qty * price:>8,.0f}  Leverage: {leverage:.2f}x")
        else:
            print(f"✗ {symbol:<6} {qty:>4} @ ${price:<6.0f}   REJECTED: {check.reason}")

    print(f"\nFinal Exposure: ${total_exposure:,.0f} ({total_exposure / 100000:.2f}x leverage)")
    print("Key Insight: System prevents over-leveraging")


def example_loss_limit():
    """Example 3: Maximum Loss Protection"""
    print_section("Example 3: Maximum Loss Protection")

    limits = qc.RiskLimits()
    limits.max_loss_pct = 0.20  # Stop if down 20%
    limits.enabled = True

    risk_mgr = qc.RiskManager(limits)

    print("Scenario: Portfolio losing money")
    print(f"Initial Capital: $100,000")
    print(f"Max Loss Allowed: {limits.max_loss_pct:.0%}\n")

    scenarios = [
        ("After small loss", 100000.0, 95000.0),
        ("After moderate loss", 100000.0, 85000.0),
        ("After large loss", 100000.0, 75000.0),
    ]

    for desc, initial, current in scenarios:
        risk_mgr.set_capital(initial, current)
        loss_pct = (initial - current) / initial
        check = risk_mgr.check_order("AAPL", qc.Side.BUY, 100, 100.0)

        status = "✓ Trading allowed" if check.is_approved() else "✗ Trading halted"
        print(f"{desc:<20} ${current:>8,.0f}  Loss: {loss_pct:>5.1%}  {status}")

        if not check.is_approved():
            print(f"  Reason: {check.reason}")

    print("\nKey Insight: Prevents revenge trading after large losses")


def example_with_backtest():
    """Example 4: Risk Limits in Backtest"""
    print_section("Example 4: Risk-Controlled Backtest")

    bars = generate_volatile_data(200)
    data = {"VOL": bars}

    print("Comparing backtest with/without risk limits...\n")

    # Backtest WITHOUT risk limits
    print("1. NO RISK LIMITS:")
    strategy1 = qc.SMACrossover(10, 30)
    engine1 = qc.BacktestEngine(100000.0)
    engine1.add_data("VOL", bars)
    engine1.set_strategy(strategy1)

    final1 = engine1.run()
    pnl1 = engine1.get_total_pnl()

    print(f"   Final Value: ${final1:,.0f}")
    print(f"   PnL: ${pnl1:,.0f}")
    print(f"   Return: {((final1 / 100000) - 1) * 100:.1f}%")

    # Backtest WITH tight risk limits
    print("\n2. WITH RISK LIMITS:")
    limits = qc.RiskLimits()
    limits.max_position_pct = 0.15
    limits.max_leverage = 1.5
    limits.enabled = True

    strategy2 = qc.SMACrossover(10, 30)
    engine2 = qc.BacktestEngine(100000.0)
    engine2.set_risk_limits(limits)
    engine2.add_data("VOL", bars)
    engine2.set_strategy(strategy2)

    final2 = engine2.run()
    pnl2 = engine2.get_total_pnl()

    print(f"   Max Position: {limits.max_position_pct:.0%}")
    print(f"   Max Leverage: {limits.max_leverage}x")
    print(f"   Final Value: ${final2:,.0f}")
    print(f"   PnL: ${pnl2:,.0f}")
    print(f"   Return: {((final2 / 100000) - 1) * 100:.1f}%")

    print("\nKey Insight: Risk limits may reduce returns but protect capital")


def example_custom_limits():
    """Example 5: Custom Risk Configurations"""
    print_section("Example 5: Custom Risk Profiles")

    profiles = {
        "Conservative": qc.RiskLimits(),
        "Moderate": qc.RiskLimits(),
        "Aggressive": qc.RiskLimits(),
    }

    profiles["Conservative"].max_position_pct = 0.10
    profiles["Conservative"].max_leverage = 1.0
    profiles["Conservative"].max_loss_pct = 0.15

    profiles["Moderate"].max_position_pct = 0.20
    profiles["Moderate"].max_leverage = 2.0
    profiles["Moderate"].max_loss_pct = 0.30

    profiles["Aggressive"].max_position_pct = 0.40
    profiles["Aggressive"].max_leverage = 3.0
    profiles["Aggressive"].max_loss_pct = 0.50

    print(f"{'Profile':<15} {'Max Position':<15} {'Max Leverage':<15} {'Max Loss':<15}")
    print("-" * 60)

    for name, limits in profiles.items():
        print(f"{name:<15} {limits.max_position_pct:<15.0%} {limits.max_leverage:<15.1f}x {limits.max_loss_pct:<15.0%}")

    print("\nKey Insight: Different strategies need different risk profiles")


def main():
    print("=" * 70)
    print("  QuantCore Pre-Trade Risk Management")
    print("=" * 70)
    print("\nProduction-grade risk controls:")
    print("  1. Position size limits")
    print("  2. Leverage limits")
    print("  3. Maximum loss protection")
    print("  4. Integration with backtests")
    print("  5. Custom risk profiles")

    try:
        example_position_limit()
        example_leverage_limit()
        example_loss_limit()
        # example_with_backtest()  # Uncomment when integrated
        example_custom_limits()

        print("\n" + "=" * 70)
        print("  All Examples Complete")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Pre-trade checks prevent dangerous orders")
        print("  • Risk limits are configurable per strategy")
        print("  • System protects capital before execution")
        print("  • Professional trading systems always have this")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()