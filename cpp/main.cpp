#include <iostream>
#include <iomanip>

#include "Execution.h"

#include "backtesting/backtest_engine.h"
#include "backtesting/data_loader.h"
#include "backtesting/market_data_event.h"
#include "backtesting/event_queue.h"
#include "backtesting/bar_data.h"

#include "strategies/buy_and_hold.h"

#include "orderbook/Order.h"
#include "orderbook/OrderType.h"

using namespace quantcore;

void print_separator(const std::string& title = "") {
    std::cout << "\n" << std::string(60, '=') << "\n";
    if (!title.empty()) {
        std::cout << "  " << title << "\n";
        std::cout << std::string(60, '=') << "\n";
    }
}

void test_orderbook_integration() {
    print_separator("TEST 1: Orderbook Integration");

    ExecutionEngine engine("AAPL");

    std::cout << "Adding buy order: 100 @ $150.00\n";
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 15000, 100);
    engine.execute_order(buy);

    std::cout << "Adding sell order: 100 @ $150.00\n";
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 15000, 100);
    auto trades = engine.execute_order(sell);

    std::cout << "Trades executed: " << trades.size() << "\n";
    std::cout << "Orders in book: " << engine.get_orderbook().Size() << "\n";
    std::cout << "Total fees: $" << std::fixed << std::setprecision(4)
              << engine.get_total_fees() << "\n";
}

void test_event_system() {
    print_separator("TEST 2: Event System");

    EventQueue queue;

    // Add events
    queue.push(std::make_shared<MarketDataEvent>("AAPL", 3000, 100.0, 0.0));
    queue.push(std::make_shared<MarketDataEvent>("AAPL", 1000, 99.0, 0.0));
    queue.push(std::make_shared<MarketDataEvent>("AAPL", 2000, 101.0, 0.0));

    std::cout << "Events in queue: " << queue.size() << "\n";

    // Should pop in chonolog order
    std::cout << "Processing events in chronological order:\n";
    while (!queue.empty()) {
        auto event = queue.pop();
        auto md_event = std::static_pointer_cast<MarketDataEvent>(event);
        std::cout << "  Timestamp: " << md_event->get_timestamp()
                  << ", Price: $" << md_event->get_price() << "\n";
    }

    std::cout << "Events processed in order!\n";
}

void test_data_generation() {
    print_separator("TEST 3: Generate Sample Data");

    // generate fake price data
    BarSeries bars;
    double price = 100.0;

    for (int i = 0; i < 10; ++i) {
        int64_t timestamp = (i + 1) * 1000000000LL;  // 1 sec intervals

        // Random walk
        price += (i % 2 == 0) ? 1.0 : -0.5;

        bars.emplace_back("TEST", timestamp, price, price + 1.0, price - 0.5, price + 0.5, 1000.0);
    }

    std::cout << "Generated " << bars.size() << " bars\n";
    std::cout << "First bar: " << bars[0].close << "\n";
    std::cout << "Last bar: " << bars.back().close << "\n";
    std::cout << "Data generation works\n";
}

void test_backtest_engine() {
    print_separator("TEST 4: Full Backtest");

    // Generate test data, simple uptrend
    BarSeries bars;
    for (int i = 0; i < 100; ++i) {
        int64_t timestamp = i * 1000000000LL;
        double price = 100.0 + i * 0.5;  // Uptrend: +0.50 per bar
        bars.emplace_back("AAPL", timestamp, price, price + 0.2, price - 0.2, price, 1000.0);
    }

    std::cout << "Generated " << bars.size() << " bars (uptrend)\n";
    std::cout << "Starting price: $" << bars.front().close << "\n";
    std::cout << "Ending price: $" << bars.back().close << "\n\n";

    // Create strategy and engine
    auto strategy = std::make_shared<BuyAndHold>();
    BacktestEngine engine(100000.0);  // 100k capital

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running backtest...\n";
    double final_value = engine.run();

    // Results
    std::cout << "\n" << std::string(40, '-') << "\n";
    std::cout << "Backtest Results:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << "Initial capital: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final value: $" << final_value << "\n";
    std::cout << "Total PnL: $" << engine.get_total_pnl() << "\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_value / 100000.0 - 1.0) * 100.0) << "%\n";

    std::cout << "\nBacktest completed\n";
}

int main() {
    print_separator("Test");
    std::cout << "\n";

    try {
        test_orderbook_integration();
        test_event_system();
        test_data_generation();
        test_backtest_engine();

        print_separator();
        std::cout << "\n";
        std::cout << "  ALL TESTS PASSED!\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << "\n\n";
        return 1;
    }
}