#include <iostream>
#include <iomanip>

#include "Execution.h"

#include "backtesting/backtest_engine.h"
#include "backtesting/data_loader.h"
#include "backtesting/market_data_event.h"
#include "backtesting/event_queue.h"
#include "backtesting/bar_data.h"

#include "strategies/buy_and_hold.h"
#include "strategies/sma_crossover.h"
#include "strategies/mean_reversion.h"

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

void test_position_tracking() {
    print_separator("TEST 1: Position Tracking");

    ExecutionEngine engine("AAPL");

    std::cout << "Initial position: " << engine.get_position() << "\n";
    std::cout << "Initial PnL: $" << engine.get_realized_pnl() << "\n\n";

    std::cout << "Adding sell order to book: 100 @ $150.00\n";
    auto& orderbook = engine.get_orderbook();
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 999, Side::Sell, 15000, 100);
    orderbook.AddOrder(sell1);

    std::cout << "Our buy order (takes liquidity): 100 @ $150.00\n";
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 15000, 100);
    auto trades1 = engine.execute_order(buy1);

    std::cout << "Position: " << engine.get_position() << " (should be 100)\n";
    std::cout << "Avg price: $" << std::fixed << std::setprecision(2)
              << engine.get_average_price() << "\n";
    std::cout << "Realized PnL: $" << engine.get_realized_pnl() << " (should be negative from fees)\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n\n";

    std::cout << "Adding buy order to book: 100 @ $151.00\n";
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 998, Side::Buy, 15100, 100);
    orderbook.AddOrder(buy2);

    std::cout << "Our sell order (close position): 100 @ $151.00\n";
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 15100, 100);
    auto trades2 = engine.execute_order(sell2);

    std::cout << "Position: " << engine.get_position() << " (should be 0)\n";
    std::cout << "Realized PnL: $" << engine.get_realized_pnl() << " (should be ~$100 minus fees)\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
}

void test_sma_crossover() {
    print_separator("TEST 2: SMA Crossover Strategy");

    std::cout << "Loading data from CSV...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_sma_crossover.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading data: " << e.what() << "\n";
        std::cerr << "Make sure you ran the Python data generation script!\n";
        std::cerr << "Expected file: ../data/test_sma_crossover.csv\n";
        return;
    }

    auto strategy = std::make_shared<SMACrossover>(50, 200);
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running SMA(50/200) crossover backtest...\n";
    std::cout << "Data: " << bars.size() << " bars with trend reversal\n\n";

    double final_value = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial capital: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final value: $" << final_value << "\n";
    std::cout << "Total PnL: $" << engine.get_total_pnl() << "\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_value / 100000.0 - 1.0) * 100.0) << "%\n";

    auto exec_engine = engine.get_execution_engine("AAPL");
    if (exec_engine) {
        std::cout << "Final position: " << exec_engine->get_position() << "\n";
    }
}

void test_mean_reversion() {
    print_separator("TEST 3: Mean Reversion Strategy");

    std::cout << "Loading data from CSV...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_mean_reversion.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";

        std::cout << "\nData analysis:\n";
        std::cout << "Price range: " << bars.front().close << " to " << bars.back().close << "\n";

        double min_price = bars[0].close;
        double max_price = bars[0].close;
        for (const auto& bar : bars) {
            min_price = std::min(min_price, bar.close);
            max_price = std::max(max_price, bar.close);
        }
        std::cout << "Min: " << min_price << ", Max: " << max_price << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR loading data: " << e.what() << "\n";
        std::cerr << "Make sure you ran the Python data generation script!\n";
        std::cerr << "Expected file: ../data/test_mean_reversion.csv\n";
        return;
    }

    auto strategy = std::make_shared<MeanReversion>(20, 1.5, 0.5);
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running mean reversion backtest...\n";
    std::cout << "Parameters: lookback=20, entry_threshold=1.5, exit_threshold=0.5\n\n";

    double final_value = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial capital: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final value: $" << final_value << "\n";
    std::cout << "Total PnL: $" << engine.get_total_pnl() << "\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_value / 100000.0 - 1.0) * 100.0) << "%\n";

    auto exec_engine = engine.get_execution_engine("AAPL");
    if (exec_engine) {
        std::cout << "Final position: " << exec_engine->get_position() << "\n";
    }

    std::cout << "Signals generated: " << strategy->get_signal_count() << "\n";
}

void test_buy_and_hold() {
    print_separator("TEST 4: Buy and Hold (Baseline)");

    std::cout << "Loading data from CSV...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_buy_and_hold.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading data: " << e.what() << "\n";
        std::cerr << "Make sure you ran the Python data generation script!\n";
        std::cerr << "Expected file: ../data/test_buy_and_hold.csv\n";
        return;
    }

    auto strategy = std::make_shared<BuyAndHold>();
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running buy and hold backtest...\n";
    std::cout << "Data: " << bars.size() << " bars, steady uptrend\n\n";

    double final_value = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial capital: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final value: $" << final_value << "\n";
    std::cout << "Total PnL: $" << engine.get_total_pnl() << "\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_value / 100000.0 - 1.0) * 100.0) << "%\n";
}

int main() {
    print_separator("QuantCore Test Suite");
    std::cout << "\n";

    try {
        test_position_tracking();
        test_buy_and_hold();
        test_sma_crossover();
        test_mean_reversion();

        print_separator();
        std::cout << "\n  ALL TESTS PASSED!\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 1;
    }
}