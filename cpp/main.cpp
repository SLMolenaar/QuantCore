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
#include "orderbook/Ordertype.h"

using namespace quantcore;

void print_title(const std::string& title = "") {
    std::cout << "\n" << std::string(60, '=') << "\n";
    if (!title.empty()) {
        std::cout << "  " << title << "\n";
        std::cout << std::string(60, '=') << "\n";
    }
}

void test_position_tracking() {
    print_title("TEST 1: Position Tracking");

    ExecutionEngine engine("AAPL");

    std::cout << "Initial position: " << engine.get_position() << "\n";
    std::cout << "Initial PnL: $" << engine.get_realized_pnl() << "\n\n";

    // Add sell order to book
    std::cout << "Adding sell order to book: 100 @ $150.00\n";
    auto& orderbook = engine.get_orderbook();
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 999, Side::Sell, 15000, 100);
    orderbook.AddOrder(sell1);

    // Buy to open position
    std::cout << "Our buy order (takes liquidity): 100 @ $150.00\n";
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 15000, 100);
    auto trades1 = engine.execute_order(buy1);

    std::cout << "Position: " << engine.get_position() << "\n";
    std::cout << "Avg price: $" << std::fixed << std::setprecision(2)
              << engine.get_average_price() << "\n";
    std::cout << "Realized PnL: $" << engine.get_realized_pnl() << " (fees)\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n\n";

    // Add buy to book
    std::cout << "Adding buy order to book: 100 @ $151.00\n";
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 998, Side::Buy, 15100, 100);
    orderbook.AddOrder(buy2);

    // Sell to close
    std::cout << "Our sell order (close position): 100 @ $151.00\n";
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 15100, 100);
    auto trades2 = engine.execute_order(sell2);

    std::cout << "Position: " << engine.get_position() << "\n";
    std::cout << "Realized PnL: $" << engine.get_realized_pnl() << "\n";
    std::cout << "Total fees: $" << engine.get_total_fees() << "\n";
}

void test_sma_crossover() {
    print_title("TEST 2: SMA Crossover Strategy");

    std::cout << "Loading data from CSV...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_sma_crossover.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cerr << "Did you run the Python data script?\n";
        std::cerr << "Expected: ../data/test_sma_crossover.csv\n";
        return;
    }

    auto strategy = std::make_shared<SMACrossover>(50, 200);
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running SMA(50/200) crossover...\n";
    std::cout << "Data: " << bars.size() << " bars\n\n";

    double final_val = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final: $" << final_val << "\n";
    std::cout << "P&L: $" << engine.get_total_pnl() << "\n";
    std::cout << "Fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_val / 100000.0 - 1.0) * 100.0) << "%\n";

    auto exec = engine.get_execution_engine("AAPL");
    if (exec) {
        std::cout << "Final pos: " << exec->get_position() << "\n";
    }
}

void test_mean_reversion() {
    print_title("TEST 3: Mean Reversion Strategy");

    std::cout << "Loading data...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_mean_reversion.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";

        // Quick stats
        double min_px = bars[0].close;
        double max_px = bars[0].close;
        for (const auto& bar : bars) {
            min_px = std::min(min_px, bar.close);
            max_px = std::max(max_px, bar.close);
        }
        std::cout << "Range: $" << min_px << " - $" << max_px << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cerr << "Need: ../data/test_mean_reversion.csv\n";
        return;
    }

    auto strategy = std::make_shared<MeanReversion>(20, 1.5, 0.5);
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running mean reversion...\n";
    std::cout << "Params: lookback=20, entry=1.5, exit=0.5\n\n";

    double final_val = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final: $" << final_val << "\n";
    std::cout << "P&L: $" << engine.get_total_pnl() << "\n";
    std::cout << "Fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_val / 100000.0 - 1.0) * 100.0) << "%\n";

    auto exec = engine.get_execution_engine("AAPL");
    if (exec) {
        std::cout << "Final pos: " << exec->get_position() << "\n";
    }

    std::cout << "Signals: " << strategy->get_signal_count() << "\n";
}

void test_buy_and_hold() {
    print_title("TEST 4: Buy and Hold (Baseline)");

    std::cout << "Loading data...\n";

    BarSeries bars;
    try {
        bars = CSVDataLoader::load("../data/test_buy_and_hold.csv", "AAPL");
        std::cout << "Loaded " << bars.size() << " bars\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cerr << "Need: ../data/test_buy_and_hold.csv\n";
        return;
    }

    auto strategy = std::make_shared<BuyAndHold>();
    BacktestEngine engine(100000.0);

    engine.add_data("AAPL", bars);
    engine.set_strategy(strategy);

    std::cout << "Running buy & hold...\n\n";

    double final_val = engine.run();

    std::cout << "Results:\n";
    std::cout << "Initial: $" << std::fixed << std::setprecision(2) << 100000.0 << "\n";
    std::cout << "Final: $" << final_val << "\n";
    std::cout << "P&L: $" << engine.get_total_pnl() << "\n";
    std::cout << "Fees: $" << engine.get_total_fees() << "\n";
    std::cout << "Return: " << std::setprecision(2)
              << ((final_val / 100000.0 - 1.0) * 100.0) << "%\n";
}

int main() {
    print_title("QuantCore Test Suite");
    std::cout << "\n";

    try {
        test_position_tracking();
        test_buy_and_hold();
        test_sma_crossover();
        test_mean_reversion();

        print_title();
        std::cout << "\n  ALL TESTS PASSED!\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 1;
    }
}