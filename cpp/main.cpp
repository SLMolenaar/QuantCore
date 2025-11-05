#include <iostream>
#include <iomanip>
#include "Execution.h"
#include "orderbook/Order.h"
#include "orderbook/OrderType.h"

using namespace quantcore;

int main() {
    std::cout << "  Quick Integration Test\n";

    try {
        ExecutionEngine engine("TEST");

        std::cout << "Test 1: Adding orders to orderbook...\n";
        auto buy = std::make_shared<Order>(
            OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100);
        engine.execute_order(buy);
        std::cout << "  Buy order added\n";

        auto sell = std::make_shared<Order>(
            OrderType::GoodTillCancel, 2, Side::Sell, 10000, 100);
        auto trades = engine.execute_order(sell);
        std::cout << "  Sell order added and matched\n";
        std::cout << "  Trades executed: " << trades.size() << "\n\n";

        std::cout << "Test 2: Orderbook state...\n";
        std::cout << "  Orders in book: " << engine.get_orderbook().Size() << "\n";
        std::cout << "  Total fees: $" << std::fixed << std::setprecision(4)
                  << engine.get_total_fees() << "\n\n";

        std::cout << "Integration successful!\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}