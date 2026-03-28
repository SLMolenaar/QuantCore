/**
 * Tests for ExecutionEngine, Position Tracking & PnL Calc
 * Tests use exact expected values with manual calculations.
 */

#include <gtest/gtest.h>
#include "Execution.h"
#include "orderbook/Order.h"
#include "orderbook/Ordertype.h"
#include <memory>
#include <cmath>

using namespace quantcore;

class ExecutionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.maker_fee = 0.001;  // 0.1%
        config_.taker_fee = 0.002;  // 0.2%
        engine_ = std::make_unique<ExecutionEngine>("TEST", config_);
    }

    void add_sell_liquidity(Price price, Quantity quantity, OrderId order_id = 9999) {
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel, order_id, Side::Sell, price, quantity
        );
        engine_->get_orderbook().AddOrder(order);
    }

    void add_buy_liquidity(Price price, Quantity quantity, OrderId order_id = 9998) {
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel, order_id, Side::Buy, price, quantity
        );
        engine_->get_orderbook().AddOrder(order);
    }

    ExecutionConfig config_;
    std::unique_ptr<ExecutionEngine> engine_;
};

TEST_F(ExecutionEngineTest, InitialState) {
    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);
    EXPECT_EQ(engine_->get_realized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_unrealized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_fees(), 0.0);
}

TEST_F(ExecutionEngineTest, OpenLongPosition) {
    add_sell_liquidity(10000, 100);

    auto buy_order = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    auto trades = engine_->execute_order(buy_order);

    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // taker fee: 100 * $100 * 0.002 = $20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 20.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -20.0);
}

TEST_F(ExecutionEngineTest, OpenShortPosition) {
    add_buy_liquidity(10000, 100);

    auto sell_order = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    auto trades = engine_->execute_order(sell_order);

    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(engine_->get_position(), -100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // taker fee: 100 * $100 * 0.002 = $20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 20.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -20.0);
}

TEST_F(ExecutionEngineTest, AddToLongPosition) {
    add_sell_liquidity(10000, 100);
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy1);

    add_sell_liquidity(11000, 50);
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 11000, 50
    );
    engine_->execute_order(buy2);

    EXPECT_EQ(engine_->get_position(), 150.0);

    // avg: (100*100 + 50*110) / 150 = 15500/150 = 103.333...
    EXPECT_NEAR(engine_->get_average_price(), 103.333333, 1e-5);

    // fees: (100*100*0.002) + (50*110*0.002) = 20 + 11 = $31
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 31.0);
}

TEST_F(ExecutionEngineTest, AddToShortPosition) {
    add_buy_liquidity(10000, 100);
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell1);

    add_buy_liquidity(9500, 50);
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 9500, 50
    );
    engine_->execute_order(sell2);

    EXPECT_EQ(engine_->get_position(), -150.0);

    // avg: (100*100 + 50*95) / 150 = 14750/150 = 98.333...
    EXPECT_NEAR(engine_->get_average_price(), 98.333333, 1e-5);
}

TEST_F(ExecutionEngineTest, CloseLongPositionWithProfit) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    add_buy_liquidity(11000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 100
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);

    // PnL: 100 * ($110 - $100) = $1000
    // fees: (100*100*0.002) + (100*110*0.002) = 20 + 22 = $42
    // realized: $1000 - $42 = $958
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 42.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 958.0);
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, CloseLongPositionWithLoss) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    add_buy_liquidity(9000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 9000, 100
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($90 - $100) = -$1000
    // fees: (100*100*0.002) + (100*90*0.002) = 20 + 18 = $38
    // realized: -$1000 - $38 = -$1038
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 38.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -1038.0);
}

TEST_F(ExecutionEngineTest, CloseShortPositionWithProfit) {
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    add_sell_liquidity(9000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($100 - $90) = $1000
    // fees: (100*100*0.002) + (100*90*0.002) = 20 + 18 = $38
    // realized: $1000 - $38 = $962
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 38.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 962.0);
}

TEST_F(ExecutionEngineTest, CloseShortPositionWithLoss) {
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    add_sell_liquidity(11000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 11000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($100 - $110) = -$1000
    // fees: (100*100*0.002) + (100*110*0.002) = 20 + 22 = $42
    // realized: -$1000 - $42 = -$1042
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 42.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -1042.0);
}

TEST_F(ExecutionEngineTest, PartialCloseLong) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    add_buy_liquidity(11000, 60);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 60
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 40.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // PnL on closed portion: 60 * ($110 - $100) = $600
    // fees: (100*100*0.002) + (60*110*0.002) = 20 + 13.2 = $33.2
    // realized: $600 - $33.2 = $566.8
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 33.2);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 566.8);
}

TEST_F(ExecutionEngineTest, PartialCloseShort) {
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    add_sell_liquidity(9000, 60);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 60
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), -40.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // PnL on covered portion: 60 * ($100 - $90) = $600
    // fees: (100*100*0.002) + (60*90*0.002) = 20 + 10.8 = $30.8
    // realized: $600 - $30.8 = $569.2
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 30.8);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 569.2);
}

TEST_F(ExecutionEngineTest, FlipLongToShort) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    add_buy_liquidity(11000, 150);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 150
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), -50.0);
    EXPECT_EQ(engine_->get_average_price(), 110.0);

    // PnL on closed long: 100 * ($110 - $100) = $1000
    // fees: (100*100*0.002) + (150*110*0.002) = 20 + 33 = $53
    // realized: $1000 - $53 = $947
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 53.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 947.0);
}

TEST_F(ExecutionEngineTest, FlipShortToLong) {
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    add_sell_liquidity(9000, 150);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 150
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 50.0);
    EXPECT_EQ(engine_->get_average_price(), 90.0);

    // PnL on closed short: 100 * ($100 - $90) = $1000
    // fees: (100*100*0.002) + (150*90*0.002) = 20 + 27 = $47
    // realized: $1000 - $47 = $953
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 47.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 953.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlLongPosition) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    add_sell_liquidity(11000, 100, 5000);
    add_buy_liquidity(10900, 100, 5001);

    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 950.0);
    EXPECT_DOUBLE_EQ(engine_->get_total_pnl(), 930.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlShortPosition) {
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    add_buy_liquidity(9400, 100, 5000);
    add_sell_liquidity(9600, 100, 5001);

    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 500.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlNoPosition) {
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlEmptyBook) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, MultipleRoundTrips) {
    add_sell_liquidity(10000, 100);
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy1);

    add_buy_liquidity(11000, 100);
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 100
    );
    engine_->execute_order(sell1);

    double pnl_after_first = engine_->get_realized_pnl();

    add_sell_liquidity(10500, 50);
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 3, Side::Buy, 10500, 50
    );
    engine_->execute_order(buy2);

    add_buy_liquidity(11500, 50);
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 4, Side::Sell, 11500, 50
    );
    engine_->execute_order(sell2);

    EXPECT_GT(engine_->get_realized_pnl(), pnl_after_first);
    EXPECT_EQ(engine_->get_position(), 0.0);
}

TEST_F(ExecutionEngineTest, VerySmallPosition) {
    add_sell_liquidity(10000, 1);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // fee: 1 * $100 * 0.002 = $0.20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 0.2);
}

TEST_F(ExecutionEngineTest, VeryLargePosition) {
    add_sell_liquidity(10000, 1000000);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1000000
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1000000.0);

    // fee: 1M * $100 * 0.002 = $200,000
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 200000.0);
}

TEST_F(ExecutionEngineTest, Reset) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    engine_->reset();

    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);
    EXPECT_EQ(engine_->get_realized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_unrealized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_fees(), 0.0);
}

TEST_F(ExecutionEngineTest, GetMidPrice) {
    EXPECT_FALSE(engine_->get_mid_price().has_value());

    add_buy_liquidity(9900, 100);
    add_sell_liquidity(10100, 100);

    auto mid = engine_->get_mid_price();
    EXPECT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid.value(), 100.0);
}

TEST_F(ExecutionEngineTest, GetBestBidAsk) {
    EXPECT_EQ(engine_->get_best_bid(), 0);
    EXPECT_EQ(engine_->get_best_ask(), 0);

    add_buy_liquidity(9900, 100);
    add_sell_liquidity(10100, 100);

    EXPECT_EQ(engine_->get_best_bid(), 9900);
    EXPECT_EQ(engine_->get_best_ask(), 10100);
}

TEST_F(ExecutionEngineTest, PnLConsistencyInvariant) {
    add_sell_liquidity(10000, 100);
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy1);

    add_sell_liquidity(10500, 50);
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 10500, 50
    );
    engine_->execute_order(buy2);

    add_buy_liquidity(11000, 75);
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 3, Side::Sell, 11000, 75
    );
    engine_->execute_order(sell1);

    add_buy_liquidity(10800, 100);
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 4, Side::Sell, 10800, 100
    );
    engine_->execute_order(sell2);

    double avg_after_2 = (100.0 * 100.0 + 50.0 * 105.0) / 150.0;
    double realized_from_trade3 = 75.0 * (110.0 - avg_after_2);
    double realized_from_trade4 = 75.0 * (108.0 - avg_after_2);
    double expected_realized = realized_from_trade3 + realized_from_trade4 - 68.60;

    EXPECT_NEAR(engine_->get_realized_pnl(), expected_realized, 0.01);

    add_buy_liquidity(10750, 100, 5000);
    add_sell_liquidity(10850, 100, 5001);

    // short 25 @ $108, mid $108: unrealized = 0
    EXPECT_NEAR(engine_->get_unrealized_pnl(), 0.0, 0.01);

    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl() + engine_->get_unrealized_pnl(),
                     engine_->get_total_pnl());
}

TEST_F(ExecutionEngineTest, ThousandSmallTradesNoRoundingAccumulation) {
    ExecutionEngine engine1("TEST", config_);
    for (int i = 0; i < 1000; ++i) {
        auto& ob1 = engine1.get_orderbook();
        auto sell = std::make_shared<Order>(
            OrderType::GoodTillCancel, 10000 + i, Side::Sell, 10000, 1
        );
        ob1.AddOrder(sell);

        auto buy = std::make_shared<Order>(
            OrderType::GoodTillCancel, i + 1, Side::Buy, 10000, 1
        );
        engine1.execute_order(buy);
    }

    ExecutionEngine engine2("TEST", config_);
    auto& ob2 = engine2.get_orderbook();
    auto sell_large = std::make_shared<Order>(
        OrderType::GoodTillCancel, 99999, Side::Sell, 10000, 1000
    );
    ob2.AddOrder(sell_large);

    auto buy_large = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1000
    );
    engine2.execute_order(buy_large);

    EXPECT_DOUBLE_EQ(engine1.get_position(), 1000.0);
    EXPECT_DOUBLE_EQ(engine2.get_position(), 1000.0);
    EXPECT_DOUBLE_EQ(engine1.get_average_price(), 100.0);
    EXPECT_DOUBLE_EQ(engine2.get_average_price(), 100.0);
    EXPECT_NEAR(engine1.get_total_fees(), 200.0, 0.01);
    EXPECT_NEAR(engine2.get_total_fees(), 200.0, 0.01);
    EXPECT_NEAR(engine1.get_realized_pnl(), -200.0, 0.01);
    EXPECT_NEAR(engine2.get_realized_pnl(), -200.0, 0.01);
}

TEST_F(ExecutionEngineTest, VeryLargePositionValuesBillions) {
    auto& ob = engine_->get_orderbook();

    ExchangeRules rules;
    rules.maxQuantity = 100000000;
    ob.SetExchangeRules(rules);

    auto sell_order = std::make_shared<Order>(
        OrderType::GoodTillCancel, 10000, Side::Sell, 10000, 10000000
    );
    ob.AddOrder(sell_order);

    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 10000000
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 10000000.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 2000000.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -2000000.0);

    auto buy_liquidity = std::make_shared<Order>(
        OrderType::GoodTillCancel, 10001, Side::Buy, 10100, 10000000
    );
    ob.AddOrder(buy_liquidity);

    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 10100, 10000000
    );
    engine_->execute_order(sell);

    // gross profit: 10M * $1 = $10M; total fees: 2M + 2.02M = 4.02M
    double expected_net = 10000000.0 - 4020000.0;
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), expected_net);
    EXPECT_EQ(engine_->get_position(), 0.0);
}

TEST_F(ExecutionEngineTest, PennyStockLowPricePrecision) {
    add_sell_liquidity(50, 1000);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 50, 1000
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1000.0);
    EXPECT_DOUBLE_EQ(engine_->get_average_price(), 0.50);
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 1.0);

    add_buy_liquidity(51, 1000);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 51, 1000
    );
    engine_->execute_order(sell);

    EXPECT_NEAR(engine_->get_realized_pnl(), 7.98, 0.0001);
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 2.02);
}

TEST_F(ExecutionEngineTest, OneCentStockMinimumPrice) {
    add_sell_liquidity(1, 10000);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 1, 10000
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 10000.0);
    EXPECT_DOUBLE_EQ(engine_->get_average_price(), 0.01);

    // fee: 10000 * $0.01 * 0.002 = $0.20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 0.20);

    add_buy_liquidity(2, 10000);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 2, 10000
    );
    engine_->execute_order(sell);

    // gross: 10000 * $0.01 = $100; fees: $0.20 + $0.40 = $0.60; net: $99.40
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 99.40);
}

TEST_F(ExecutionEngineTest, SingleSharePositionClosedResets) {
    add_sell_liquidity(10000, 1);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    add_buy_liquidity(11000, 1);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 1
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);

    // fees: ($100 * 0.002) + ($110 * 0.002) = $0.20 + $0.22 = $0.42
    // net: $10 - $0.42 = $9.58
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 9.58);
}

TEST_F(ExecutionEngineTest, FeesCalculatedPerFill) {
    auto& ob = engine_->get_orderbook();

    ob.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, 1001, Side::Sell, 10000, 30));
    ob.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, 1002, Side::Sell, 10000, 40));
    ob.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, 1003, Side::Sell, 10000, 30));

    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    auto trades = engine_->execute_order(buy);

    EXPECT_EQ(trades.size(), 3);

    // fees per fill: 30*$100*0.002 + 40*$100*0.002 + 30*$100*0.002 = 6 + 8 + 6 = $20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 20.0);
    EXPECT_EQ(engine_->get_position(), 100.0);
}

TEST_F(ExecutionEngineTest, RapidPositionFlipsLongShortLong) {
    add_sell_liquidity(10000, 100);
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy1);

    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    add_buy_liquidity(10500, 150);
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 10500, 150
    );
    engine_->execute_order(sell1);

    EXPECT_EQ(engine_->get_position(), -50.0);
    EXPECT_EQ(engine_->get_average_price(), 105.0);

    add_sell_liquidity(10200, 150);
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 3, Side::Buy, 10200, 150
    );
    engine_->execute_order(buy2);

    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_EQ(engine_->get_average_price(), 102.0);

    EXPECT_NEAR(engine_->get_realized_pnl(), 567.9, 0.01);
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 82.1);
}

TEST_F(ExecutionEngineTest, UnrealizedPnLWithNoMarketQuotes) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnLWithOneSidedMarket) {
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    add_sell_liquidity(10500, 100, 5000);

    auto mid = engine_->get_mid_price();
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid.value(), 105.0);

    // unrealized: 100 * ($105 - $100) = $500
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 500.0);
}

TEST_F(ExecutionEngineTest, ZeroQuantityOrderRejected) {
    EXPECT_THROW({
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel, 1, Side::Buy, 10000, 0
        );
    }, std::invalid_argument);
}
