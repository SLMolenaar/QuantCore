/**
 * Tests for ExecutionEngine, Position Tracking & PnL Calc
 * Tests use exact expected values with manual calculations.
 */

#include <gtest/gtest.h>
#include "Execution.h"
#include "orderbook/Order.h"
#include "orderbook/OrderType.h"
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

    // add liquidity to book (becomes maker)
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

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

TEST_F(ExecutionEngineTest, InitialState) {
    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);
    EXPECT_EQ(engine_->get_realized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_unrealized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_fees(), 0.0);
}

TEST_F(ExecutionEngineTest, OpenLongPosition) {
    add_sell_liquidity(10000, 100);  // $100.00 ask

    // Buy 100 shares @ $100.00 (taker)
    auto buy_order = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );

    auto trades = engine_->execute_order(buy_order);

    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(engine_->get_position(), 100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // Taker fee: 100 shares * $100 * 0.002 = $20.00
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 20.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -20.0);  // Just fees so far
}

TEST_F(ExecutionEngineTest, OpenShortPosition) {
    add_buy_liquidity(10000, 100);  // $100.00 bid

    // Sell 100 shares @ $100.00 (taker)
    auto sell_order = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );

    auto trades = engine_->execute_order(sell_order);

    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(engine_->get_position(), -100.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // Taker fee: 100 * $100 * 0.002 = $20.00
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 20.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -20.0);
}

// ============================================================================
// ADDING TO POSITIONS
// ============================================================================

TEST_F(ExecutionEngineTest, AddToLongPosition) {
    // First trade: Buy 100 at $100
    add_sell_liquidity(10000, 100);
    auto buy1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy1);

    // Second trade: Buy 50 @ $110
    add_sell_liquidity(11000, 50);
    auto buy2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 11000, 50
    );
    engine_->execute_order(buy2);

    EXPECT_EQ(engine_->get_position(), 150.0);

    // Average: (100*100 + 50*110) / 150 = 15500 / 150 = 103.333...
    EXPECT_NEAR(engine_->get_average_price(), 103.333333, 1e-5);

    // Fees: (100*100*0.002) + (50*110*0.002) = 20 + 11 = $31
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 31.0);
}

TEST_F(ExecutionEngineTest, AddToShortPosition) {
    // First: Sell 100 @ $100
    add_buy_liquidity(10000, 100);
    auto sell1 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell1);

    // Second: Sell 50 @ $95
    add_buy_liquidity(9500, 50);
    auto sell2 = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 9500, 50
    );
    engine_->execute_order(sell2);

    EXPECT_EQ(engine_->get_position(), -150.0);

    // Average: (100*100 + 50*95) / 150 = 14750 / 150 = 98.333...
    EXPECT_NEAR(engine_->get_average_price(), 98.333333, 1e-5);
}

// ============================================================================
// CLOSING POSITIONS - PNL REALIZATION
// ============================================================================

TEST_F(ExecutionEngineTest, CloseLongPositionWithProfit) {
    // Buy 100 at $100
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Sell 100 at $110 (close with profit)
    add_buy_liquidity(11000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 100
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);

    // PnL: 100 shares * ($110 - $100) = $1000
    // Fees: (100*100*0.002) + (100*110*0.002) = 20 + 22 = $42
    // Realized PnL: $1000 - $42 = $958
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 42.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 958.0);
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, CloseLongPositionWithLoss) {
    // Buy 100 @ $100
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Sell 100 @ $90 (close with loss)
    add_buy_liquidity(9000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 9000, 100
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($90 - $100) = -$1000
    // Fees: (100*100*0.002) + (100*90*0.002) = 20 + 18 = $38
    // Realized PnL: -$1000 - $38 = -$1038
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 38.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -1038.0);
}

TEST_F(ExecutionEngineTest, CloseShortPositionWithProfit) {
    // Sell 100 @ $100 (open short)
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    // Buy 100 @ $90 (cover short with profit)
    add_sell_liquidity(9000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($100 - $90) = $1000 (short profit)
    // Fees: (100*100*0.002) + (100*90*0.002) = 20 + 18 = $38
    // Realized PnL: $1000 - $38 = $962
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 38.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 962.0);
}

TEST_F(ExecutionEngineTest, CloseShortPositionWithLoss) {
    // Sell 100 @ $100 (open short)
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    // Buy 100 @ $110 (cover short with loss)
    add_sell_liquidity(11000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 11000, 100
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 0.0);

    // PnL: 100 * ($100 - $110) = -$1000 (short loss)
    // Fees: (100*100*0.002) + (100*110*0.002) = 20 + 22 = $42
    // Realized PnL: -$1000 - $42 = -$1042
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 42.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), -1042.0);
}

// ============================================================================
// PARTIAL CLOSES
// ============================================================================

TEST_F(ExecutionEngineTest, PartialCloseLong) {
    // Buy 100 @ $100
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Sell 60 @ $110 (partial close)
    add_buy_liquidity(11000, 60);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 60
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), 40.0);  // 100 - 60
    EXPECT_EQ(engine_->get_average_price(), 100.0);  // Original avg remains

    // PnL on closed portion: 60 * ($110 - $100) = $600
    // Fees: (100*100*0.002) + (60*110*0.002) = 20 + 13.2 = $33.2
    // Realized PnL: $600 - $33.2 = $566.8
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 33.2);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 566.8);
}

TEST_F(ExecutionEngineTest, PartialCloseShort) {
    // Sell 100 @ $100 (open short)
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    // Buy 60 @ $90 (partial cover)
    add_sell_liquidity(9000, 60);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 60
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), -40.0);  // Still short 40
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // PnL on covered portion: 60 * ($100 - $90) = $600
    // Fees: (100*100*0.002) + (60*90*0.002) = 20 + 10.8 = $30.8
    // Realized PnL: $600 - $30.8 = $569.2
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 30.8);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 569.2);
}

// ============================================================================
// POSITION FLIPS
// ============================================================================

TEST_F(ExecutionEngineTest, FlipLongToShort) {
    // Buy 100 @ $100 (long)
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Sell 150 @ $110 (close long + open short 50)
    add_buy_liquidity(11000, 150);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Sell, 11000, 150
    );
    engine_->execute_order(sell);

    EXPECT_EQ(engine_->get_position(), -50.0);  // Now short 50
    EXPECT_EQ(engine_->get_average_price(), 110.0);  // New short at $110

    // PnL on closed long: 100 * ($110 - $100) = $1000
    // Fees: (100*100*0.002) + (150*110*0.002) = 20 + 33 = $53
    // Realized PnL: $1000 - $53 = $947
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 53.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 947.0);
}

TEST_F(ExecutionEngineTest, FlipShortToLong) {
    // Sell 100 @ $100 (short)
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    // Buy 150 @ $90 (cover short + open long 50)
    add_sell_liquidity(9000, 150);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2, Side::Buy, 9000, 150
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 50.0);  // Now long 50
    EXPECT_EQ(engine_->get_average_price(), 90.0);  // New long at $90

    // PnL on closed short: 100 * ($100 - $90) = $1000
    // Fees: (100*100*0.002) + (150*90*0.002) = 20 + 27 = $47
    // Realized PnL: $1000 - $47 = $953
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 47.0);
    EXPECT_DOUBLE_EQ(engine_->get_realized_pnl(), 953.0);
}

// ============================================================================
// UNREALIZED PNL
// ============================================================================

TEST_F(ExecutionEngineTest, UnrealizedPnlLongPosition) {
    // Buy 100 @ $100
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Add new quotes at $110
    add_sell_liquidity(11000, 100, 5000);
    add_buy_liquidity(10900, 100, 5001);

    // Unrealized PnL: 100 * ($105 - $100) = $500 (using mid price $105)
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 500.0);

    // Total PnL = Realized + Unrealized
    // Realized = -$20 (fees), Unrealized = $500
    EXPECT_DOUBLE_EQ(engine_->get_total_pnl(), 480.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlShortPosition) {
    // Sell 100 @ $100
    add_buy_liquidity(10000, 100);
    auto sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(sell);

    // Add new quotes at $95
    add_buy_liquidity(9400, 100, 5000);
    add_sell_liquidity(9600, 100, 5001);

    // Unrealized PnL: 100 * ($100 - $95) = $500 (using mid price $95)
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 500.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlNoPosition) {
    // No position = no unrealized PnL
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

TEST_F(ExecutionEngineTest, UnrealizedPnlEmptyBook) {
    // Buy 100 @ $100
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // No quotes in book → no unrealized PnL calculation
    EXPECT_DOUBLE_EQ(engine_->get_unrealized_pnl(), 0.0);
}

// ============================================================================
// MAKER VS TAKER FEES
// ============================================================================

TEST_F(ExecutionEngineTest, MakerFees) {
    // Place resting order (becomes maker)
    auto our_sell = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Sell, 10000, 100
    );
    engine_->execute_order(our_sell);

    // Someone else buys (we're the maker)
    add_buy_liquidity(10000, 100, 2000);
    auto their_buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 2000, Side::Buy, 10000, 100
    );
    auto trades = engine_->get_orderbook().AddOrder(their_buy);

    // Manual update since this is external matching
    for (const auto& trade : trades) {
        // In real system, ExecutionEngine would track this
        // For now, verify maker fee would be: 100 * $100 * 0.001 = $10
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(ExecutionEngineTest, MultipleRoundTrips) {
    // Round trip 1: Buy 100 @ $100, Sell @ $110
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

    // Round trip 2: Buy 50 @ $105, Sell @ $115
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

    // Both round trips should accumulate PnL
    EXPECT_GT(engine_->get_realized_pnl(), pnl_after_first);
    EXPECT_EQ(engine_->get_position(), 0.0);
}

TEST_F(ExecutionEngineTest, VerySmallPosition) {
    // 1 share @ $100
    add_sell_liquidity(10000, 1);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1.0);
    EXPECT_EQ(engine_->get_average_price(), 100.0);

    // Fee: 1 * $100 * 0.002 = $0.20
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 0.2);
}

TEST_F(ExecutionEngineTest, VeryLargePosition) {
    // 1 million shares @ $100
    add_sell_liquidity(10000, 1000000);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 1000000
    );
    engine_->execute_order(buy);

    EXPECT_EQ(engine_->get_position(), 1000000.0);

    // Fee: 1M * $100 * 0.002 = $200,000
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), 200000.0);
}

TEST_F(ExecutionEngineTest, Reset) {
    // Create some position and PnL
    add_sell_liquidity(10000, 100);
    auto buy = std::make_shared<Order>(
        OrderType::GoodTillCancel, 1, Side::Buy, 10000, 100
    );
    engine_->execute_order(buy);

    // Reset should clear everything
    engine_->reset();

    EXPECT_EQ(engine_->get_position(), 0.0);
    EXPECT_EQ(engine_->get_average_price(), 0.0);
    EXPECT_EQ(engine_->get_realized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_unrealized_pnl(), 0.0);
    EXPECT_EQ(engine_->get_total_fees(), 0.0);
}

TEST_F(ExecutionEngineTest, GetMidPrice) {
    // No quotes
    EXPECT_FALSE(engine_->get_mid_price().has_value());

    // Add quotes
    add_buy_liquidity(9900, 100);   // $99 bid
    add_sell_liquidity(10100, 100); // $101 ask

    auto mid = engine_->get_mid_price();
    EXPECT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid.value(), 100.0);  // ($99 + $101) / 2
}

TEST_F(ExecutionEngineTest, GetBestBidAsk) {
    EXPECT_EQ(engine_->get_best_bid(), 0);
    EXPECT_EQ(engine_->get_best_ask(), 0);

    add_buy_liquidity(9900, 100);
    add_sell_liquidity(10100, 100);

    EXPECT_EQ(engine_->get_best_bid(), 9900);  // In cents
    EXPECT_EQ(engine_->get_best_ask(), 10100);
}