/**
 * Tests for Orderbook - Core Matching Engine
 */

#include <gtest/gtest.h>
#include "orderbook/Orderbook.h"
#include "orderbook/Order.h"
#include "orderbook/OrderModify.h"
#include "orderbook/OrderType.h"
#include "orderbook/Trade.h"
#include "orderbook/Types.h"
#include "orderbook/ExchangeRules.h"
#include <memory>
#include <algorithm>

class OrderbookTest : public ::testing::Test {
protected:
    void SetUp() override {
        orderbook_ = std::make_unique<Orderbook>();
        ASSERT_EQ(orderbook_->Size(), 0) << "Orderbook not empty in SetUp!";
    }

    // Helper to create buy order
    OrderPointer create_buy_order(OrderId id, Price price, Quantity qty, OrderType type = OrderType::GoodTillCancel) {
        return std::make_shared<Order>(type, id, Side::Buy, price, qty);
    }

    // Helper to create sell order
    OrderPointer create_sell_order(OrderId id, Price price, Quantity qty, OrderType type = OrderType::GoodTillCancel) {
        return std::make_shared<Order>(type, id, Side::Sell, price, qty);
    }

    std::unique_ptr<Orderbook> orderbook_;
};

// ============================================================================
// INITIALIZATION & BASIC OPERATIONS
// ============================================================================

TEST_F(OrderbookTest, InitialStateEmpty) {
    EXPECT_EQ(orderbook_->Size(), 0);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_TRUE(infos.GetBids().empty());
    EXPECT_TRUE(infos.GetAsks().empty());
}

TEST_F(OrderbookTest, AddSingleBuyOrder) {
    // Add buy order: 100 shares @ $100.00
    auto order = create_buy_order(1, 10000, 100);
    auto trades = orderbook_->AddOrder(order);

    // Should not match (no sell orders)
    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);

    // Verify order in book
    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();

    ASSERT_EQ(bids.size(), 1);
    EXPECT_EQ(bids[0].price_, 10000);  // $100.00 in cents
    EXPECT_EQ(bids[0].quantity_, 100);
}

TEST_F(OrderbookTest, AddSingleSellOrder) {
    // Add sell order: 100 shares @ $101.00
    auto order = create_sell_order(1, 10100, 100);
    auto trades = orderbook_->AddOrder(order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    const auto& asks = infos.GetAsks();

    ASSERT_EQ(asks.size(), 1);
    EXPECT_EQ(asks[0].price_, 10100);
    EXPECT_EQ(asks[0].quantity_, 100);
}

TEST_F(OrderbookTest, BidAskSpreadMaintained) {
    // Add orders with spread
    orderbook_->AddOrder(create_buy_order(1, 9900, 100));   // $99.00 bid
    orderbook_->AddOrder(create_sell_order(2, 10100, 100)); // $101.00 ask

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    const auto& asks = infos.GetAsks();

    ASSERT_FALSE(bids.empty());
    ASSERT_FALSE(asks.empty());

    // Best bid < Best ask (no cross)
    EXPECT_LT(bids[0].price_, asks[0].price_);
    EXPECT_EQ(bids[0].price_, 9900);
    EXPECT_EQ(asks[0].price_, 10100);
}

// ============================================================================
// SIMPLE MATCHING
// ============================================================================

TEST_F(OrderbookTest, SimpleMatchExactQuantity) {
    // Add sell order first
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // Buy order matches exactly
    auto buy_order = create_buy_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    // Should generate exactly 1 trade
    ASSERT_EQ(trades.size(), 1);

    const auto& trade = trades[0];
    EXPECT_EQ(trade.GetBidTrade().orderId_, 2);
    EXPECT_EQ(trade.GetAskTrade().orderId_, 1);
    EXPECT_EQ(trade.GetBidTrade().quantity_, 100);
    EXPECT_EQ(trade.GetBidTrade().price_, 10000);

    // Both orders should be filled and removed
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, SimpleMatchBuyerIsTaker) {
    // Sell order rests in book (maker)
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // Buy order crosses spread (taker)
    auto buy_order = create_buy_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);

    // Trade price should be maker's price (seller @ $100.00)
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(trades[0].GetAskTrade().price_, 10000);
}

TEST_F(OrderbookTest, SimpleMatchSellerIsTaker) {
    // Buy order rests in book (maker)
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    // Sell order crosses spread (taker)
    auto sell_order = create_sell_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(sell_order);

    ASSERT_EQ(trades.size(), 1);

    // Trade price should be maker's price (buyer @ $100.00)
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(trades[0].GetAskTrade().price_, 10000);
}

// ============================================================================
// PARTIAL FILLS
// ============================================================================

TEST_F(OrderbookTest, PartialFillBuyOrderTooLarge) {
    // Sell order: 100 shares available
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // Buy order: 150 shares wanted
    auto buy_order = create_buy_order(2, 10000, 150);
    auto trades = orderbook_->AddOrder(buy_order);

    // Should trade 100 shares
    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);

    // Sell order completely filled (removed)
    // Buy order partially filled (50 shares remaining at $100.00)
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();

    ASSERT_EQ(bids.size(), 1);
    EXPECT_EQ(bids[0].price_, 10000);
    EXPECT_EQ(bids[0].quantity_, 50);  // 150 - 100 = 50 remaining
}

TEST_F(OrderbookTest, PartialFillSellOrderTooLarge) {
    // Buy order: 100 shares available
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    // Sell order: 150 shares
    auto sell_order = create_sell_order(2, 10000, 150);
    auto trades = orderbook_->AddOrder(sell_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);

    // Buy order filled, sell order has 50 remaining
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    const auto& asks = infos.GetAsks();

    ASSERT_EQ(asks.size(), 1);
    EXPECT_EQ(asks[0].price_, 10000);
    EXPECT_EQ(asks[0].quantity_, 50);
}

TEST_F(OrderbookTest, MultiplePartialFills) {
    // Add multiple sell orders
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10000, 30));
    orderbook_->AddOrder(create_sell_order(3, 10000, 40));

    // Large buy order that matches all
    auto buy_order = create_buy_order(4, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    // Should generate 3 trades (one per sell order)
    ASSERT_EQ(trades.size(), 3);

    // Verify quantities: 50 + 30 + 20 = 100
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 30);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 20);  // Partial fill of 3rd order

    // First two sell orders filled, third has 20 remaining (40-20)
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 20);
}

// ============================================================================
// PRICE-TIME PRIORITY (FIFO)
// ============================================================================

TEST_F(OrderbookTest, PriceTimePrioritySamePrice) {
    // Add three sell orders at same price
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));  // First
    orderbook_->AddOrder(create_sell_order(2, 10000, 100));  // Second
    orderbook_->AddOrder(create_sell_order(3, 10000, 100));  // Third

    // Buy order matches first one (FIFO)
    auto buy_order = create_buy_order(4, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 1);  // First order matched

    // Orders 2 and 3 remain
    EXPECT_EQ(orderbook_->Size(), 2);
}

TEST_F(OrderbookTest, BestPriceMatchedFirst) {
    // Add sell orders at different prices
    orderbook_->AddOrder(create_sell_order(1, 10200, 100));  // $102.00
    orderbook_->AddOrder(create_sell_order(2, 10000, 100));  // $100.00 (best)
    orderbook_->AddOrder(create_sell_order(3, 10100, 100));  // $101.00

    // Buy at $102 should match best price first ($100)
    auto buy_order = create_buy_order(4, 10200, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 2);  // Best price matched
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);  // Traded at $100, not $102
}

TEST_F(OrderbookTest, PriceThenTimePriority) {
    // Add sell orders
    orderbook_->AddOrder(create_sell_order(1, 10100, 100));  // $101 first
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));   // $100 second (best price)
    orderbook_->AddOrder(create_sell_order(3, 10000, 50));   // $100 third

    // Buy 150 shares at $101
    auto buy_order = create_buy_order(4, 10100, 150);
    auto trades = orderbook_->AddOrder(buy_order);

    // Should generate 3 trades: best price first, then FIFO at same price
    ASSERT_EQ(trades.size(), 3);

    // First: order 2 @ $100 (50 shares)
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 2);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);

    // Second: order 3 @ $100 (50 shares)
    EXPECT_EQ(trades[1].GetAskTrade().orderId_, 3);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().price_, 10000);

    // Third: order 1 @ $101 (50 shares)
    EXPECT_EQ(trades[2].GetAskTrade().orderId_, 1);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[2].GetBidTrade().price_, 10100);
}

// ============================================================================
// ORDER CANCELLATION
// ============================================================================

TEST_F(OrderbookTest, CancelExistingOrder) {
    auto order = create_buy_order(1, 10000, 100);
    orderbook_->AddOrder(order);

    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->CancelOrder(1);

    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelNonExistentOrder) {
    // Should not throw, just no-op
    EXPECT_NO_THROW(orderbook_->CancelOrder(999));
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelPartiallyFilledOrder) {
    // Add sell orders
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));

    // Large buy order (partially filled)
    auto buy_order = create_buy_order(2, 10000, 100);
    orderbook_->AddOrder(buy_order);

    // 50 shares traded, 50 remaining in buy order
    EXPECT_EQ(orderbook_->Size(), 1);

    // Cancel the remaining part
    orderbook_->CancelOrder(2);

    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelMaintainsBookIntegrity) {
    // Add multiple orders
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900, 100));
    orderbook_->AddOrder(create_buy_order(3, 10100, 100));

    // Cancel middle price
    orderbook_->CancelOrder(1);

    EXPECT_EQ(orderbook_->Size(), 2);

    // Verify correct orders remain
    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();

    ASSERT_EQ(bids.size(), 2);
    EXPECT_EQ(bids[0].price_, 10100);  // Best bid
    EXPECT_EQ(bids[1].price_, 9900);
}

// ============================================================================
// ORDER MODIFICATION
// ============================================================================

TEST_F(OrderbookTest, ModifyOrderPrice) {
    auto order = create_buy_order(1, 10000, 100);
    orderbook_->AddOrder(order);

    // Modify to higher price
    OrderModify modify(1, Side::Buy, 10100, 100);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());  // No match
    EXPECT_EQ(orderbook_->Size(), 1);

    // Verify new price
    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetBids()[0].price_, 10100);
}

TEST_F(OrderbookTest, ModifyOrderQuantity) {
    auto order = create_buy_order(1, 10000, 100);
    orderbook_->AddOrder(order);

    // Modify quantity
    OrderModify modify(1, Side::Buy, 10000, 150);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetBids()[0].quantity_, 150);
}

TEST_F(OrderbookTest, ModifyTriggersMatch) {
    // Add sell order
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // Add buy order below market
    orderbook_->AddOrder(create_buy_order(2, 9900, 100));

    // Modify buy order to match
    OrderModify modify(2, Side::Buy, 10000, 100);
    auto trades = orderbook_->MatchOrder(modify);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);

    // Both orders filled
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, ModifyNonExistentOrder) {
    OrderModify modify(999, Side::Buy, 10000, 100);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

// ============================================================================
// ORDER TYPES - MARKET ORDERS
// ============================================================================

TEST_F(OrderbookTest, MarketOrderBuyWithLiquidity) {
    // Add sell orders
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // Market buy order (converted to limit with max price)
    auto market_order = create_buy_order(2, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);

    // Should match at seller's price
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, MarketOrderSellWithLiquidity) {
    // Add buy orders
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    // Market sell order
    auto market_order = create_sell_order(2, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, MarketOrderNoLiquidity) {
    // Empty book
    auto market_order = create_buy_order(1, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    // Should be rejected (no liquidity)
    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, MarketOrderSweepsMultipleLevels) { // TODO: Fix failing test issue.

    orderbook_ = std::make_unique<Orderbook>();  // Force reset

    ASSERT_EQ(orderbook_->Size(), 0) << "Orderbook not empty after reset!";
    auto check = orderbook_->GetOrderInfos();
    ASSERT_EQ(check.GetBids().size(), 0) << "Bids not empty!";
    ASSERT_EQ(check.GetAsks().size(), 0) << "Asks not empty!";

    // Add multiple sell levels
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10100, 50));
    orderbook_->AddOrder(create_sell_order(3, 10200, 50));

    std::cout << "After adding sells, orderbook size: " << orderbook_->Size() << std::endl;

    // Large market buy
    auto market_order = create_buy_order(4, 0, 125, OrderType::Market);
    std::cout << "Market order ID: " << market_order->GetOrderId() << " Price: " << market_order->GetPrice() << std::endl;
    auto trades = orderbook_->AddOrder(market_order);

    // Should match 3 levels: 50@100, 50@101, 25@102
    ASSERT_EQ(trades.size(), 3);

    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);

    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().price_, 10100);

    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 25);
    EXPECT_EQ(trades[2].GetBidTrade().price_, 10200);
}

// ============================================================================
// ORDER TYPES - IMMEDIATE OR CANCEL (IOC)
// ============================================================================

TEST_F(OrderbookTest, IOCFullFill) {
    // Add liquidity
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // IOC order that can be fully filled
    auto ioc_order = create_buy_order(2, 10000, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);

    // Both orders filled
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, IOCPartialFill) {
    // Only 50 shares available
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));

    // IOC wants 100 shares
    auto ioc_order = create_buy_order(2, 10000, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    // Should fill 50 shares
    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);

    // Remaining 50 shares of IOC cancelled (not in book)
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, IOCNoFillCancelled) {
    // No matching orders
    auto ioc_order = create_buy_order(1, 9900, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    // Should be rejected/cancelled immediately
    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

// ============================================================================
// ORDER TYPES - FILL OR KILL (FOK)
// ============================================================================

TEST_F(OrderbookTest, FOKFullFillAvailable) {
    // Exactly 100 shares available
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    // FOK for 100 shares
    auto fok_order = create_buy_order(2, 10000, 100, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    // Should fill completely
    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, FOKPartialFillRejected) {
    // Only 50 shares available
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));

    // FOK wants 100 shares (all or nothing)
    auto fok_order = create_buy_order(2, 10000, 100, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    // Should be rejected (can't fill completely)
    EXPECT_TRUE(trades.empty());

    // Original sell order should remain untouched
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 50);  // Still 50 available
}

TEST_F(OrderbookTest, FOKMultipleLevels) {
    // 150 shares available across levels
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));
    orderbook_->AddOrder(create_sell_order(3, 10100, 50));

    // FOK for 150 shares
    auto fok_order = create_buy_order(4, 10100, 150, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    // Should fill all 150 shares
    ASSERT_EQ(trades.size(), 3);

    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 50);

    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, FOKMultipleLevelsInsufficient) {
    // Only 100 shares available
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10100, 50));

    // FOK wants 150 shares
    auto fok_order = create_buy_order(3, 10100, 150, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    // Should reject (insufficient liquidity)
    EXPECT_TRUE(trades.empty());

    // Original orders remain
    EXPECT_EQ(orderbook_->Size(), 2);
}

// ============================================================================
// ORDER TYPES - GOOD FOR DAY (GFD)
// ============================================================================

TEST_F(OrderbookTest, GFDOrderAdded) {
    auto gfd_order = create_buy_order(1, 10000, 100, OrderType::GoodForDay);
    auto trades = orderbook_->AddOrder(gfd_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);
}

// Note: Day reset functionality would need time manipulation to test properly
// Skipping comprehensive GFD tests as they require system time control

// ============================================================================
// EXCHANGE RULES - VALIDATION
// ============================================================================

TEST_F(OrderbookTest, TickSizeValidation) {
    ExchangeRules rules;
    rules.tickSize = 5;  // Only prices divisible by 5 allowed

    orderbook_->SetExchangeRules(rules);

    // Valid price (10000 divisible by 5)
    auto valid_order = create_buy_order(1, 10000, 100);
    auto trades = orderbook_->AddOrder(valid_order);
    EXPECT_EQ(orderbook_->Size(), 1);

    // Invalid price (10001 not divisible by 5)
    auto invalid_order = create_buy_order(2, 10001, 100);
    trades = orderbook_->AddOrder(invalid_order);

    // Should be rejected
    EXPECT_EQ(orderbook_->Size(), 1);  // Still only first order
}

TEST_F(OrderbookTest, LotSizeValidation) {
    ExchangeRules rules;
    rules.lotSize = 10;  // Only quantities divisible by 10

    orderbook_->SetExchangeRules(rules);

    // Valid quantity
    auto valid_order = create_buy_order(1, 10000, 100);
    orderbook_->AddOrder(valid_order);
    EXPECT_EQ(orderbook_->Size(), 1);

    // Invalid quantity (not divisible by 10)
    auto invalid_order = create_buy_order(2, 10000, 105);
    orderbook_->AddOrder(invalid_order);

    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MinQuantityValidation) {
    ExchangeRules rules;
    rules.minQuantity = 50;

    orderbook_->SetExchangeRules(rules);

    // Below minimum
    auto below_min = create_buy_order(1, 10000, 25);
    orderbook_->AddOrder(below_min);
    EXPECT_EQ(orderbook_->Size(), 0);

    // At minimum
    auto at_min = create_buy_order(2, 10000, 50);
    orderbook_->AddOrder(at_min);
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MaxQuantityValidation) {
    ExchangeRules rules;
    rules.maxQuantity = 1000;

    orderbook_->SetExchangeRules(rules);

    // Above maximum
    auto above_max = create_buy_order(1, 10000, 1500);
    orderbook_->AddOrder(above_max);
    EXPECT_EQ(orderbook_->Size(), 0);

    // At maximum
    auto at_max = create_buy_order(2, 10000, 1000);
    orderbook_->AddOrder(at_max);
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MinNotionalValidation) {
    ExchangeRules rules;
    rules.minNotional = 10000;  // $100 minimum (100 cents * 100 shares)

    orderbook_->SetExchangeRules(rules);

    // Below minimum notional: 50 * 100 = 5000
    auto below_min = create_buy_order(1, 100, 50);
    orderbook_->AddOrder(below_min);
    EXPECT_EQ(orderbook_->Size(), 0);

    // At minimum notional: 100 * 100 = 10000
    auto at_min = create_buy_order(2, 100, 100);
    orderbook_->AddOrder(at_min);
    EXPECT_EQ(orderbook_->Size(), 1);
}

// ============================================================================
// ORDERBOOK STATE & QUERIES
// ============================================================================

TEST_F(OrderbookTest, GetOrderInfosMultipleLevels) {
    // Add multiple price levels
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900, 150));
    orderbook_->AddOrder(create_buy_order(3, 10000, 50));   // Same price as order 1

    orderbook_->AddOrder(create_sell_order(4, 10100, 200));
    orderbook_->AddOrder(create_sell_order(5, 10200, 100));

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    const auto& asks = infos.GetAsks();

    // Bids sorted best first (highest price)
    ASSERT_EQ(bids.size(), 2);
    EXPECT_EQ(bids[0].price_, 10000);
    EXPECT_EQ(bids[0].quantity_, 150);  // 100 + 50 aggregated
    EXPECT_EQ(bids[1].price_, 9900);
    EXPECT_EQ(bids[1].quantity_, 150);

    // Asks sorted best first (lowest price)
    ASSERT_EQ(asks.size(), 2);
    EXPECT_EQ(asks[0].price_, 10100);
    EXPECT_EQ(asks[0].quantity_, 200);
    EXPECT_EQ(asks[1].price_, 10200);
    EXPECT_EQ(asks[1].quantity_, 100);
}

TEST_F(OrderbookTest, SizeTracksOrderCount) {
    EXPECT_EQ(orderbook_->Size(), 0);

    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->AddOrder(create_buy_order(2, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 2);

    orderbook_->AddOrder(create_sell_order(3, 10100, 100));
    EXPECT_EQ(orderbook_->Size(), 3);

    // Cancel one
    orderbook_->CancelOrder(2);
    EXPECT_EQ(orderbook_->Size(), 2);
}

TEST_F(OrderbookTest, EmptyPriceLevelsRemoved) {
    // Add two orders at same price
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetAsks().size(), 1);  // One price level
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 150);  // Total quantity

    // Buy all shares at this level
    auto buy_order = create_buy_order(3, 10000, 150);
    orderbook_->AddOrder(buy_order);

    // Price level should be removed
    infos = orderbook_->GetOrderInfos();
    EXPECT_TRUE(infos.GetAsks().empty());
}

// ============================================================================
// EDGE CASES & STRESS TESTS
// ============================================================================

TEST_F(OrderbookTest, LargeNumberOfOrders) {
    // Add 1000 orders
    for (int i = 0; i < 1000; ++i) {
        orderbook_->AddOrder(create_buy_order(i, 10000 - i, 100));
    }

    EXPECT_EQ(orderbook_->Size(), 1000);

    // Add matching sell order
    auto sell = create_sell_order(10000, 9000, 100);
    auto trades = orderbook_->AddOrder(sell);

    // Should match with best bid (10000)
    ASSERT_EQ(trades.size(), 1);
    std::cout << "DEBUG - trades[0] bid order ID: " << trades[0].GetBidTrade().orderId_ << std::endl;
    std::cout << "DEBUG - trades[0] bid price: " << trades[0].GetBidTrade().price_ << std::endl;
    std::cout << "DEBUG - trades[0] ask order ID: " << trades[0].GetAskTrade().orderId_ << std::endl;
    std::cout << "DEBUG - trades[0] ask price: " << trades[0].GetAskTrade().price_ << std::endl;
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, VerySmallQuantities) {
    // 1 share orders
    orderbook_->AddOrder(create_sell_order(1, 10000, 1));

    auto buy = create_buy_order(2, 10000, 1);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 1);
}

TEST_F(OrderbookTest, VeryLargeQuantities) {
    // Maximum quantity
    uint32_t max_qty = 1000000;

    orderbook_->AddOrder(create_sell_order(1, 10000, max_qty));

    auto buy = create_buy_order(2, 10000, max_qty);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, max_qty);
}

TEST_F(OrderbookTest, ExtremePrice) {
    // Very low price (1 cent)
    orderbook_->AddOrder(create_sell_order(1, 1, 100));

    auto buy = create_buy_order(2, 1, 100);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 1);
}

TEST_F(OrderbookTest, SelfMatchPrevention) {
    // This test verifies that the orderbook doesn't crash with same order ID
    // (though in production, order IDs should be unique per participant)

    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    // Try to add sell with same ID (should be prevented by validation)
    ExchangeRules rules;
    orderbook_->SetExchangeRules(rules);

    auto sell = create_sell_order(1, 10000, 100);
    auto trades = orderbook_->AddOrder(sell);

    // Should not crash - duplicate ID rejected
    EXPECT_TRUE(trades.empty());
}

TEST_F(OrderbookTest, ZeroPrice) {
    // Zero price should be invalid for non-market orders
    auto order = create_buy_order(1, 0, 100);
    auto trades = orderbook_->AddOrder(order);

    // Should be rejected
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, SequentialOrderIds) {
    // Ensure order IDs don't interfere with matching
    for (OrderId id = 1; id <= 10; ++id) {
        orderbook_->AddOrder(create_buy_order(id, 10000 - id * 10, 100));
    }

    EXPECT_EQ(orderbook_->Size(), 10);

    // All orders should be independent
    orderbook_->CancelOrder(5);
    EXPECT_EQ(orderbook_->Size(), 9);
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

TEST_F(OrderbookTest, MatchDoesNotModifyUnmatchedOrders) {
    // Add sell at $101
    orderbook_->AddOrder(create_sell_order(1, 10100, 100));

    // Buy at $100 (doesn't match)
    auto buy = create_buy_order(2, 10000, 100);
    orderbook_->AddOrder(buy);

    // Both should remain unchanged
    EXPECT_EQ(orderbook_->Size(), 2);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetBids()[0].quantity_, 100);
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 100);
}

TEST_F(OrderbookTest, CancelDoesNotAffectOtherOrders) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900, 100));
    orderbook_->AddOrder(create_buy_order(3, 10100, 100));

    orderbook_->CancelOrder(2);

    // Orders 1 and 3 unaffected
    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();

    EXPECT_EQ(bids.size(), 2);

    // Find order 1 and 3
    bool found_order1 = false;
    bool found_order3 = false;

    for (const auto& level : bids) {
        if (level.price_ == 10000) found_order1 = true;
        if (level.price_ == 10100) found_order3 = true;
    }

    EXPECT_TRUE(found_order1);
    EXPECT_TRUE(found_order3);
}

TEST_F(OrderbookTest, MultipleMatchesInSingleCall) {
    // Add multiple sell orders
    orderbook_->AddOrder(create_sell_order(1, 10000, 30));
    orderbook_->AddOrder(create_sell_order(2, 10000, 40));
    orderbook_->AddOrder(create_sell_order(3, 10000, 50));

    // Large buy that matches all
    auto buy = create_buy_order(4, 10000, 120);
    auto trades = orderbook_->AddOrder(buy);

    // Verify all three sell orders matched
    ASSERT_EQ(trades.size(), 3);

    // Total matched = 30 + 40 + 50 = 120
    Quantity total_matched = 0;
    for (const auto& trade : trades) {
        total_matched += trade.GetBidTrade().quantity_;
    }
    EXPECT_EQ(total_matched, 120);
}