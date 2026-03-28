/**
 * Tests for Orderbook
 */

#include <gtest/gtest.h>
#include "orderbook/Orderbook.h"
#include "orderbook/Order.h"
#include "orderbook/OrderModify.h"
#include "orderbook/Ordertype.h"
#include "orderbook/Trade.h"
#include "orderbook/Types.h"
#include "orderbook/ExchangeRules.h"
#include <memory>
#include <algorithm>

class OrderbookTest : public ::testing::Test {
protected:
    void SetUp() override {
        orderbook_.reset();
        orderbook_ = std::make_unique<Orderbook>();
        ASSERT_EQ(orderbook_->Size(), 0) << "Orderbook not empty in SetUp!";
    }

    OrderPointer create_buy_order(OrderId id, Price price, Quantity qty,
                                  OrderType type = OrderType::GoodTillCancel) {
        return std::make_shared<Order>(type, id, Side::Buy, price, qty);
    }

    OrderPointer create_sell_order(OrderId id, Price price, Quantity qty,
                                   OrderType type = OrderType::GoodTillCancel) {
        return std::make_shared<Order>(type, id, Side::Sell, price, qty);
    }

    std::unique_ptr<Orderbook> orderbook_;
};

TEST_F(OrderbookTest, InitialStateEmpty) {
    EXPECT_EQ(orderbook_->Size(), 0);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_TRUE(infos.GetBids().empty());
    EXPECT_TRUE(infos.GetAsks().empty());
}

TEST_F(OrderbookTest, AddSingleBuyOrder) {
    auto order = create_buy_order(1, 10000, 100);
    auto trades = orderbook_->AddOrder(order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();

    ASSERT_EQ(bids.size(), 1);
    EXPECT_EQ(bids[0].price_, 10000);
    EXPECT_EQ(bids[0].quantity_, 100);
}

TEST_F(OrderbookTest, AddSingleSellOrder) {
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
    orderbook_->AddOrder(create_buy_order(1, 9900, 100));
    orderbook_->AddOrder(create_sell_order(2, 10100, 100));

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    const auto& asks = infos.GetAsks();

    ASSERT_FALSE(bids.empty());
    ASSERT_FALSE(asks.empty());
    EXPECT_LT(bids[0].price_, asks[0].price_);
    EXPECT_EQ(bids[0].price_, 9900);
    EXPECT_EQ(asks[0].price_, 10100);
}

TEST_F(OrderbookTest, SimpleMatchExactQuantity) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto buy_order = create_buy_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().orderId_, 2);
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, SimpleMatchBuyerIsTaker) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto buy_order = create_buy_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(trades[0].GetAskTrade().price_, 10000);
}

TEST_F(OrderbookTest, SimpleMatchSellerIsTaker) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    auto sell_order = create_sell_order(2, 10000, 100);
    auto trades = orderbook_->AddOrder(sell_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(trades[0].GetAskTrade().price_, 10000);
}

TEST_F(OrderbookTest, PartialFillBuyOrderTooLarge) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto buy_order = create_buy_order(2, 10000, 150);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    ASSERT_EQ(infos.GetBids().size(), 1);
    EXPECT_EQ(infos.GetBids()[0].price_, 10000);
    EXPECT_EQ(infos.GetBids()[0].quantity_, 50);
}

TEST_F(OrderbookTest, PartialFillSellOrderTooLarge) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    auto sell_order = create_sell_order(2, 10000, 150);
    auto trades = orderbook_->AddOrder(sell_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 1);

    auto infos = orderbook_->GetOrderInfos();
    ASSERT_EQ(infos.GetAsks().size(), 1);
    EXPECT_EQ(infos.GetAsks()[0].price_, 10000);
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 50);
}

TEST_F(OrderbookTest, MultiplePartialFills) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10000, 30));
    orderbook_->AddOrder(create_sell_order(3, 10000, 40));

    auto buy_order = create_buy_order(4, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 3);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 30);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 20);
    EXPECT_EQ(orderbook_->Size(), 1);

    EXPECT_EQ(orderbook_->GetOrderInfos().GetAsks()[0].quantity_, 20);
}

TEST_F(OrderbookTest, PriceTimePrioritySamePrice) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));
    orderbook_->AddOrder(create_sell_order(2, 10000, 100));
    orderbook_->AddOrder(create_sell_order(3, 10000, 100));

    auto buy_order = create_buy_order(4, 10000, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 1);
    EXPECT_EQ(orderbook_->Size(), 2);
}

TEST_F(OrderbookTest, BestPriceMatchedFirst) {
    orderbook_->AddOrder(create_sell_order(1, 10200, 100));
    orderbook_->AddOrder(create_sell_order(2, 10000, 100));
    orderbook_->AddOrder(create_sell_order(3, 10100, 100));

    auto buy_order = create_buy_order(4, 10200, 100);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 2);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, PriceThenTimePriority) {
    orderbook_->AddOrder(create_sell_order(1, 10100, 100));
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));
    orderbook_->AddOrder(create_sell_order(3, 10000, 50));

    auto buy_order = create_buy_order(4, 10100, 150);
    auto trades = orderbook_->AddOrder(buy_order);

    ASSERT_EQ(trades.size(), 3);

    EXPECT_EQ(trades[0].GetAskTrade().orderId_, 2);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);

    EXPECT_EQ(trades[1].GetAskTrade().orderId_, 3);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().price_, 10000);

    EXPECT_EQ(trades[2].GetAskTrade().orderId_, 1);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[2].GetBidTrade().price_, 10100);
}

TEST_F(OrderbookTest, CancelExistingOrder) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->CancelOrder(1);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelNonExistentOrder) {
    EXPECT_NO_THROW(orderbook_->CancelOrder(999));
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelPartiallyFilledOrder) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_buy_order(2, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->CancelOrder(2);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, CancelMaintainsBookIntegrity) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900,  100));
    orderbook_->AddOrder(create_buy_order(3, 10100, 100));

    orderbook_->CancelOrder(1);
    EXPECT_EQ(orderbook_->Size(), 2);

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    ASSERT_EQ(bids.size(), 2);
    EXPECT_EQ(bids[0].price_, 10100);
    EXPECT_EQ(bids[1].price_, 9900);
}

TEST_F(OrderbookTest, ModifyOrderPrice) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    OrderModify modify(1, Side::Buy, 10100, 100);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);
    EXPECT_EQ(orderbook_->GetOrderInfos().GetBids()[0].price_, 10100);
}

TEST_F(OrderbookTest, ModifyOrderQuantity) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    OrderModify modify(1, Side::Buy, 10000, 150);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->GetOrderInfos().GetBids()[0].quantity_, 150);
}

TEST_F(OrderbookTest, ModifyTriggersMatch) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900, 100));

    OrderModify modify(2, Side::Buy, 10000, 100);
    auto trades = orderbook_->MatchOrder(modify);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, ModifyNonExistentOrder) {
    OrderModify modify(999, Side::Buy, 10000, 100);
    auto trades = orderbook_->MatchOrder(modify);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, MarketOrderBuyWithLiquidity) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto market_order = create_buy_order(2, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, MarketOrderSellWithLiquidity) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    auto market_order = create_sell_order(2, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
}

TEST_F(OrderbookTest, MarketOrderNoLiquidity) {
    auto market_order = create_buy_order(1, 0, 100, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, MarketOrderSweepsMultipleLevels) {
    ASSERT_EQ(orderbook_->Size(), 0);

    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10100, 50));
    orderbook_->AddOrder(create_sell_order(3, 10200, 50));

    ASSERT_EQ(orderbook_->Size(), 3);

    auto market_order = create_buy_order(4, 0, 125, OrderType::Market);
    auto trades = orderbook_->AddOrder(market_order);

    ASSERT_EQ(trades.size(), 3);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 10000);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().price_, 10100);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 25);
    EXPECT_EQ(trades[2].GetBidTrade().price_, 10200);
}

TEST_F(OrderbookTest, IOCFullFill) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto ioc_order = create_buy_order(2, 10000, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, IOCPartialFill) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));

    auto ioc_order = create_buy_order(2, 10000, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, IOCNoFillCancelled) {
    auto ioc_order = create_buy_order(1, 9900, 100, OrderType::ImmediateOrCancel);
    auto trades = orderbook_->AddOrder(ioc_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, FOKFullFillAvailable) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));

    auto fok_order = create_buy_order(2, 10000, 100, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 100);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, FOKPartialFillRejected) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));

    auto fok_order = create_buy_order(2, 10000, 100, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);
    EXPECT_EQ(orderbook_->GetOrderInfos().GetAsks()[0].quantity_, 50);
}

TEST_F(OrderbookTest, FOKMultipleLevels) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));
    orderbook_->AddOrder(create_sell_order(3, 10100, 50));

    auto fok_order = create_buy_order(4, 10100, 150, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    ASSERT_EQ(trades.size(), 3);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[1].GetBidTrade().quantity_, 50);
    EXPECT_EQ(trades[2].GetBidTrade().quantity_, 50);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, FOKMultipleLevelsInsufficient) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 50));
    orderbook_->AddOrder(create_sell_order(2, 10100, 50));

    auto fok_order = create_buy_order(3, 10100, 150, OrderType::FillOrKill);
    auto trades = orderbook_->AddOrder(fok_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 2);
}

TEST_F(OrderbookTest, GFDOrderAdded) {
    auto gfd_order = create_buy_order(1, 10000, 100, OrderType::GoodForDay);
    auto trades = orderbook_->AddOrder(gfd_order);

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, TickSizeValidation) {
    ExchangeRules rules;
    rules.tickSize = 5;
    orderbook_->SetExchangeRules(rules);

    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->AddOrder(create_buy_order(2, 10001, 100));
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, LotSizeValidation) {
    ExchangeRules rules;
    rules.lotSize = 10;
    orderbook_->SetExchangeRules(rules);

    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    EXPECT_EQ(orderbook_->Size(), 1);

    orderbook_->AddOrder(create_buy_order(2, 10000, 105));
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MinQuantityValidation) {
    ExchangeRules rules;
    rules.minQuantity = 50;
    orderbook_->SetExchangeRules(rules);

    orderbook_->AddOrder(create_buy_order(1, 10000, 25));
    EXPECT_EQ(orderbook_->Size(), 0);

    orderbook_->AddOrder(create_buy_order(2, 10000, 50));
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MaxQuantityValidation) {
    ExchangeRules rules;
    rules.maxQuantity = 1000;
    orderbook_->SetExchangeRules(rules);

    orderbook_->AddOrder(create_buy_order(1, 10000, 1500));
    EXPECT_EQ(orderbook_->Size(), 0);

    orderbook_->AddOrder(create_buy_order(2, 10000, 1000));
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, MinNotionalValidation) {
    ExchangeRules rules;
    rules.minNotional = 10000;
    orderbook_->SetExchangeRules(rules);

    orderbook_->AddOrder(create_buy_order(1, 100, 50));
    EXPECT_EQ(orderbook_->Size(), 0);

    orderbook_->AddOrder(create_buy_order(2, 100, 100));
    EXPECT_EQ(orderbook_->Size(), 1);
}

TEST_F(OrderbookTest, GetOrderInfosMultipleLevels) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900,  150));
    orderbook_->AddOrder(create_buy_order(3, 10000, 50));
    orderbook_->AddOrder(create_sell_order(4, 10100, 200));
    orderbook_->AddOrder(create_sell_order(5, 10200, 100));

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    const auto& asks = infos.GetAsks();

    ASSERT_EQ(bids.size(), 2);
    EXPECT_EQ(bids[0].price_, 10000);
    EXPECT_EQ(bids[0].quantity_, 150);  // 100 + 50 aggregated
    EXPECT_EQ(bids[1].price_, 9900);
    EXPECT_EQ(bids[1].quantity_, 150);

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

    orderbook_->CancelOrder(2);
    EXPECT_EQ(orderbook_->Size(), 2);
}

TEST_F(OrderbookTest, EmptyPriceLevelsRemoved) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 100));
    orderbook_->AddOrder(create_sell_order(2, 10000, 50));

    EXPECT_EQ(orderbook_->GetOrderInfos().GetAsks()[0].quantity_, 150);

    orderbook_->AddOrder(create_buy_order(3, 10000, 150));

    EXPECT_TRUE(orderbook_->GetOrderInfos().GetAsks().empty());
}

TEST_F(OrderbookTest, LargeNumberOfOrders) {
    for (int i = 0; i < 1000; ++i) {
        orderbook_->AddOrder(create_buy_order(100000 + i, 10000 - i, 100));
    }
    EXPECT_EQ(orderbook_->Size(), 1000);

    auto sell = create_sell_order(200000, 9000, 100);
    auto trades = orderbook_->AddOrder(sell);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().orderId_, 100000);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 9000);
}

TEST_F(OrderbookTest, VerySmallQuantities) {
    orderbook_->AddOrder(create_sell_order(1, 10000, 1));

    auto buy = create_buy_order(2, 10000, 1);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, 1);
}

TEST_F(OrderbookTest, VeryLargeQuantities) {
    uint32_t max_qty = 1000000;
    orderbook_->AddOrder(create_sell_order(1, 10000, max_qty));

    auto buy = create_buy_order(2, 10000, max_qty);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().quantity_, max_qty);
}

TEST_F(OrderbookTest, ExtremePrice) {
    orderbook_->AddOrder(create_sell_order(1, 1, 100));

    auto buy = create_buy_order(2, 1, 100);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].GetBidTrade().price_, 1);
}

TEST_F(OrderbookTest, SelfMatchPrevention) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));

    ExchangeRules rules;
    orderbook_->SetExchangeRules(rules);

    auto sell = create_sell_order(1, 10000, 100);
    auto trades = orderbook_->AddOrder(sell);

    EXPECT_TRUE(trades.empty());
}

TEST_F(OrderbookTest, ZeroPrice) {
    auto order = create_buy_order(1, 0, 100);
    orderbook_->AddOrder(order);
    EXPECT_EQ(orderbook_->Size(), 0);
}

TEST_F(OrderbookTest, SequentialOrderIds) {
    for (OrderId id = 1; id <= 10; ++id) {
        orderbook_->AddOrder(create_buy_order(id, 10000 - id * 10, 100));
    }
    EXPECT_EQ(orderbook_->Size(), 10);

    orderbook_->CancelOrder(5);
    EXPECT_EQ(orderbook_->Size(), 9);
}

TEST_F(OrderbookTest, MatchDoesNotModifyUnmatchedOrders) {
    orderbook_->AddOrder(create_sell_order(1, 10100, 100));
    orderbook_->AddOrder(create_buy_order(2, 10000, 100));

    EXPECT_EQ(orderbook_->Size(), 2);

    auto infos = orderbook_->GetOrderInfos();
    EXPECT_EQ(infos.GetBids()[0].quantity_, 100);
    EXPECT_EQ(infos.GetAsks()[0].quantity_, 100);
}

TEST_F(OrderbookTest, CancelDoesNotAffectOtherOrders) {
    orderbook_->AddOrder(create_buy_order(1, 10000, 100));
    orderbook_->AddOrder(create_buy_order(2, 9900,  100));
    orderbook_->AddOrder(create_buy_order(3, 10100, 100));

    orderbook_->CancelOrder(2);

    auto infos = orderbook_->GetOrderInfos();
    const auto& bids = infos.GetBids();
    EXPECT_EQ(bids.size(), 2);

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
    orderbook_->AddOrder(create_sell_order(1, 10000, 30));
    orderbook_->AddOrder(create_sell_order(2, 10000, 40));
    orderbook_->AddOrder(create_sell_order(3, 10000, 50));

    auto buy = create_buy_order(4, 10000, 120);
    auto trades = orderbook_->AddOrder(buy);

    ASSERT_EQ(trades.size(), 3);

    Quantity total_matched = 0;
    for (const auto& trade : trades) {
        total_matched += trade.GetBidTrade().quantity_;
    }
    EXPECT_EQ(total_matched, 120);
}
