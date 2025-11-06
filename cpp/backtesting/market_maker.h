#pragma once

#include "../Execution.h"
#include "market_data_event.h"
#include <memory>

namespace quantcore {

/**
 * Provides synthetic liquidity for backtesting
 *
 * In real markets, there are market makers providing bid/ask quotes.
 * This simulates that by placing limit orders at bid/ask around the
 * current market price with a configurable spread.
 *
 * Without this, strategies would have no liquidity to trade against.
 */
class MarketMaker {
public:
    MarketMaker(double spread_pct = 0.001, Quantity depth = 10000)
        : spread_pct_(spread_pct)
        , depth_(depth)
        , next_order_id_(1000000000ULL)
    {
    }

    void update_quotes(ExecutionEngine& engine, const MarketDataEvent& event) {
        cancel_old_orders(engine);

        double mid_price = event.get_close();
        double half_spread = mid_price * spread_pct_ / 2.0;

        Price bid_price = static_cast<Price>((mid_price - half_spread) * 100.0);
        Price ask_price = static_cast<Price>((mid_price + half_spread) * 100.0);

        auto bid_order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            next_order_id_++,
            Side::Buy,
            bid_price,
            depth_
        );

        auto ask_order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            next_order_id_++,
            Side::Sell,
            ask_price,
            depth_
        );

        auto& orderbook = engine.get_orderbook();
        orderbook.AddOrder(bid_order);
        orderbook.AddOrder(ask_order);

        active_bid_ = bid_order->GetOrderId();
        active_ask_ = ask_order->GetOrderId();
    }

    void set_spread(double spread_pct) {
        spread_pct_ = spread_pct;
    }

    void set_depth(Quantity depth) {
        depth_ = depth;
    }

private:
    double spread_pct_;
    Quantity depth_;
    uint64_t next_order_id_;
    OrderId active_bid_ = 0;
    OrderId active_ask_ = 0;

    void cancel_old_orders(ExecutionEngine& engine) {
        if (active_bid_ != 0) {
            engine.get_orderbook().CancelOrder(active_bid_);
        }
        if (active_ask_ != 0) {
            engine.get_orderbook().CancelOrder(active_ask_);
        }
    }
};

}