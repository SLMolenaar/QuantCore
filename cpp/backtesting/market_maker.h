#pragma once

#include "../Execution.h"
#include "market_data_event.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <deque>

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
    MarketMaker(
        double base_spread_pct = 0.0001,
        int num_levels = 5,
        Quantity base_depth = 10000
    )
        : base_spread_pct_(base_spread_pct)
        , current_spread_pct_(base_spread_pct)
        , num_levels_(num_levels)
        , base_depth_(base_depth)
        , next_order_id_(10000000000000ULL)
        , rng_(std::random_device{}())
    {
    }

    void update_quotes(ExecutionEngine& engine, const MarketDataEvent& event) {
        cancel_orders(engine);

        double mid_price = event.get_close();
        double volume = event.get_volume();
        int64_t timestamp = event.get_timestamp();

        price_history_.push_back(mid_price);
        if (price_history_.size() > 20) {
            price_history_.pop_front();
        }

        double volatility = calc_volatility();
        double time_factor = get_time_factor(timestamp);

        update_spread(volatility, volume, time_factor);

        place_orders(engine, mid_price, volume, Side::Buy);
        place_orders(engine, mid_price, volume, Side::Sell);
    }

    double get_spread() const { return current_spread_pct_; }

    void set_num_levels(int levels) {
        num_levels_ = std::max(1, std::min(10, levels));
    }

    void set_base_spread(double spread) {
        base_spread_pct_ = std::max(0.00001, spread);
    }

    void set_base_depth(Quantity depth) {
        base_depth_ = std::max(static_cast<Quantity>(100), depth);
    }

private:
    double base_spread_pct_;
    double current_spread_pct_;
    int num_levels_;
    Quantity base_depth_;
    uint64_t next_order_id_;

    std::deque<double> price_history_;
    std::vector<OrderId> active_orders_;
    mutable std::mt19937 rng_;

    void cancel_orders(ExecutionEngine& engine) {
        auto& orderbook = engine.get_orderbook();
        for (OrderId id : active_orders_) {
            orderbook.CancelOrder(id);
        }
        active_orders_.clear();
    }

    void place_orders(ExecutionEngine& engine, double mid_price,
                     double volume, Side side) {
        auto& orderbook = engine.get_orderbook();

        for (int level = 0; level < num_levels_; ++level) {
            double offset = (level + 0.5) * current_spread_pct_;

            double price = (side == Side::Buy)
                ? mid_price * (1.0 - offset)
                : mid_price * (1.0 + offset);

            Price price_cents = static_cast<Price>(price * 100.0);
            Quantity quantity = calc_quantity(level, volume);

            auto order = std::make_shared<Order>(
                OrderType::GoodTillCancel,
                next_order_id_++,
                side,
                price_cents,
                quantity
            );

            orderbook.AddOrder(order);
            active_orders_.push_back(order->GetOrderId());
        }
    }

    Quantity calc_quantity(int level, double volume) const {
        double decay = std::exp(-0.3 * level);
        double vol_factor = 1.0 + std::log1p(volume / 1000000.0) * 0.1;

        Quantity base = static_cast<Quantity>(base_depth_ * decay * vol_factor);

        std::uniform_real_distribution<double> dist(0.9, 1.1);
        return static_cast<Quantity>(base * dist(rng_));
    }

    double calc_volatility() const {
        if (price_history_.size() < 2) return 0.02;

        std::vector<double> returns;
        for (size_t i = 1; i < price_history_.size(); ++i) {
            returns.push_back((price_history_[i] - price_history_[i-1]) / price_history_[i-1]);
        }

        double mean = 0.0;
        for (double r : returns) mean += r;
        mean /= returns.size();

        double var = 0.0;
        for (double r : returns) {
            double diff = r - mean;
            var += diff * diff;
        }
        var /= returns.size();

        return std::max(0.001, std::sqrt(var) * 10.0);
    }

    void update_spread(double vol, double volume, double time_factor) {
        double spread = base_spread_pct_;

        spread *= (1.0 + (vol / 0.02 - 1.0) * 0.5);
        spread *= (1.0 + std::max(0.0, (1000000.0 - volume) / 1000000.0) * 0.3);
        spread *= time_factor;

        current_spread_pct_ = std::max(base_spread_pct_ * 0.5,
                                       std::min(base_spread_pct_ * 3.0, spread));
    }

    double get_time_factor(int64_t timestamp_ns) const {
        int hour = ((timestamp_ns / 1000000000LL) % 86400 / 3600) % 24;

        if (hour < 9 || hour >= 16) return 2.0;
        if (hour == 9 || hour == 15) return 1.5;
        return 1.0;
    }
};

}