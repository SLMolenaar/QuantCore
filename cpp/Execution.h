#pragma once

#include "orderbook/OrderBook.h"
#include "orderbook/Order.h"
#include "orderbook/Types.h"
#include "orderbook/Trade.h"
#include "orderbook/OrderType.h"
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <stdexcept>
#include <optional>

namespace quantcore {

namespace defaults {
    constexpr double MAKER_FEE = 0.0001;
    constexpr double TAKER_FEE = 0.0002;
    constexpr int64_t LATENCY_NS = 1000000;
    constexpr double SLIPPAGE_PCT = 0.0001;
}

struct ExecutionConfig {
    double maker_fee = defaults::MAKER_FEE;
    double taker_fee = defaults::TAKER_FEE;
    int64_t latency_ns = defaults::LATENCY_NS;
    double slippage_pct = defaults::SLIPPAGE_PCT;
};

// wraps orderbook with backtesting features
class ExecutionEngine {
public:
    ExecutionEngine(const std::string& symbol = "DEFAULT",
                   ExecutionConfig config = ExecutionConfig())
        : symbol_(symbol)
        , config_(config)
        , orderbook_()
        , realized_pnl_(0.0)
        , unrealized_pnl_(0.0)
        , total_fees_(0.0)
    {
        if (symbol.empty()) {
            throw std::invalid_argument("Symbol cannot be empty");
        }
    }

    // execute order through orderbook
    // returns trades that occurred
    Trades execute_order(OrderPointer order) {
        if (!order) {
            throw std::invalid_argument("Cannot execute null order");
        }

        OrderId order_id = order->GetOrderId();
        orders_owned_[order_id] = order->GetSide();

        // Determine if we're a taker (crossing the spread) or maker (resting in book)
        bool is_taker = can_match_immediately(order);

        auto trades = orderbook_.AddOrder(order);

        // Update positions - all trades from this order use the same fee type
        for (const auto& trade : trades) {
            update_position(trade, is_taker);
        }

        return trades;
    }

    void cancel_order(OrderId order_id) {
        orders_owned_.erase(order_id);
        orderbook_.CancelOrder(order_id);
    }

    Trades modify_order(const OrderModify& modify) {
        auto trades = orderbook_.MatchOrder(modify);

        for (const auto& trade : trades) {
            // Modified orders that trade immediately are takers
            update_position(trade, true);
        }

        return trades;
    }

    double get_position() const {
        auto it = positions_.find(symbol_);
        return it != positions_.end() ? it->second : 0.0;
    }

    // avg entry
    double get_average_price() const {
        auto it = avg_prices_.find(symbol_);
        return it != avg_prices_.end() ? it->second : 0.0;
    }

    double get_realized_pnl() const {
        return realized_pnl_;
    }

    double get_unrealized_pnl() const {
        double position = get_position();
        if (position == 0.0) return 0.0;

        auto mid_price_opt = get_mid_price();
        if (!mid_price_opt.has_value()) return 0.0;

        double current_price = mid_price_opt.value();
        double avg_price = get_average_price();

        if (avg_price == 0.0) return 0.0;

        return position * (current_price - avg_price);
    }

    double get_total_pnl() const {
        return realized_pnl_ + get_unrealized_pnl();
    }

    double get_total_fees() const {
        return total_fees_;
    }

    // maybe for debugging later?
    Orderbook& get_orderbook() {
        return orderbook_;
    }

    const Orderbook& get_orderbook() const {
        return orderbook_;
    }

    Price get_best_bid() const {
        auto infos = orderbook_.GetOrderInfos();
        const auto& bids = infos.GetBids();
        return bids.empty() ? 0 : bids[0].price_;
    }

    Price get_best_ask() const {
        auto infos = orderbook_.GetOrderInfos();
        const auto& asks = infos.GetAsks();
        return asks.empty() ? 0 : asks[0].price_;
    }

    // avg of best bid & ask
    std::optional<double> get_mid_price() const {
        Price bid = get_best_bid();
        Price ask = get_best_ask();

        if (bid == 0 && ask == 0) return std::nullopt;
        if (bid == 0) return ask / 100.0;
        if (ask == 0) return bid / 100.0;

        return (bid + ask) / 200.0; // cents to dollars and average
    }

    // Reset all state (for new backtest run)
    void reset() {
        positions_.clear();
        avg_prices_.clear();
        orders_owned_.clear();
        realized_pnl_ = 0.0;
        unrealized_pnl_ = 0.0;
        total_fees_ = 0.0;
        orderbook_ = Orderbook();
    }

    // execution statistics
    struct Stats {
        size_t total_trades = 0;
        double total_volume = 0.0;
        double total_fees = 0.0;
        size_t orders_in_book = 0;
    };

    Stats get_stats() const {
        Stats stats;
        stats.total_fees = total_fees_;
        stats.orders_in_book = orderbook_.Size();
        // Could add more stats tracking if needed
        return stats;
    }

private:
    std::string symbol_;
    ExecutionConfig config_;
    Orderbook orderbook_;

    // Position tracking, symbol & quantity
    std::map<std::string, double> positions_;
    std::map<std::string, double> avg_prices_;
    std::unordered_map<OrderId, Side> orders_owned_;

    double realized_pnl_;
    double unrealized_pnl_;
    double total_fees_;

    // Check if order will match immediately (making us a taker)
    bool can_match_immediately(OrderPointer order) const {
        if (order->GetSide() == Side::Buy) {
            Price best_ask = get_best_ask();
            if (best_ask == 0) return false;
            return order->GetPrice() >= best_ask;
        } else {
            Price best_bid = get_best_bid();
            if (best_bid == 0) return false;
            return order->GetPrice() <= best_bid;
        }
    }

    void update_position(const Trade& trade, bool is_taker) {
        const auto& bid_trade = trade.GetBidTrade();
        const auto& ask_trade = trade.GetAskTrade();

        // convert cents to dollar
        double quantity = static_cast<double>(bid_trade.quantity_);
        double price = static_cast<double>(bid_trade.price_) / 100.0;

        auto bid_it = orders_owned_.find(bid_trade.orderId_);
        auto ask_it = orders_owned_.find(ask_trade.orderId_);

        bool we_are_buyer = (bid_it != orders_owned_.end() && bid_it->second == Side::Buy);
        bool we_are_seller = (ask_it != orders_owned_.end() && ask_it->second == Side::Sell);

        if (!we_are_buyer && !we_are_seller) {
            return;
        }

        if (we_are_buyer && we_are_seller) {
            return;
        }

        double current_position = get_position();
        double current_avg_price = get_average_price();

        if (we_are_buyer) {
            double fee = calculate_fee(price, quantity, is_taker);
            total_fees_ += fee;
            realized_pnl_ -= fee;

            if (current_position < 0) {
                // Covering short position
                double cover_qty = std::min(quantity, -current_position);
                realized_pnl_ += cover_qty * (current_avg_price - price);

                if (quantity > -current_position) {
                    // Covered short and going long
                    double new_long_qty = quantity + current_position;
                    positions_[symbol_] = new_long_qty;
                    avg_prices_[symbol_] = price; // New long position starts at current price
                } else {
                    // Still short or flat
                    positions_[symbol_] = current_position + quantity;
                    if (positions_[symbol_] == 0.0) {
                        avg_prices_[symbol_] = 0.0;
                    }
                }
            } else {
                // Adding to long or initiating long
                if (current_position == 0.0) {
                    avg_prices_[symbol_] = price;
                } else {
                    double total_cost = current_position * current_avg_price + quantity * price;
                    avg_prices_[symbol_] = total_cost / (current_position + quantity);
                }
                positions_[symbol_] = current_position + quantity;
            }

            orders_owned_.erase(bid_trade.orderId_);
        }

        if (we_are_seller) {
            double fee = calculate_fee(price, quantity, is_taker);
            total_fees_ += fee;
            realized_pnl_ -= fee;

            if (current_position > 0) {
                // Reducing long position
                double sell_qty = std::min(quantity, current_position);
                realized_pnl_ += sell_qty * (price - current_avg_price);

                if (quantity > current_position) {
                    // Sold long and going short
                    double new_short_qty = -(quantity - current_position);
                    positions_[symbol_] = new_short_qty;
                    avg_prices_[symbol_] = price;
                } else {
                    // Still long or flat
                    positions_[symbol_] = current_position - quantity;
                    if (positions_[symbol_] == 0.0) {
                        avg_prices_[symbol_] = 0.0;
                    }
                }
            } else {
                // Adding to short or initiating short
                if (current_position == 0.0) {
                    avg_prices_[symbol_] = price;
                } else {
                    double total_cost = -current_position * current_avg_price + quantity * price;
                    avg_prices_[symbol_] = total_cost / std::abs(current_position - quantity);
                }
                positions_[symbol_] = current_position - quantity;
            }

            orders_owned_.erase(ask_trade.orderId_);
        }
    }

    double calculate_fee(double price, double quantity, bool is_taker) const {
        double notional = price * quantity;
        double fee_rate = is_taker ? config_.taker_fee : config_.maker_fee;
        return notional * fee_rate;
    }

    double calculate_slippage(double price, double quantity) const {
        return price * quantity * config_.slippage_pct;
    }
};

}