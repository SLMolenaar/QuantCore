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
#include <unordered_set>

namespace quantcore {

struct ExecutionConfig {
    double maker_fee = 0.0001;
    double taker_fee = 0.0002;
    int64_t latency_ns = 1000000;
    double slippage_pct = 0.0001;
};

/**
 * Wrapper around Orderbook for backtesting
 * 
 * Adds backtesting-specific features on top of the orderbook:
 * - Fee calculation
 * - Slippage simulation
 * - Position tracking
 * - PnL calculation (realized + unrealized)
 * - Latency simulation
 */
class ExecutionEngine {
public:
    ExecutionEngine(const std::string& symbol = "DEFAULT",
                   ExecutionConfig config = ExecutionConfig())
        : symbol_(symbol)
        , config_(config)
        , orderbook_()
        , realized_pnl_(0.0)
        , unrealized_pnl_(0.0)
    {
    }

    // execute order through orderbook
    // returns trades tht occurred
    Trades execute_order(OrderPointer order) {
        orders_owned_.insert(order->GetOrderId());

        auto trades = orderbook_.AddOrder(order);

        for (const auto& trade : trades) {
            update_position(trade);
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
            update_position(trade);
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

        double current_price = get_mid_price();
        double avg_price = get_average_price();

        if (current_price == 0.0 || avg_price == 0.0) return 0.0;

        return position * (current_price - avg_price);
    }
    
    //realized + unrealized
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
    
    // abg of best bid & ask
    double get_mid_price() const {
        Price bid = get_best_bid();
        Price ask = get_best_ask();
        if (bid == 0 && ask == 0) return 0.0;
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
    std::unordered_set<OrderId> orders_owned_;

    double realized_pnl_;
    double unrealized_pnl_;
    double total_fees_ = 0.0;

    void update_position(const Trade& trade) {
        const auto& bid_trade = trade.GetBidTrade();
        const auto& ask_trade = trade.GetAskTrade();

        // cnvert cents to dollar
        double quantity = static_cast<double>(bid_trade.quantity_);
        double price = static_cast<double>(bid_trade.price_) / 100.0;

        bool we_are_buyer = orders_owned_.contains(bid_trade.orderId_);
        bool we_are_seller = orders_owned_.contains(ask_trade.orderId_);

        if (!we_are_buyer && !we_are_seller) {
            return;
        }

        if (we_are_buyer && we_are_seller) {
            return;
        }

        double current_position = get_position();
        double current_avg_price = get_average_price();

        if (we_are_buyer) {
            bool is_taker = true;
            double fee = calculate_fee(price, quantity, !is_taker);
            total_fees_ += fee;
            realized_pnl_ -= fee;

            if (current_position < 0) {
                // Covering short position
                double cover_qty = std::min(quantity, -current_position);
                realized_pnl_ += cover_qty * (current_avg_price - price);

                if (quantity > -current_position) {
                    double new_long_qty = quantity + current_position;
                    positions_[symbol_] = new_long_qty;
                    avg_prices_[symbol_] = price; // New position starts at current price
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
            bool is_taker = true; // In backtest, we're always takers against market maker
            double fee = calculate_fee(price, quantity, !is_taker);
            total_fees_ += fee;
            realized_pnl_ -= fee;

            if (current_position > 0) {
                // Reducing long position
                double sell_qty = std::min(quantity, current_position);
                realized_pnl_ += sell_qty * (price - current_avg_price);

                if (quantity > current_position) {
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

    double calculate_fee(double price, double quantity, bool is_maker) {
        double notional = price * quantity;
        double fee_rate = is_maker ? config_.maker_fee : config_.taker_fee;
        return notional * fee_rate;
    }

    double calculate_slippage(double price, double quantity) {
        return price * quantity * config_.slippage_pct;
    }
};

}