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
        // Simulate latency
        // std::this_thread::sleep_for(std::chrono::nanoseconds(config_.latency_ns));
        // TODO: look at ts^^

        auto trades = orderbook_.AddOrder(order);
        
        // Update positions and PnL for each trade
        for (const auto& trade : trades) {
            update_position(trade);
        }
        
        return trades;
    }

    void cancel_order(OrderId order_id) {
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
        return unrealized_pnl_;
    }
    
    //realized + unrealized
    double get_total_pnl() const {
        return realized_pnl_ + unrealized_pnl_;
    }

    double get_total_fees() const {
        return total_fees_;
    }
    
    // maybe for debugging later?
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
        if (bid == 0 || ask == 0) return 0.0;
        return (bid + ask) / 200.0; // cents to dollars and average
    }

    // Reset all state (for new backtest run)
    void reset() {
        positions_.clear();
        avg_prices_.clear();
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
    
    // Average entry prices, symbol & avg price
    std::map<std::string, double> avg_prices_;
    
    double realized_pnl_;
    double unrealized_pnl_;
    double total_fees_ = 0.0;
    

    //Update position and PnL
    void update_position(const Trade& trade) {
        const auto& bid_trade = trade.GetBidTrade();
        const auto& ask_trade = trade.GetAskTrade();
        
        // cnvert cents to dollar
        double quantity = static_cast<double>(bid_trade.quantity_);
        double price = static_cast<double>(bid_trade.price_) / 100.0;
        
        // fees
        double bid_fee = calculate_fee(price, quantity, false); // Taker
        double ask_fee = calculate_fee(price, quantity, true);  // Maker
        double total_trade_fees = bid_fee + ask_fee;

        total_fees_ += total_trade_fees;
        
        // Update realized PnL
        realized_pnl_ -= total_trade_fees;
        
        // Update position tracking
        // Simplified, just track net position
        // In real system, would track each fill separately for FIFO/LIFO
        double current_position = get_position();
        double new_position = current_position; // Will be updated based on trade
        
        // This is simplified, in reality you'd need to know which side YOU are on
        // For now, just tracking that a trade happened
        positions_[symbol_] = new_position;
    }

    double calculate_fee(double price, double quantity, bool is_maker) {
        double notional = price * quantity;
        double fee_rate = is_maker ? config_.maker_fee : config_.taker_fee;
        return notional * fee_rate;
    }

    //future enhancement
    double calculate_slippage(double price, double quantity) {
        // Simple percentage-based slippage model
        // In reality would be based on orderbook depth
        return price * quantity * config_.slippage_pct;
    }
};

}