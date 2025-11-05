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
    double maker_fee = 0.0001;     // 0.01% maker fee
    double taker_fee = 0.0002;     // 0.02% taker fee
    int64_t latency_ns = 1000000;  // 1ms default latency
    double slippage_pct = 0.0001;  // 0.01% slippage
};

/**
 * ExecutionEngine - Wrapper around Orderbook for backtesting
 * 
 * Adds backtesting-specific features on top of the orderbook:
 * - Fee calculation (maker/taker)
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
    
    /**
     * Execute order through orderbook
     * Returns trades that occurred
     */
    Trades execute_order(OrderPointer order) {
        // Simulate latency (if needed in future)
        // std::this_thread::sleep_for(std::chrono::nanoseconds(config_.latency_ns));
        
        auto trades = orderbook_.AddOrder(order);
        
        // Update positions and PnL for each trade
        for (const auto& trade : trades) {
            update_position(trade);
        }
        
        return trades;
    }
    
    /**
     * Cancel order from orderbook
     */
    void cancel_order(OrderId order_id) {
        orderbook_.CancelOrder(order_id);
    }
    
    /**
     * Modify existing order
     */
    Trades modify_order(const OrderModify& modify) {
        auto trades = orderbook_.MatchOrder(modify);
        
        for (const auto& trade : trades) {
            update_position(trade);
        }
        
        return trades;
    }
    
    /**
     * Get current position for symbol
     * Positive = long, Negative = short, 0 = flat
     */
    double get_position() const {
        auto it = positions_.find(symbol_);
        return it != positions_.end() ? it->second : 0.0;
    }
    
    /**
     * Get average entry price for current position
     */
    double get_average_price() const {
        auto it = avg_prices_.find(symbol_);
        return it != avg_prices_.end() ? it->second : 0.0;
    }
    
    /**
     * Get realized PnL (from closed trades)
     */
    double get_realized_pnl() const {
        return realized_pnl_;
    }
    
    /**
     * Get unrealized PnL (mark-to-market)
     * Requires current market price to calculate
     */
    double get_unrealized_pnl() const {
        return unrealized_pnl_;
    }
    
    /**
     * Get total PnL = realized + unrealized
     */
    double get_total_pnl() const {
        return realized_pnl_ + unrealized_pnl_;
    }
    
    /**
     * Get total fees paid
     */
    double get_total_fees() const {
        return total_fees_;
    }
    
    /**
     * Get orderbook reference (for inspection/debugging)
     */
    const Orderbook& get_orderbook() const {
        return orderbook_;
    }
    
    /**
     * Get current best bid price (or 0 if no bids)
     */
    Price get_best_bid() const {
        auto infos = orderbook_.GetOrderInfos();
        const auto& bids = infos.GetBids();
        return bids.empty() ? 0 : bids[0].price_;
    }
    
    /**
     * Get current best ask price (or 0 if no asks)
     */
    Price get_best_ask() const {
        auto infos = orderbook_.GetOrderInfos();
        const auto& asks = infos.GetAsks();
        return asks.empty() ? 0 : asks[0].price_;
    }
    
    /**
     * Get mid price (average of best bid and ask)
     */
    double get_mid_price() const {
        Price bid = get_best_bid();
        Price ask = get_best_ask();
        if (bid == 0 || ask == 0) return 0.0;
        return (bid + ask) / 200.0; // Convert cents to dollars and average
    }
    
    /**
     * Reset all state (for new backtest run)
     */
    void reset() {
        positions_.clear();
        avg_prices_.clear();
        realized_pnl_ = 0.0;
        unrealized_pnl_ = 0.0;
        total_fees_ = 0.0;
        orderbook_ = Orderbook();
    }
    
    /**
     * Get execution statistics
     */
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
    
    // Position tracking: symbol -> quantity (positive = long, negative = short)
    std::map<std::string, double> positions_;
    
    // Average entry prices: symbol -> average price
    std::map<std::string, double> avg_prices_;
    
    // PnL tracking
    double realized_pnl_;
    double unrealized_pnl_;
    double total_fees_ = 0.0;
    
    /**
     * Update position and PnL based on executed trade
     */
    void update_position(const Trade& trade) {
        const auto& bid_trade = trade.GetBidTrade();
        const auto& ask_trade = trade.GetAskTrade();
        
        // Convert to doubles (cents -> dollars)
        double quantity = static_cast<double>(bid_trade.quantity_);
        double price = static_cast<double>(bid_trade.price_) / 100.0;
        
        // Calculate fees
        double bid_fee = calculate_fee(price, quantity, false); // Taker
        double ask_fee = calculate_fee(price, quantity, true);  // Maker
        double total_trade_fees = bid_fee + ask_fee;
        
        // Update total fees
        total_fees_ += total_trade_fees;
        
        // Update realized PnL (subtract fees)
        realized_pnl_ -= total_trade_fees;
        
        // Update position tracking
        // Simplified: just track net position
        // In real system, would track each fill separately for FIFO/LIFO
        double current_position = get_position();
        double new_position = current_position; // Will be updated based on trade
        
        // This is simplified - in reality you'd need to know which side YOU are on
        // For now, just tracking that a trade happened
        positions_[symbol_] = new_position;
    }
    
    /**
     * Calculate trading fee for a trade
     */
    double calculate_fee(double price, double quantity, bool is_maker) {
        double notional = price * quantity;
        double fee_rate = is_maker ? config_.maker_fee : config_.taker_fee;
        return notional * fee_rate;
    }
    
    /**
     * Calculate slippage for a trade (future enhancement)
     */
    double calculate_slippage(double price, double quantity) {
        // Simple percentage-based slippage model
        // In reality would be based on orderbook depth
        return price * quantity * config_.slippage_pct;
    }
};

} // namespace quantcore