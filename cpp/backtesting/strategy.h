#pragma once

#include "market_data_event.h"
#include "signal_event.h"
#include "fill_event.h"
#include "order_event.h"
#include "portfolio_context.h"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

namespace quantcore {

// Represents a pending stop or stop-limit order emitted by the strategy.
// The engine drains these alongside signals after each on_data() call.
struct StopOrder {
    std::string symbol;
    int64_t     timestamp_ns;
    Side        side;
    OrderType   order_type;  // Stop or StopLimit
    double      quantity;
    double      stop_price;  // trigger level
    double      limit_price; // limit price for StopLimit; 0.0 for Stop
};

// Base for trading strategies; override on_data() to implement your logic.
class Strategy {
public:
    Strategy(const std::string& name = "Strategy") : name_(name), portfolio_(nullptr) {}

    virtual ~Strategy() = default;

    // Called on each bar or tick. Implement your strategy logic here.
    virtual void on_data(const MarketDataEvent& event) = 0;

    // Called after each fill. Override to react to executions.
    virtual void on_fill(const FillEvent& event) {
        (void)event;
    }

    // Called when a signal is rejected by risk limits. Override to react.
    virtual void on_rejected(const std::string& symbol, const std::string& reason) {
        (void)symbol;
        (void)reason;
    }

    std::string get_name() const { return name_; }

    // Drain and return all pending signals since the last call.
    std::vector<std::shared_ptr<SignalEvent>> get_signals() {
        std::vector<std::shared_ptr<SignalEvent>> temp;
        temp.swap(signals_);
        return temp;
    }

    // Drain and return all pending stop orders since the last call.
    std::vector<StopOrder> get_stop_orders() {
        std::vector<StopOrder> temp;
        temp.swap(stop_orders_);
        return temp;
    }

    bool has_signals()     const { return !signals_.empty(); }
    bool has_stop_orders() const { return !stop_orders_.empty(); }

    virtual void reset() {
        signals_.clear();
        stop_orders_.clear();
        positions_.clear();
    }

    // -----------------------------------------------------------------------
    // Signal generation
    // -----------------------------------------------------------------------

    void generate_signal(const std::string& symbol,
                         SignalType sig_type,
                         double strength,
                         int64_t timestamp_ns) {
        auto sig = std::make_shared<SignalEvent>(
            symbol, timestamp_ns, sig_type, strength, name_);
        signals_.push_back(sig);
    }

    // -----------------------------------------------------------------------
    // Stop order generation
    // -----------------------------------------------------------------------

    // Place a stop order that triggers a Market fill when price touches
    // stop_price.
    //
    // side:       Sell to protect a long, Buy to protect a short.
    // stop_price: Trigger level. Must be positive.
    // quantity:   Shares to trade. Pass 0.0 to close the full position at
    //             trigger time.
    void generate_stop(const std::string& symbol,
                       Side side,
                       double stop_price,
                       double quantity,
                       int64_t timestamp_ns) {
        if (stop_price <= 0.0)
            throw std::invalid_argument("stop_price must be positive");
        stop_orders_.push_back({
            symbol, timestamp_ns, side,
            OrderType::Stop,
            quantity, stop_price, 0.0
        });
    }

    // Place a stop-limit order that converts to a GoodTillCancel limit at
    // limit_price when stop_price is touched.
    //
    // For sell stop-limit: stop_price <= current price,
    //                      limit_price <= stop_price
    // For buy  stop-limit: stop_price >= current price,
    //                      limit_price >= stop_price
    //
    // quantity: Pass 0.0 to close the full position at trigger time.
    void generate_stop_limit(const std::string& symbol,
                             Side side,
                             double stop_price,
                             double limit_price,
                             double quantity,
                             int64_t timestamp_ns) {
        if (stop_price  <= 0.0)
            throw std::invalid_argument("stop_price must be positive");
        if (limit_price <= 0.0)
            throw std::invalid_argument("limit_price must be positive");
        stop_orders_.push_back({
            symbol, timestamp_ns, side,
            OrderType::StopLimit,
            quantity, stop_price, limit_price
        });
    }

    // -----------------------------------------------------------------------
    // Position queries
    // -----------------------------------------------------------------------

    void set_position(const std::string& symbol, double qty) {
        positions_[symbol] = qty;
    }

    double get_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? it->second : 0.0;
    }

    bool has_position(const std::string& symbol) const {
        return get_position(symbol) != 0.0;
    }

    PortfolioContext* get_portfolio() const { return portfolio_; }
    bool has_portfolio_context()      const { return portfolio_ != nullptr; }

private:
    std::string name_;
    std::vector<std::shared_ptr<SignalEvent>> signals_;
    std::vector<StopOrder>                   stop_orders_;
    std::map<std::string, double>            positions_;
    PortfolioContext*                        portfolio_;

    friend class BacktestEngine;
};

} // namespace quantcore
