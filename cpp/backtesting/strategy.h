#pragma once

#include "market_data_event.h"
#include "signal_event.h"
#include "fill_event.h"
#include "portfolio_context.h"
#include <memory>
#include <string>
#include <map>

namespace quantcore {

/**
 * Base class for trading strategies
 * 
 * Users implement their strategy by inheriting from this class
 * and overriding the on_data() method.
 * 
 * Single-asset strategies can use get_position(symbol) as before.
 * Multi-asset strategies use get_portfolio() for portfolio-level logic.
 */
class Strategy {
public:
    Strategy(const std::string& name = "Strategy")
        : name_(name)
        , portfolio_(nullptr)
    {}
    
    virtual ~Strategy() = default;
    
    /**
     * Called on each market data update
     * Override this to implement your strategy logic
     */
    virtual void on_data(const MarketDataEvent& event) = 0;

    /**
     * Called when an order is filled
     * Override to react to fills (optional)
     */
    virtual void on_fill(const FillEvent& event) {
        (void)event;
    }

    std::string get_name() const { return name_; }
    
    std::vector<std::shared_ptr<SignalEvent>> get_signals() {
        std::vector<std::shared_ptr<SignalEvent>> temp;
        temp.swap(signals_);
        return temp;
    }

    bool has_signals() const {
        return !signals_.empty();
    }

    virtual void reset() {
        signals_.clear();
        positions_.clear();
    }
    
protected:
    void generate_signal(const std::string& symbol, 
                        SignalType signal_type,
                        double strength = 1.0,
                        int64_t timestamp_ns = 0) {
        auto signal = std::make_shared<SignalEvent>(
            symbol, timestamp_ns, signal_type, strength, name_);
        signals_.push_back(signal);
    }

    void set_position(const std::string& symbol, double quantity) {
        positions_[symbol] = quantity;
    }

    double get_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? it->second : 0.0;
    }

    bool has_position(const std::string& symbol) const {
        return get_position(symbol) != 0.0;
    }

    /**
     * Get portfolio context for multi-asset strategies
     * Returns nullptr if not set by engine yet
     */
    PortfolioContext* get_portfolio() const {
        return portfolio_;
    }

    /**
     * Check if portfolio context is available
     */
    bool has_portfolio_context() const {
        return portfolio_ != nullptr;
    }
    
private:
    std::string name_;
    std::vector<std::shared_ptr<SignalEvent>> signals_;
    std::map<std::string, double> positions_;
    PortfolioContext* portfolio_;
    
    friend class BacktestEngine;
};

}