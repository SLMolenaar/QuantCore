#pragma once

#include "market_data_event.h"
#include "signal_event.h"
#include "fill_event.h"
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
 * Example:
 *   class MeanReversion : public Strategy {
 *       void on_data(const MarketDataEvent& event) override {
 *           // strategy logic here
 *           if (should_buy) {
 *               generate_signal(event.get_symbol(), SignalType::BUY);
 *           }
 *       }
 *   };
 */
class Strategy {
public:
    Strategy(const std::string& name = "Strategy")
        : name_(name)
    {
    }
    
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
        // Default: do nothing
        (void)event;
    }

    // strat name
    std::string get_name() const { return name_; }
    
    //Get signals generated since last check
    std::vector<std::shared_ptr<SignalEvent>> get_signals() {
        auto signals = signals_;
        signals_.clear();
        return signals;
    }

    // Check if strat has pending signals
    bool has_signals() const {
        return !signals_.empty();
    }

    // Reset strat state (for new backtest run)
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

    // Updated by backtesting engine(will implement later)
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
    
private:
    std::string name_;
    std::vector<std::shared_ptr<SignalEvent>> signals_;
    std::map<std::string, double> positions_;  // symbol, quantity
    
    friend class BacktestEngine;  // Allow engine to update positions
};

}