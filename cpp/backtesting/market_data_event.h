#pragma once

#include "event.h"
#include <string>

namespace quantcore {

 // represents a bar or tick update
class MarketDataEvent : public Event {
public:
    // OHLCV bar data
    MarketDataEvent(
        const std::string& symbol,
        int64_t timestamp_ns,
        double open,
        double high,
        double low,
        double close,
        double volume
    )
        : Event(EventType::MARKET_DATA, timestamp_ns)
        , symbol_(symbol)
        , open_(open)
        , high_(high)
        , low_(low)
        , close_(close)
        , volume_(volume)
    {
    }

    // tick data, just price
    MarketDataEvent(
        const std::string& symbol,
        int64_t timestamp_ns,
        double price,
        double volume = 0.0
    )
        : Event(EventType::MARKET_DATA, timestamp_ns)
        , symbol_(symbol)
        , open_(price)
        , high_(price)
        , low_(price)
        , close_(price)
        , volume_(volume)
    {
    }
    
    std::string get_symbol() const { return symbol_; }
    double get_open() const { return open_; }
    double get_high() const { return high_; }
    double get_low() const { return low_; }
    double get_close() const { return close_; }
    double get_volume() const { return volume_; }
    
    // uses close
    double get_price() const { return close_; }
    
    std::string to_string() const override {
        return "MarketDataEvent(symbol=" + symbol_ + 
               ", timestamp=" + std::to_string(timestamp_ns_) +
               ", close=" + std::to_string(close_) +
               ", volume=" + std::to_string(volume_) + ")";
    }
    
private:
    std::string symbol_;
    double open_;
    double high_;
    double low_;
    double close_;
    double volume_;
};

}