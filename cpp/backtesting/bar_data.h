#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace quantcore {

    /**
     * Single OHLCV bar
     *
     * Represents price action over a time period (1min, 5min, 1day, etc.)
     */
    struct BarData {
        std::string symbol;
        int64_t timestamp_ns;   // Bar close time
        double open;
        double high;
        double low;
        double close;
        double volume;

        BarData() = default;

        BarData(const std::string& sym, int64_t ts, double o, double h, double l, double c, double v)
            : symbol(sym), timestamp_ns(ts), open(o), high(h), low(l), close(c), volume(v)
        {
        }

        // Get typical price (HLC/3)
        double typical_price() const {
            return (high + low + close) / 3.0;
        }

        double range() const {
            return high - low;
        }

        bool is_bullish() const {
            return close > open;
        }

        bool is_bearish() const {
            return close < open;
        }
    };

     //Collection of bars for a symbol
    using BarSeries = std::vector<BarData>;

}