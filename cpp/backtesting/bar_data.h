#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace quantcore {

    // single OHLCV bar
    struct BarData {
        std::string symbol;
        int64_t timestamp_ns;// close time
        double open;
        double high;
        double low;
        double close;
        double volume;

        BarData() = default;

        BarData(const std::string& sym, int64_t ts, double o, double h, double l, double c, double v)
            : symbol(sym), timestamp_ns(ts), open(o), high(h), low(l), close(c), volume(v)
        {
            validate();
        }

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

    private:
        void validate() const {
            if (high < low) {
                throw std::invalid_argument("Bar validation failed: high < low");
            }
            if (open > high || open < low) {
                throw std::invalid_argument("Bar validation failed: open outside high/low range");
            }
            if (close > high || close < low) {
                throw std::invalid_argument("Bar validation failed: close outside high/low range");
            }
            if (volume < 0) {
                throw std::invalid_argument("Bar validation failed: negative volume");
            }
            if (open <= 0 || high <= 0 || low <= 0 || close <= 0) {
                throw std::invalid_argument("Bar validation failed: non-positive prices");
            }
        }
    };

     //Collection of bars for a symbol
    using BarSeries = std::vector<BarData>;

}