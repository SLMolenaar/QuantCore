#pragma once

#include "tick_data.h"
#include "bar_data.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace quantcore {

// load tick data from csv and optionally aggregate to bars
class TickDataLoader {
public:
    // Expected formats (auto-detected by column count):
    //   3 cols: timestamp, price, quantity
    //   4 cols: timestamp, price, quantity, side   (B/b/buy/BUY or S/s/sell/SELL)
    static TickSeries load(const std::string& filepath,
                           const std::string& symbol   = "",
                           bool  has_header            = true,
                           double max_skip_pct         = 0.20)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + filepath);

        TickSeries ticks;
        std::string line;
        size_t line_nr    = 0;
        size_t data_lines = 0;
        size_t bad_lines  = 0;

        if (has_header) {
            std::getline(file, line);
            line_nr++;
        }

        while (std::getline(file, line)) {
            line_nr++;
            if (line.empty()) continue;
            data_lines++;

            try {
                ticks.push_back(parse_line(line, symbol));
            } catch (const std::exception& e) {
                std::cerr << "WARNING: skipping line " << line_nr
                          << " in " << filepath << ": " << e.what() << "\n";
                bad_lines++;
            }
        }

        double skip_pct = (data_lines > 0)
            ? static_cast<double>(bad_lines) / data_lines
            : 0.0;

        if (skip_pct > max_skip_pct) {
            throw std::runtime_error(
                "Data quality error: " + std::to_string(bad_lines) +
                " out of " + std::to_string(data_lines) + " lines skipped (" +
                std::to_string(skip_pct * 100.0) + "%), exceeds threshold of " +
                std::to_string(max_skip_pct * 100.0) + "%"
            );
        }

        if (ticks.empty())
            throw std::runtime_error("No valid tick data loaded from: " + filepath);

        std::sort(ticks.begin(), ticks.end(),
            [](const TickData& a, const TickData& b) {
                return a.timestamp_ns < b.timestamp_ns;
            });

        return ticks;
    }

    // Aggregate ticks into OHLCV bars of a fixed duration.
    // bar_duration_ns: bar width in nanoseconds (e.g. 60_000_000_000 for 1-minute)
    static BarSeries aggregate_to_bars(const TickSeries& ticks,
                                       int64_t bar_duration_ns)
    {
        if (ticks.empty()) return {};
        if (bar_duration_ns <= 0)
            throw std::invalid_argument("Bar duration must be positive");

        BarSeries bars;
        const std::string& symbol = ticks.front().symbol;

        // all ticks must share the same symbol
        for (const auto& t : ticks) {
            if (t.symbol != symbol)
                throw std::invalid_argument(
                    "aggregate_to_bars: mixed symbols in TickSeries ('" +
                    symbol + "' vs '" + t.symbol + "'). Split by symbol first."
                );
        }

        int64_t bar_start = (ticks.front().timestamp_ns / bar_duration_ns) * bar_duration_ns;
        double  open = 0, high = 0, low = 0, close = 0, volume = 0;
        bool    has_ticks = false;

        auto flush_bar = [&]() {
            if (!has_ticks) return;
            bars.emplace_back(symbol,
                bar_start + bar_duration_ns - 1, // bar close timestamp
                open, high, low, close, volume);
        };

        for (const auto& tick : ticks) {
            int64_t tick_bar = (tick.timestamp_ns / bar_duration_ns) * bar_duration_ns;

            if (tick_bar != bar_start) {
                flush_bar();
                bar_start = tick_bar;
                has_ticks = false;
                volume    = 0.0;
            }

            if (!has_ticks) {
                open      = tick.price;
                high      = tick.price;
                low       = tick.price;
                has_ticks = true;
            } else {
                high = std::max(high, tick.price);
                low  = std::min(low,  tick.price);
            }
            close   = tick.price;
            volume += tick.quantity;
        }

        flush_bar();
        return bars;
    }

private:
    static TickData parse_line(const std::string& line, const std::string& default_symbol) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ','))
            tokens.push_back(token);

        if (tokens.size() < 3 || tokens.size() > 5)
            throw std::runtime_error("Expected 3-5 columns, got " +
                                     std::to_string(tokens.size()));

        TickData tick;

        if (tokens.size() == 3) {
            // timestamp, price, quantity
            tick.symbol        = default_symbol;
            tick.timestamp_ns  = parse_timestamp(tokens[0]);
            tick.price         = std::stod(tokens[1]);
            tick.quantity      = std::stod(tokens[2]);
            tick.aggressor_side = Side::Buy;
        } else if (tokens.size() == 4) {
            // timestamp, price, quantity, side
            tick.symbol        = default_symbol;
            tick.timestamp_ns  = parse_timestamp(tokens[0]);
            tick.price         = std::stod(tokens[1]);
            tick.quantity      = std::stod(tokens[2]);
            tick.aggressor_side = parse_side(tokens[3]);
        } else {
            // symbol, timestamp, price, quantity, side
            tick.symbol        = tokens[0];
            tick.timestamp_ns  = parse_timestamp(tokens[1]);
            tick.price         = std::stod(tokens[2]);
            tick.quantity      = std::stod(tokens[3]);
            tick.aggressor_side = tokens.size() == 5
                ? parse_side(tokens[4])
                : Side::Buy;
        }

        return tick;
    }

    static int64_t parse_timestamp(const std::string& ts_str) {
        try {
            int64_t ts = std::stoll(ts_str);
            if      (ts < 4'000'000'000LL)              return ts * 1'000'000'000LL;
            else if (ts < 4'000'000'000'000LL)          return ts * 1'000'000LL;
            else if (ts < 4'000'000'000'000'000LL)      return ts * 1'000LL;
            else                                         return ts;
        } catch (...) {
            throw std::runtime_error("Could not parse timestamp: " + ts_str);
        }
    }

    static Side parse_side(const std::string& raw) {
        std::string s = raw;
        // trim whitespace
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s == "b" || s == "buy")  return Side::Buy;
        if (s == "s" || s == "sell") return Side::Sell;
        throw std::runtime_error("Unknown side: " + raw);
    }
};

} // namespace quantcore