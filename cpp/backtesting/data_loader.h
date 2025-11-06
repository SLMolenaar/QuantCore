#pragma once

#include "bar_data.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace quantcore {
    /**
     * Load OHLCV data from CSV files
     */
    class CSVDataLoader {
    public:
        static BarSeries load(const std::string& filepath,
                              const std::string& symbol = "",
                              bool has_header = true) {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filepath);
            }

            BarSeries bars;
            std::string line;
            size_t line_number = 0;
            size_t skipped_lines = 0;

            // Skip header if present
            if (has_header) {
                std::getline(file, line);
                line_number++;
            }

            while (std::getline(file, line)) {
                line_number++;
                if (line.empty()) continue;

                try {
                    auto bar = parse_line(line, symbol);
                    bars.push_back(bar);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Skipping line " << line_number
                              << " in " << filepath << ": " << e.what() << "\n";
                    skipped_lines++;
                    continue;
                }
            }

            if (skipped_lines > 0) {
                std::cerr << "Warning: Skipped " << skipped_lines
                          << " invalid lines out of " << line_number << " total lines\n";
            }

            if (bars.empty()) {
                throw std::runtime_error("No valid data loaded from file: " + filepath);
            }

            // Sort by timestamp (just in case)
            std::sort(bars.begin(), bars.end(),
                      [](const BarData& a, const BarData& b) {
                          return a.timestamp_ns < b.timestamp_ns;
                      });

            return bars;
        }

    private:
        static BarData parse_line(const std::string& line, const std::string& default_symbol) {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;

            // Split by comma
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            BarData bar;

            // Parse based on number of columns
            if (tokens.size() == 6) {
                // timestamp,open,high,low,close,volume
                bar.symbol = default_symbol;
                bar.timestamp_ns = parse_timestamp(tokens[0]);
                bar.open = std::stod(tokens[1]);
                bar.high = std::stod(tokens[2]);
                bar.low = std::stod(tokens[3]);
                bar.close = std::stod(tokens[4]);
                bar.volume = std::stod(tokens[5]);
            } else if (tokens.size() == 7) {
                // symbol,timestamp,open,high,low,close,volume
                bar.symbol = tokens[0];
                bar.timestamp_ns = parse_timestamp(tokens[1]);
                bar.open = std::stod(tokens[2]);
                bar.high = std::stod(tokens[3]);
                bar.low = std::stod(tokens[4]);
                bar.close = std::stod(tokens[5]);
                bar.volume = std::stod(tokens[6]);
            } else {
                throw std::runtime_error("Invalid CSV format");
            }

            return bar;
        }

        static int64_t parse_timestamp(const std::string& ts_str) {

            try {
                int64_t ts = std::stoll(ts_str);

                // If timestamp looks like seconds, convert to nanoseconds
                if (ts < 4000000000LL) {
                    return ts * 1000000000LL;
                }
                // If timestamp looks like milliseconds, convert to nanoseconds
                else if (ts < 4000000000000LL) {
                    return ts * 1000000LL;
                }
                // Otherwise assume already in nanoseconds
                else {
                    return ts;
                }
            } catch (...) {
                throw std::runtime_error("Could not parse timestamp: " + ts_str);
            }
        }
    };
}