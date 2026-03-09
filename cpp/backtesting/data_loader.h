#pragma once

#include "bar_data.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace quantcore {

    // load ohlcv from csv
    class CSVDataLoader {
    public:

        /* public interface for loading data */
        static BarSeries load(const std::string& filepath,
                              const std::string& symbol = "",
                              bool has_header = true,
                              double max_skip_pct = 0.20) { // fails if 20% of lines were skipped

            std::ifstream file(filepath);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filepath);
            }

            BarSeries bars;
            std::string line;
            size_t line_nr = 0;      // Total lines read (including header)
            size_t data_lines = 0;   // Count only data lines, not header
            size_t bad_lines = 0;

            // Skip header
            if (has_header) {
                std::getline(file, line);
                line_nr++;
                // Note: do NOT increment data_lines for header
            }

            // main reading loop
            while (std::getline(file, line)) {
                line_nr++;
                if (line.empty()) continue;

                data_lines++;  // Count this as a data line

                try {
                    auto bar = parse_line(line, symbol);
                    bars.push_back(bar);
                } catch (const std::exception& e) {
                    std::cerr << "WARNING: skipping line " << line_nr
                              << " in " << filepath << ": " << e.what() << "\n";
                    bad_lines++;
                    continue;
                }
            }

            // Check if too many lines were skipped, if more than 20% of lines are bad the file is probably corrupted
            double skip_pct = (data_lines > 0)
                ? static_cast<double>(bad_lines) / data_lines
                : 0.0;

            if (skip_pct > max_skip_pct) {
                throw std::runtime_error(
                    "Data quality error: " + std::to_string(bad_lines) +
                    " out of " + std::to_string(data_lines) +
                    " data lines skipped (" + std::to_string(skip_pct * 100) +
                    "%), exceeds threshold of " + std::to_string(max_skip_pct * 100) + "%"
                );
            }

            if (bad_lines > 0) {
                std::cerr << "INFO: Skipped " << bad_lines
                          << " lines out of " << data_lines << " data lines ("
                          << (skip_pct * 100) << "%)\n";
            }

            if (bars.empty()) {
                throw std::runtime_error("No valid data loaded from file: " + filepath);
            }

            // sort by timestamp (just in case the data isn't sorted already)
            std::sort(bars.begin(), bars.end(),
                      [](const BarData& a, const BarData& b) {
                          return a.timestamp_ns < b.timestamp_ns;
                      });

            return bars;
        }

    private: // helper functions
        static BarData parse_line(const std::string& line, const std::string& default_symbol) {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;

            // Split with comma
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            BarData bar;

            // Parse based on nr of cols
            if (tokens.size() == 6) {
                bar.symbol = default_symbol;
                bar.timestamp_ns = parse_timestamp(tokens[0]);
                bar.open = std::stod(tokens[1]);
                bar.high = std::stod(tokens[2]);
                bar.low = std::stod(tokens[3]);
                bar.close = std::stod(tokens[4]);
                bar.volume = std::stod(tokens[5]);
            } else if (tokens.size() == 7) {
                bar.symbol = tokens[0];
                bar.timestamp_ns = parse_timestamp(tokens[1]);
                bar.open = std::stod(tokens[2]);
                bar.high = std::stod(tokens[3]);
                bar.low = std::stod(tokens[4]);
                bar.close = std::stod(tokens[5]);
                bar.volume = std::stod(tokens[6]);
            } else {
                throw std::runtime_error("Invalid CSV format: expected 6 or 7 columns, got " +
                                       std::to_string(tokens.size()));
            }

            return bar;
        }

        static int64_t parse_timestamp(const std::string& ts_str) {

            try {
                int64_t ts = std::stoll(ts_str);


                if (ts < 4'000'000'000LL) {
                    // Seconds -> nanoseconds
                    return ts * 1'000'000'000LL;
                }
                else if (ts < 4'000'000'000'000LL) {
                    // Milliseconds -> nanoseconds
                    return ts * 1'000'000LL;
                }
                else if (ts < 4'000'000'000'000'000LL) {
                    // Microseconds -> nanoseconds
                    return ts * 1'000LL;
                }
                else {
                    // Already nanoseconds
                    return ts;
                }
            } catch (...) {
                throw std::runtime_error("Could not parse timestamp: " + ts_str);
            }
        }
    };
}