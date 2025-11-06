#pragma once

#include "backtesting/strategy.h"
#include "backtesting/market_data_event.h"
#include <deque>
#include <numeric>
#include <cmath>
#include <iostream>

namespace quantcore {

class MeanReversion : public Strategy {
public:
    MeanReversion(size_t lookback = 20, double entry_threshold = 1.5, double exit_threshold = 0.5)
        : Strategy("MeanReversion")
        , lookback_(lookback)
        , entry_threshold_(entry_threshold)
        , exit_threshold_(exit_threshold)
    {
    }

    void on_data(const MarketDataEvent& event) override {
        std::string symbol = event.get_symbol();
        double price = event.get_close();

        price_history_[symbol].push_back(price);

        if (price_history_[symbol].size() > lookback_) {
            price_history_[symbol].pop_front();
        }

        if (price_history_[symbol].size() < lookback_) {
            return;
        }

        double mean = calculate_mean(symbol);
        double std_dev = calculate_std_dev(symbol, mean);

        if (std_dev == 0.0) {
            return;
        }

        double z_score = (price - mean) / std_dev;
        double position = get_position(symbol);

        if (z_score < -entry_threshold_ && position == 0) {
            generate_signal(symbol, SignalType::BUY, 1.0, event.get_timestamp());
            signal_count_++;
        }
        else if (z_score > entry_threshold_ && position == 0) {
            generate_signal(symbol, SignalType::SELL, 1.0, event.get_timestamp());
            signal_count_++;
        }
        else if (position > 0 && z_score > -exit_threshold_) {
            generate_signal(symbol, SignalType::SELL, 1.0, event.get_timestamp());
            signal_count_++;
        }
        else if (position < 0 && z_score < exit_threshold_) {
            generate_signal(symbol, SignalType::BUY, 1.0, event.get_timestamp());
            signal_count_++;
        }
    }

    void reset() override {
        Strategy::reset();
        price_history_.clear();
        signal_count_ = 0;
    }

    int get_signal_count() const { return signal_count_; }

private:
    size_t lookback_;
    double entry_threshold_;
    double exit_threshold_;
    int signal_count_ = 0;

    std::map<std::string, std::deque<double>> price_history_;

    double calculate_mean(const std::string& symbol) {
        const auto& prices = price_history_[symbol];
        if (prices.empty()) return 0.0;

        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        return sum / prices.size();
    }

    double calculate_std_dev(const std::string& symbol, double mean) {
        const auto& prices = price_history_[symbol];

        if (prices.empty()) return 0.0;

        double sq_sum = 0.0;
        for (double price : prices) {
            sq_sum += (price - mean) * (price - mean);
        }

        return std::sqrt(sq_sum / prices.size());
    }
};

}