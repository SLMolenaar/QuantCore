#pragma once

#include "backtesting/strategy.h"
#include "backtesting/market_data_event.h"
#include <deque>
#include <numeric>
#include <stdexcept>

namespace quantcore {

class SMACrossover : public Strategy {
public:
    SMACrossover(size_t fast_period = 50, size_t slow_period = 200)
        : Strategy("SMACrossover")
        , fast_period_(fast_period)
        , slow_period_(slow_period)
    {
        if (fast_period == 0 || slow_period == 0) {
            throw std::invalid_argument("SMA periods must be greater than 0");
        }
        if (fast_period >= slow_period) {
            throw std::invalid_argument("Fast period must be less than slow period");
        }
    }

    void on_data(const MarketDataEvent& event) override {
        std::string symbol = event.get_symbol();
        double price = event.get_close();

        price_history_[symbol].push_back(price);

        if (price_history_[symbol].size() > slow_period_) {
            price_history_[symbol].pop_front();
        }

        if (price_history_[symbol].size() < slow_period_) {
            return;
        }

        double fast_sma = calculate_sma(symbol, fast_period_);
        double slow_sma = calculate_sma(symbol, slow_period_);

        bool curr_bullish = fast_sma > slow_sma;
        double position = get_position(symbol);

        if (has_previous_state_[symbol]) {
            bool prev_bullish = fast_sma_prev_[symbol] > slow_sma_prev_[symbol];

            if (!prev_bullish && curr_bullish && position <= 0) {
                generate_signal(symbol, SignalType::BUY, 1.0, event.get_timestamp());
            }
            else if (prev_bullish && !curr_bullish && position >= 0) {
                generate_signal(symbol, SignalType::SELL, 1.0, event.get_timestamp());
            }
        }

        has_previous_state_[symbol] = true;
        fast_sma_prev_[symbol] = fast_sma;
        slow_sma_prev_[symbol] = slow_sma;
    }

    void reset() override {
        Strategy::reset();
        price_history_.clear();
        fast_sma_prev_.clear();
        slow_sma_prev_.clear();
        has_previous_state_.clear();
    }

private:
    size_t fast_period_;
    size_t slow_period_;

    std::map<std::string, std::deque<double>> price_history_;
    std::map<std::string, double> fast_sma_prev_;
    std::map<std::string, double> slow_sma_prev_;
    std::map<std::string, bool> has_previous_state_;
    
    double calculate_sma(const std::string& symbol, size_t period) {
        const auto& prices = price_history_[symbol];
        
        if (prices.size() < period) {
            return 0.0;
        }

        size_t start_idx = prices.size() - period;
        double sum = 0.0;
        for (size_t i = start_idx; i < prices.size(); ++i) {
            sum += prices[i];
        }
        return sum / period;
    }
};

}