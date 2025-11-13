#pragma once

#include "../backtesting/strategy.h"
#include "../backtesting/market_data_event.h"
#include <deque>
#include <cmath>
#include <stdexcept>

namespace quantcore {

//Pairs trading strategy
// trades the espread between two correlated assets.
class PairsTrading : public Strategy {
public:
    PairsTrading(
        const std::string& symbol1,
        const std::string& symbol2,
        size_t lookback = 20,
        double entry_zscore = 2.0,
        double exit_zscore = 0.5
    )
        : Strategy("PairsTrading")
        , symbol1_(symbol1)
        , symbol2_(symbol2)
        , lookback_(lookback)
        , entry_zscore_(entry_zscore)
        , exit_zscore_(exit_zscore)
        , in_trade_(false)
    {
        if (symbol1 == symbol2) {
            throw std::invalid_argument("Symbols must be different for pairs trading");
        }
        if (lookback == 0) {
            throw std::invalid_argument("Lookback must be positive");
        }
    }

    void on_data(const MarketDataEvent& event) override {
        std::string symbol = event.get_symbol();

        if (symbol == symbol1_) {
            price1_ = event.get_close();
        } else if (symbol == symbol2_) {
            price2_ = event.get_close();
        } else {
            return;
        }

        if (price1_ == 0.0 || price2_ == 0.0) {
            return;
        }

        double spread = calculate_spread();
        spread_history_.push_back(spread);

        if (spread_history_.size() > lookback_) {
            spread_history_.pop_front();
        }

        if (spread_history_.size() < lookback_) {
            return;
        }

        double mean = calculate_mean();
        double std = calculate_std(mean);

        if (std < 1e-8) {
            return;
        }

        double zscore = (spread - mean) / std;

        if (!in_trade_) {
            if (zscore > entry_zscore_) {
                generate_signal(symbol1_, SignalType::SELL, 0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::BUY, 0.5, event.get_timestamp());
                in_trade_ = true;
            } else if (zscore < -entry_zscore_) {
                generate_signal(symbol1_, SignalType::BUY, 0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::SELL, 0.5, event.get_timestamp());
                in_trade_ = true;
            }
        } else {
            if (std::abs(zscore) < exit_zscore_) {
                if (has_position(symbol1_) || has_position(symbol2_)) {
                    generate_signal(symbol1_, SignalType::SELL, 0.0, event.get_timestamp());
                    generate_signal(symbol2_, SignalType::SELL, 0.0, event.get_timestamp());
                    in_trade_ = false;
                }
            }
        }
    }

    void reset() override {
        Strategy::reset();
        spread_history_.clear();
        price1_ = 0.0;
        price2_ = 0.0;
        in_trade_ = false;
    }

private:
    std::string symbol1_;
    std::string symbol2_;
    size_t lookback_;
    double entry_zscore_;
    double exit_zscore_;

    std::deque<double> spread_history_;
    double price1_ = 0.0;
    double price2_ = 0.0;
    bool in_trade_;

    double calculate_spread() const {
        return std::log(price1_ / price2_);
    }

    double calculate_mean() const {
        double sum = 0.0;
        for (double val : spread_history_) {
            sum += val;
        }
        return sum / spread_history_.size();
    }

    double calculate_std(double mean) const {
        double sum_sq = 0.0;
        for (double val : spread_history_) {
            double diff = val - mean;
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / spread_history_.size());
    }
};

}