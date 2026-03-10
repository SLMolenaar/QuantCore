#pragma once

#include "../backtesting/strategy.h"
#include "../backtesting/market_data_event.h"
#include <deque>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace quantcore {

// Statistical arbitrage strategy that trades the spread between two correlated assets.
// Enters when the z-score of log(price1 / price2) crosses entry_zscore, and exits
// when the spread reverts back within exit_zscore of the mean.
class PairsTrading : public Strategy {
public:
    PairsTrading(
        const std::string& symbol1,
        const std::string& symbol2,
        size_t lookback     = 20,
        double entry_zscore = 2.0,
        double exit_zscore  = 0.5
    )
        : Strategy("PairsTrading")
        , symbol1_(symbol1)
        , symbol2_(symbol2)
        , lookback_(lookback)
        , entry_zscore_(entry_zscore)
        , exit_zscore_(exit_zscore)
        , direction_(Direction::NONE)
        , price1_(0.0)
        , price2_(0.0)
        , symbol1_seen_(false)
        , symbol2_seen_(false)
        , warning_issued_(false)
        , bars_processed_(0)
    {
        if (symbol1 == symbol2)
            throw std::invalid_argument("Symbols must be different for pairs trading");
        if (lookback == 0)
            throw std::invalid_argument("Lookback must be positive");
        if (exit_zscore >= entry_zscore)
            throw std::invalid_argument("exit_zscore must be less than entry_zscore");
    }

    void on_data(const MarketDataEvent& event) override {
        const auto& symbol = event.get_symbol();

        if (symbol == symbol1_) {
            price1_ = event.get_close();
            symbol1_seen_ = true;
        } else if (symbol == symbol2_) {
            price2_ = event.get_close();
            symbol2_seen_ = true;
        } else {
            return;
        }

        if (!warning_issued_ && (symbol1_seen_ != symbol2_seen_)) {
            // We've seen at least some bars, but only one symbol
            // Check if we've waited long enough (more than lookback bars)
            size_t bars_seen = symbol1_seen_ ?
                spread_history_.size() : 0;
            if (symbol2_seen_ && !symbol1_seen_) {
                bars_seen = spread_history_.size();
            }

            // After a reasonable amount of data, warn if we're still missing a symbol
            if (bars_seen == 0) {
                // First bar - issue warning immediately if one symbol arrives but not both
                // Wait for more data before warning
            }
        }

        if (price1_ == 0.0 || price2_ == 0.0) {
            if (!warning_issued_) {
                bool one_seen  = symbol1_seen_ || symbol2_seen_;
                bool both_seen = symbol1_seen_ && symbol2_seen_;
                if (one_seen && !both_seen) {
                    bars_processed_++;
                    if (bars_processed_ > lookback_) {
                        std::cerr << "WARNING: PairsTrading strategy is missing data for ";
                        if (!symbol1_seen_) {
                            std::cerr << "symbol1 (" << symbol1_ << ")";
                        } else {
                            std::cerr << "symbol2 (" << symbol2_ << ")";
                        }
                        std::cerr << ". No signals will be generated until both symbols have data.\n";
                        warning_issued_ = true;
                    }
                }
            }
            return;
        }

        double spread = std::log(price1_ / price2_);
        spread_history_.push_back(spread);
        if (spread_history_.size() > lookback_) spread_history_.pop_front();
        if (spread_history_.size() < lookback_) return;

        double mean   = calculate_mean();
        double stddev = calculate_std(mean);
        if (stddev < 1e-8) return;

        double zscore = (spread - mean) / stddev;

        if (direction_ == Direction::NONE) {
            if (zscore > entry_zscore_) {
                // Spread is too high: sell sym1, buy sym2
                generate_signal(symbol1_, SignalType::SELL, 0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::BUY,  0.5, event.get_timestamp());
                direction_ = Direction::SHORT_SPREAD;
            } else if (zscore < -entry_zscore_) {
                // Spread is too low: buy sym1, sell sym2
                generate_signal(symbol1_, SignalType::BUY,  0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::SELL, 0.5, event.get_timestamp());
                direction_ = Direction::LONG_SPREAD;
            }
        } else if (std::abs(zscore) < exit_zscore_) {
            // Spread has reverted, unwind in the direction opposite to entry.
            if (direction_ == Direction::LONG_SPREAD) {
                generate_signal(symbol1_, SignalType::SELL, 0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::BUY,  0.5, event.get_timestamp());
            } else {
                generate_signal(symbol1_, SignalType::BUY,  0.5, event.get_timestamp());
                generate_signal(symbol2_, SignalType::SELL, 0.5, event.get_timestamp());
            }
            direction_ = Direction::NONE;
        }
    }

    void reset() override {
        Strategy::reset();
        spread_history_.clear();
        price1_          = 0.0;
        price2_          = 0.0;
        direction_       = Direction::NONE;
        symbol1_seen_    = false;
        symbol2_seen_    = false;
        warning_issued_  = false;
        bars_processed_  = 0;
    }

    bool in_trade() const { return direction_ != Direction::NONE; }
    bool has_both_symbols() const { return symbol1_seen_ && symbol2_seen_; }
    bool has_symbol1() const { return symbol1_seen_; }
    bool has_symbol2() const { return symbol2_seen_; }

private:
    // Tracks which leg was bought vs. sold so that exit signals are always
    // the correct mirror of the entry.
    enum class Direction { NONE, LONG_SPREAD, SHORT_SPREAD };

    std::string symbol1_;
    std::string symbol2_;
    size_t      lookback_;
    double      entry_zscore_;
    double      exit_zscore_;

    Direction          direction_;
    std::deque<double> spread_history_;
    double             price1_;
    double             price2_;

    bool   symbol1_seen_;
    bool   symbol2_seen_;
    bool   warning_issued_;
    size_t bars_processed_;

    double calculate_mean() const {
        double sum = 0.0;
        for (double v : spread_history_) sum += v;
        return sum / spread_history_.size();
    }

    double calculate_std(double mean) const {
        double sq_sum = 0.0;
        for (double v : spread_history_) {
            double diff = v - mean;
            sq_sum += diff * diff;
        }
        return std::sqrt(sq_sum / spread_history_.size());
    }
};

}