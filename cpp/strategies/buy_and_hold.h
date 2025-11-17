#pragma once

#include "backtesting/strategy.h"
#include "backtesting/market_data_event.h"
#include <set>

namespace quantcore {

    class BuyAndHold : public Strategy {
    public:
        BuyAndHold() : Strategy("BuyAndHold") {}

        void on_data(const MarketDataEvent& event) override {
            std::string symbol = event.get_symbol();

            // Buy once PER SYMBOL
            if (bought_symbols_.find(symbol) == bought_symbols_.end()) {
                generate_signal(symbol, SignalType::BUY, 1.0, event.get_timestamp());
                bought_symbols_.insert(symbol);
            }
        }

        void reset() override {
            Strategy::reset();
            bought_symbols_.clear();
        }

    private:
        std::set<std::string> bought_symbols_;  // Track per symbol, not global
    };

}