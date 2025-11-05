#pragma once

#include "backtesting/strategy.h"
#include "backtesting/market_data_event.h"

namespace quantcore {

    // Just for testing the backtesting engine
    class BuyAndHold : public Strategy {
    public:
        BuyAndHold() : Strategy("BuyAndHold"), bought_(false) {}

        void on_data(const MarketDataEvent& event) override {
            // Buy once on first bar
            if (!bought_) {
                generate_signal(event.get_symbol(), SignalType::BUY, 1.0, event.get_timestamp());
                bought_ = true;
            }
        }

        void reset() override {
            Strategy::reset();
            bought_ = false;
        }

    private:
        bool bought_;
    };

}