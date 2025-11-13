#pragma once

#include "event.h"
#include "../orderbook/OrderType.h"
#include <string>

namespace quantcore {

    //  Order execution confirmation, informs portfolio manager to update positions and PnL
    class FillEvent : public Event {
    public:
        FillEvent(
            const std::string& symbol,
            int64_t timestamp_ns,
            uint64_t order_id,
            Side side,
            double quantity,
            double price,
            double commission = 0.0
        )
            : Event(EventType::FILL, timestamp_ns)
            , symbol_(symbol)
            , order_id_(order_id)
            , side_(side)
            , quantity_(quantity)
            , price_(price)
            , commission_(commission)
        {
        }

        std::string get_symbol() const { return symbol_; }
        uint64_t get_order_id() const { return order_id_; }
        Side get_side() const { return side_; }
        double get_quantity() const { return quantity_; }
        double get_price() const { return price_; }
        double get_commission() const { return commission_; }

        double get_total_cost() const {
            double notional = quantity_ * price_;
            return (side_ == Side::Buy) ? (notional + commission_) : (notional - commission_);
        }

        std::string to_string() const override {
            std::string side_str = (side_ == Side::Buy) ? "BUY" : "SELL";
            return "FillEvent(symbol=" + symbol_ +
                   ", side=" + side_str +
                   ", qty=" + std::to_string(quantity_) +
                   ", price=" + std::to_string(price_) +
                   ", commission=" + std::to_string(commission_) + ")";
        }

    private:
        std::string symbol_;
        uint64_t order_id_;
        Side side_;
        double quantity_;
        double price_;
        double commission_;
    };

}