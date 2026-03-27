#pragma once

#include "event.h"
#include "../orderbook/Ordertype.h"
#include <string>

namespace quantcore {

// Command to place/modify/cancel an order.
// For Stop orders:      price_ is unused; stop_price_ is the trigger level.
// For StopLimit orders: price_ is the limit price; stop_price_ is the trigger level.
// For all other types:  stop_price_ is 0.0 and ignored.
class OrderEvent : public Event {
public:
    // New order constructor.
    OrderEvent(
        const std::string& symbol,
        int64_t timestamp_ns,
        Side side,
        OrderType order_type,
        double quantity,
        double price = 0.0,
        uint64_t order_id = 0,
        double stop_price = 0.0
    )
        : Event(EventType::ORDER, timestamp_ns)
        , symbol_(symbol)
        , side_(side)
        , order_type_(order_type)
        , quantity_(quantity)
        , price_(price)
        , order_id_(order_id)
        , is_cancel_(false)
        , stop_price_(stop_price)
    {
    }

    // Cancel order constructor.
    static OrderEvent create_cancel(
        const std::string& symbol,
        int64_t timestamp_ns,
        uint64_t order_id
    ) {
        OrderEvent event(symbol, timestamp_ns, Side::Buy,
                         OrderType::GoodTillCancel, 0, 0, order_id);
        event.is_cancel_ = true;
        return event;
    }

    std::string get_symbol()     const { return symbol_; }
    Side        get_side()       const { return side_; }
    OrderType   get_order_type() const { return order_type_; }
    double      get_quantity()   const { return quantity_; }
    double      get_price()      const { return price_; }
    uint64_t    get_order_id()   const { return order_id_; }
    bool        is_cancel()      const { return is_cancel_; }
    double      get_stop_price() const { return stop_price_; }

    void set_order_id(uint64_t id) { order_id_ = id; }

    std::string to_string() const override {
        if (is_cancel_) {
            return "OrderEvent(CANCEL, order_id=" + std::to_string(order_id_) + ")";
        }

        std::string side_str = (side_ == Side::Buy) ? "BUY" : "SELL";
        std::string type_str;
        switch (order_type_) {
            case OrderType::Market:            type_str = "MARKET";    break;
            case OrderType::GoodTillCancel:    type_str = "LIMIT";     break;
            case OrderType::ImmediateOrCancel: type_str = "IOC";       break;
            case OrderType::FillOrKill:        type_str = "FOK";       break;
            case OrderType::GoodForDay:        type_str = "GFD";       break;
            case OrderType::Stop:              type_str = "STOP";      break;
            case OrderType::StopLimit:         type_str = "STOPLIMIT"; break;
        }

        std::string s = "OrderEvent(symbol=" + symbol_ +
                        ", side=" + side_str +
                        ", type=" + type_str +
                        ", qty=" + std::to_string(quantity_);
        if (order_type_ == OrderType::Stop || order_type_ == OrderType::StopLimit)
            s += ", stop_price=" + std::to_string(stop_price_);
        if (order_type_ != OrderType::Stop && order_type_ != OrderType::Market)
            s += ", price=" + std::to_string(price_);
        s += ")";
        return s;
    }

private:
    std::string symbol_;
    Side        side_;
    OrderType   order_type_;
    double      quantity_;
    double      price_;
    uint64_t    order_id_;
    bool        is_cancel_;
    double      stop_price_; // trigger level; 0.0 for non-stop order types
};

} // namespace quantcore
