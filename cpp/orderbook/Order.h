#pragma once

#include <memory>
#include <list>
#include <format>
#include <stdexcept>
#include "Types.h"
#include "Ordertype.h"
#include "Constants.h"

class Order {
    // represents one individual order
public:
    Order(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity)
        : orderType_{orderType}
        , orderId_{orderId}
        , side_{side}
        , price_{price}
        , initialQuantity_{quantity}
        , remainingQuantity_{quantity}
    {
        if (quantity < 1e-8) {
            throw std::invalid_argument("Order quantity must be greater than zero");
        }
    }

    Order(OrderId orderId, Side side, Quantity quantity)
        : Order(OrderType::Market, orderId, side, Constants::InvalidPrice, quantity) {
    }

    OrderId   GetOrderId()          const { return orderId_; }
    Side      GetSide()             const { return side_; }
    Price     GetPrice()            const { return price_; }
    OrderType GetOrderType()        const { return orderType_; }
    Quantity  GetInitialQuantity()  const { return initialQuantity_; }
    Quantity  GetRemainingQuantity()const { return remainingQuantity_; }
    Quantity  GetFilledQuantity()   const { return initialQuantity_ - remainingQuantity_; }
    bool      IsFilled()            const { return remainingQuantity_ < 1e-8; }

    void Fill(Quantity quantity) {
        if (quantity > remainingQuantity_ + 1e-8) {
            throw std::logic_error(std::format("Order ({}) cannot be filled for more than its remaining quantity.",
                                               GetOrderId()));
        }
        remainingQuantity_ -= quantity;
        // Clamp to zero to avoid floating point drift below zero
        if (remainingQuantity_ < 1e-8) remainingQuantity_ = 0.0;
    }

    void ToGoodTillCancel(Price price) {
        if (orderType_ != OrderType::Market) {
            throw std::logic_error("Cannot convert non-market order to GoodTillCancel");
        }
        price_     = price;
        orderType_ = OrderType::GoodTillCancel;
    }

private:
    OrderType orderType_;
    OrderId   orderId_;
    Side      side_;
    Price     price_;
    Quantity  initialQuantity_;
    Quantity  remainingQuantity_;
};