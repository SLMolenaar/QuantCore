#pragma once

#include <cmath>
#include "Types.h"

// Exchange trading rules
struct ExchangeRules {
    Price    tickSize    = 1;        // min price increment (in cents)
    Quantity lotSize     = 1.0;      // min quantity increment
    Quantity minQuantity = 1e-8;     // min order size (near-zero guard)
    Quantity maxQuantity = 1000000;  // max order size
    Price    minNotional = 0;        // min order value (price * quantity)

    bool IsValidPrice(Price price) const {
        if (price <= 0) return false;
        return price % tickSize == 0;
    }

    bool IsValidQuantity(Quantity quantity) const {
        if (quantity < minQuantity || quantity > maxQuantity) return false;
        // Lot size enforcement via fmod — works correctly for double Quantity.
        // A lotSize of 1.0 (default) passes everything since fmod(x, 1.0) == 0
        // for all whole numbers, and fractional shares are multiples of 1e-8.
        if (lotSize > 1.0 + 1e-9) {
            return std::fmod(quantity, lotSize) < 1e-9;
        }
        return true;
    }

    bool IsValidNotional(Price price, Quantity quantity) const {
        double notional = static_cast<double>(price) * quantity;
        return notional >= static_cast<double>(minNotional);
    }

    bool IsValidOrder(Price price, Quantity quantity) const {
        return IsValidPrice(price) &&
               IsValidQuantity(quantity) &&
               IsValidNotional(price, quantity);
    }

    Price RoundToTick(Price price) const {
        if (tickSize <= 1) return price;
        return (price / tickSize) * tickSize;
    }

    Quantity RoundToLot(Quantity quantity) const {
        // No rounding — fractional shares are supported
        return quantity;
    }
};

enum class RejectReason {
    None,
    InvalidPrice,
    InvalidQuantity,
    BelowMinQuantity,
    AboveMaxQuantity,
    BelowMinNotional,
    DuplicateOrderId,
    InvalidOrderType,
    EmptyBook
};

// Structure to hold order validation result
struct OrderValidation {
    bool         isValid = true;
    RejectReason reason  = RejectReason::None;

    static OrderValidation Accept() {
        return OrderValidation{true, RejectReason::None};
    }

    static OrderValidation Reject(RejectReason reason) {
        return OrderValidation{false, reason};
    }
};