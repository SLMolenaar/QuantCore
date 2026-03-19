#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "../orderbook/Ordertype.h"

namespace quantcore {

    // single trade tick
    struct TickData {
        std::string symbol;
        int64_t     timestamp_ns;
        double      price;
        double      quantity;
        Side        aggressor_side; // which side initiated the trade

        TickData() = default;

        TickData(const std::string& sym, int64_t ts, double px, double qty,
                 Side side = Side::Buy)
            : symbol(sym)
            , timestamp_ns(ts)
            , price(px)
            , quantity(qty)
            , aggressor_side(side)
        {
            if (price <= 0.0)
                throw std::invalid_argument("Tick price must be positive");
            if (quantity < 1e-8)
                throw std::invalid_argument("Tick quantity must be positive");
        }
    };

    using TickSeries = std::vector<TickData>;

} // namespace quantcore