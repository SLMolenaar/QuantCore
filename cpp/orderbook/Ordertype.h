#pragma once

enum class OrderType {
    GoodTillCancel,     // Active until completely filled
    ImmediateOrCancel,  // Fill for as far as possible and kill immediately
    Market,             // Fill at any price
    GoodForDay,         // Cancelled at a specific time every day
    FillOrKill,         // Fill fully or kill immediately
    Stop,               // Becomes a Market order when stop_price is touched
    StopLimit           // Becomes a GoodTillCancel limit when stop_price is touched
};

enum class Side {
    Buy,
    Sell
};
