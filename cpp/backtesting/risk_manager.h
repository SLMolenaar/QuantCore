#pragma once

#include <string>
#include <map>
#include <memory>
#include <optional>
#include "orderbook/Order.h"
#include "orderbook/Types.h"

namespace quantcore {

enum class RiskCheckResult {
    APPROVED,
    REJECTED_POSITION_LIMIT,
    REJECTED_LEVERAGE_LIMIT,
    REJECTED_CAPITAL_LIMIT,
    REJECTED_LOSS_LIMIT,
    REJECTED_ORDER_SIZE
};

struct RiskCheckResponse {
    RiskCheckResult result;
    std::string     reason;

    bool is_approved() const { return result == RiskCheckResult::APPROVED; }

    static RiskCheckResponse approve() {
        return {RiskCheckResult::APPROVED, ""};
    }

    static RiskCheckResponse reject(RiskCheckResult result, const std::string& reason) {
        return {result, reason};
    }
};

struct RiskLimits {
    double max_position_pct  = 0.20;   // max single-asset notional / capital; values > 1.0 allow leverage
    double max_leverage      = 2.0;    // max total notional / capital
    double max_loss_pct      = 0.50;   // max drawdown from initial capital
    double max_order_value   = 0.0;    // 0 = disabled
    bool   enabled           = true;

    void validate() const {
        if (max_position_pct <= 0.0)
            throw std::invalid_argument("max_position_pct must be positive");
        // No upper bound: values > 1.0 allow per-asset leverage, which is
        // valid when the user explicitly configures it.
        if (max_leverage <= 0.0)
            throw std::invalid_argument("max_leverage must be positive");
        if (max_loss_pct <= 0.0 || max_loss_pct > 1.0)
            throw std::invalid_argument("max_loss_pct must be between 0 and 1");
    }
};

class RiskManager {
public:
    explicit RiskManager(const RiskLimits& limits = RiskLimits())
        : limits_(limits)
        , initial_capital_(0.0)
        , current_capital_(0.0)
    {
        limits_.validate();
    }

    void set_capital(double initial, double current) {
        initial_capital_ = initial;
        current_capital_ = current;
    }

    // Update the tracked quantity and last-known price for a symbol.
    // Providing a price enables accurate notional-based leverage checks.
    // A price of 0 leaves the existing notional unchanged (use when only
    // correcting quantity after a partial fill).
    //
    // IMPORTANT: always provide a price when leverage above 1x is in use.
    // Calling with price == 0 retains the previous notional, which will be
    // stale after a price move and will cause incorrect leverage calculations
    // in check_order(). Prefer set_position(sym, qty, price) over
    // update_position(sym, side, qty) in any leveraged context.
    void set_position(const std::string& symbol, double quantity, double price = 0.0) {
        positions_[symbol] = quantity;

        if (quantity == 0.0) {
            position_notionals_.erase(symbol);
            last_prices_.erase(symbol);
        } else if (price > 0.0) {
            position_notionals_[symbol] = std::abs(quantity) * price;
            last_prices_[symbol] = price;
        }
        // If price == 0 and quantity != 0, the notional entry from the previous
        // fill is retained — better than zeroing it out with no price information.
    }

    void update_price(const std::string& symbol, double price) {
        if (price <= 0.0) return;

        last_prices_[symbol] = price;

        auto pos_it = positions_.find(symbol);
        if (pos_it != positions_.end() && pos_it->second != 0.0) {
            position_notionals_[symbol] = std::abs(pos_it->second) * price;
        }
    }

    void update_prices(const std::map<std::string, double>& prices) {
        for (const auto& [symbol, price] : prices) {
            update_price(symbol, price);
        }
    }

    double get_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? it->second : 0.0;
    }

    double get_last_price(const std::string& symbol) const {
        auto it = last_prices_.find(symbol);
        return it != last_prices_.end() ? it->second : 0.0;
    }

    void set_limits(const RiskLimits& limits) {
        limits_ = limits;
        limits_.validate();
    }

    const RiskLimits& get_limits() const { return limits_; }

    RiskCheckResponse check_order(
        const std::string& symbol,
        Side   side,
        double quantity,
        double price
    ) const {
        if (!limits_.enabled) return RiskCheckResponse::approve();

        if (current_capital_ <= 0.0) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_CAPITAL_LIMIT, "No capital available"
            );
        }

        double order_value = quantity * price;

        if (limits_.max_order_value > 0.0 && order_value > limits_.max_order_value) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_ORDER_SIZE,
                "Order value " + std::to_string(order_value) +
                " exceeds max order value " + std::to_string(limits_.max_order_value)
            );
        }

        double current_qty      = get_position(symbol);
        double new_qty          = (side == Side::Buy)
                                    ? current_qty + quantity
                                    : current_qty - quantity;
        double current_notional = std::abs(current_qty) * price;
        double new_notional     = std::abs(new_qty) * price;

        // Only apply the position limit check when the order increases notional
        // exposure. Reduces and closes always decrease risk and must never be
        // blocked by a sizing limit — doing so would trap the strategy in a
        // position it cannot exit.
        if (new_notional > current_notional) {
            double position_pct = new_notional / current_capital_;
            if (position_pct > limits_.max_position_pct) {
                return RiskCheckResponse::reject(
                    RiskCheckResult::REJECTED_POSITION_LIMIT,
                    "Position would be " + std::to_string(position_pct * 100.0) +
                    "% of capital, max is " + std::to_string(limits_.max_position_pct * 100.0) + "%"
                );
            }
        }

        // Leverage is total portfolio notional exposure / capital.
        // We compute it using the last-known notional for each symbol so that
        // positions in different instruments are compared on equal footing.
        // Only check when the order increases total exposure.
        double total_exposure = calculate_total_exposure(symbol, new_notional);
        if (new_notional > current_notional) {
            double leverage = total_exposure / current_capital_;
            if (leverage > limits_.max_leverage) {
                return RiskCheckResponse::reject(
                    RiskCheckResult::REJECTED_LEVERAGE_LIMIT,
                    "Leverage would be " + std::to_string(leverage) +
                    "x, max is " + std::to_string(limits_.max_leverage) + "x"
                );
            }
        }

        // The loss limit is a hard drawdown stop and is always checked regardless
        // of order direction — we do not want to allow new entries when the
        // portfolio is already past the configured loss threshold.
        double loss_pct = (initial_capital_ - current_capital_) / initial_capital_;
        if (loss_pct > limits_.max_loss_pct) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_LOSS_LIMIT,
                "Portfolio down " + std::to_string(loss_pct * 100.0) +
                "%, max loss is " + std::to_string(limits_.max_loss_pct * 100.0) + "%"
            );
        }

        return RiskCheckResponse::approve();
    }

    // Updates position quantity only; does not update the stored notional.
    // Prefer set_position(symbol, quantity, price) in any context where
    // accurate leverage calculations matter, especially when leverage > 1x
    // is in use and notional staleness could cause incorrect check_order() results.
    void update_position(const std::string& symbol, Side side, double quantity) {
        double current = get_position(symbol);
        positions_[symbol] = (side == Side::Buy)
            ? current + quantity
            : current - quantity;
    }

    void reset() {
        positions_.clear();
        position_notionals_.clear();
        last_prices_.clear();
        initial_capital_ = 0.0;
        current_capital_ = 0.0;
    }

    std::map<std::string, double> get_all_positions() const { return positions_; }

    std::map<std::string, double> get_all_prices() const { return last_prices_; }

    // Sum of absolute notional exposures across all positions.
    double calculate_total_exposure() const {
        double total = 0.0;
        for (const auto& [sym, notional] : position_notionals_) total += notional;
        return total;
    }

private:
    RiskLimits limits_;
    double     initial_capital_;
    double     current_capital_;

    std::map<std::string, double> positions_;
    // Last-known notional value (|qty| * price) per symbol, updated on each fill.
    std::map<std::string, double> position_notionals_;
    std::map<std::string, double> last_prices_;

    // Returns total portfolio notional after hypothetically replacing `symbol`'s
    // exposure with `new_notional`. Uses stored notionals for all other symbols.
    double calculate_total_exposure(
        const std::string& symbol_to_update,
        double new_notional
    ) const {
        double total = 0.0;
        for (const auto& [sym, notional] : position_notionals_) {
            if (sym == symbol_to_update) continue;
            total += notional;
        }
        total += new_notional;
        return total;
    }
};

}