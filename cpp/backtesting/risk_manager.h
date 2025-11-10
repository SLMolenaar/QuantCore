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
    std::string reason;

    bool is_approved() const {
        return result == RiskCheckResult::APPROVED;
    }

    static RiskCheckResponse approve() {
        return {RiskCheckResult::APPROVED, ""};
    }

    static RiskCheckResponse reject(RiskCheckResult result, const std::string& reason) {
        return {result, reason};
    }
};

struct RiskLimits {
    double max_position_pct = 0.20;
    double max_leverage = 2.0;
    double max_loss_pct = 0.50;
    double max_order_value = 0.0;
    bool enabled = true;

    void validate() const {
        if (max_position_pct <= 0.0 || max_position_pct > 1.0) {
            throw std::invalid_argument("max_position_pct must be between 0 and 1");
        }
        if (max_leverage <= 0.0 || max_leverage > 10.0) {
            throw std::invalid_argument("max_leverage must be between 0 and 10");
        }
        if (max_loss_pct <= 0.0 || max_loss_pct > 1.0) {
            throw std::invalid_argument("max_loss_pct must be between 0 and 1");
        }
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

    void set_position(const std::string& symbol, double quantity) {
        positions_[symbol] = quantity;
    }

    double get_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? it->second : 0.0;
    }

    void set_limits(const RiskLimits& limits) {
        limits_ = limits;
        limits_.validate();
    }

    const RiskLimits& get_limits() const {
        return limits_;
    }

    RiskCheckResponse check_order(
        const std::string& symbol,
        Side side,
        double quantity,
        double price
    ) const {
        if (!limits_.enabled) {
            return RiskCheckResponse::approve();
        }

        if (current_capital_ <= 0.0) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_CAPITAL_LIMIT,
                "No capital available"
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

        double current_position = get_position(symbol);
        double new_position = current_position;

        if (side == Side::Buy) {
            new_position += quantity;
        } else {
            new_position -= quantity;
        }

        double position_value = std::abs(new_position) * price;
        double position_pct = position_value / current_capital_;

        if (position_pct > limits_.max_position_pct) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_POSITION_LIMIT,
                "Position would be " + std::to_string(position_pct * 100.0) +
                "% of capital, max is " + std::to_string(limits_.max_position_pct * 100.0) + "%"
            );
        }

        double total_exposure = calculate_total_exposure(symbol, new_position, price);
        double leverage = total_exposure / current_capital_;

        if (leverage > limits_.max_leverage) {
            return RiskCheckResponse::reject(
                RiskCheckResult::REJECTED_LEVERAGE_LIMIT,
                "Leverage would be " + std::to_string(leverage) +
                "x, max is " + std::to_string(limits_.max_leverage) + "x"
            );
        }

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

    void update_position(const std::string& symbol, Side side, double quantity) {
        double current = get_position(symbol);

        if (side == Side::Buy) {
            positions_[symbol] = current + quantity;
        } else {
            positions_[symbol] = current - quantity;
        }
    }

    void reset() {
        positions_.clear();
        initial_capital_ = 0.0;
        current_capital_ = 0.0;
    }

    std::map<std::string, double> get_all_positions() const {
        return positions_;
    }

    double calculate_total_exposure() const {
        return calculate_total_exposure("", 0.0, 0.0);
    }

private:
    RiskLimits limits_;
    double initial_capital_;
    double current_capital_;
    std::map<std::string, double> positions_;

    double calculate_total_exposure(
        const std::string& symbol_to_update,
        double new_position,
        double price
    ) const {
        double total = 0.0;

        for (const auto& [sym, qty] : positions_) {
            if (sym == symbol_to_update) {
                continue;
            }
            total += std::abs(qty);
        }

        if (!symbol_to_update.empty()) {
            total += std::abs(new_position);
        }

        return total;
    }
};

}