#pragma once

#include <map>
#include <cmath>
#include <string>
#include <memory>
#include "../Execution.h"

namespace quantcore {

/**
 * Portfolio context for multi-asset strategies
 *
 * Provides strategies with complete portfolio state including:
 * - Current positions across all symbols
 * - Mark-to-market values
 * - Available capital
 * - Leverage and exposure metrics
 *
 * This enables strategies to make portfolio-level decisions
 * rather than treating each asset independently.
 */
class PortfolioContext {
public:
    PortfolioContext(double initial_capital)
        : initial_capital_(initial_capital)
        , cash_(initial_capital)
        , cash_balance_(initial_capital)
    {}

    void update_price(const std::string& symbol, double price) {
        current_prices_[symbol] = price;
    }

    void update_position(const std::string& symbol, double quantity) {
        if (quantity == 0.0) {
            positions_.erase(symbol);
        } else {
            positions_[symbol] = quantity;
        }
    }

    // "Cash" is uncommitted capital: equity minus the gross (absolute) value of
    // all positions. Longs and shorts commit capital symmetrically, so this is
    // buying power rather than a literal cash balance (short-sale proceeds are
    // not added back). get_portfolio_value() = cash + gross value = equity.
    void set_cash(double cash) {
        cash_ = cash;
    }

    double get_cash() const {
        return cash_;
    }

    // Literal cash balance: equity minus the *signed* market value of positions.
    // Unlike get_cash(), short-sale proceeds are included (a 10k short adds ~10k).
    void set_cash_balance(double balance) { cash_balance_ = balance; }
    double get_cash_balance() const { return cash_balance_; }

    double get_initial_capital() const {
        return initial_capital_;
    }

    double get_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? it->second : 0.0;
    }

    double get_price(const std::string& symbol) const {
        auto it = current_prices_.find(symbol);
        return it != current_prices_.end() ? it->second : 0.0;
    }

    double get_position_value(const std::string& symbol) const {
        return get_position(symbol) * get_price(symbol);
    }

    double get_total_position_value() const {
        double total = 0.0;
        for (const auto& [symbol, quantity] : positions_) {
            total += std::abs(quantity * get_price(symbol));
        }
        return total;
    }

    double get_portfolio_value() const {
        return cash_ + get_total_position_value();
    }

    double get_leverage() const {
        double total_exposure = get_total_position_value();
        double equity = get_portfolio_value();
        return equity > 0.0 ? total_exposure / equity : 0.0;
    }

    double get_position_weight(const std::string& symbol) const {
        double portfolio_value = get_portfolio_value();
        if (portfolio_value == 0.0) return 0.0;
        return get_position_value(symbol) / portfolio_value;
    }

    std::map<std::string, double> get_all_positions() const {
        return positions_;
    }

    std::map<std::string, double> get_all_prices() const {
        return current_prices_;
    }

    size_t num_positions() const {
        return positions_.size();
    }

    bool has_position(const std::string& symbol) const {
        return positions_.find(symbol) != positions_.end() &&
               positions_.at(symbol) != 0.0;
    }

private:
    double initial_capital_;
    double cash_;
    double cash_balance_;
    std::map<std::string, double> positions_;
    std::map<std::string, double> current_prices_;
};

}