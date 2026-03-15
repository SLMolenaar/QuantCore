#pragma once

#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

namespace quantcore {

struct PositionSizingContext {
    double signal_strength;
    double current_capital;
    double current_price;
    double current_position;
    double portfolio_volatility;
    double stop_loss_distance;

    PositionSizingContext(
        double strength      = 1.0,
        double capital       = 100000.0,
        double price         = 100.0,
        double position      = 0.0,
        double volatility    = 0.02,
        double stop_distance = 0.05
    )
        : signal_strength(strength)
        , current_capital(capital)
        , current_price(price)
        , current_position(position)
        , portfolio_volatility(volatility)
        , stop_loss_distance(stop_distance)
    {}
};

class PositionSizer {
public:
    virtual ~PositionSizer() = default;

    virtual double calculate_size(const PositionSizingContext& ctx) = 0;
    virtual std::string get_name() const = 0;

    void set_max_position_size(double max_size) { max_position_size_ = max_size; }
    void set_min_position_size(double min_size) { min_position_size_ = min_size; }
    void set_max_leverage(double max_lev)       { max_leverage_ = max_lev; }

protected:
    double apply_constraints(double raw_size, const PositionSizingContext& ctx) const {
        if (ctx.current_capital <= 0.0) return 0.0;

        double size = raw_size;

        if (min_position_size_ > 0 && std::abs(size) < min_position_size_) {
            return 0.0;
        }

        if (max_position_size_ > 0 && std::abs(size) > max_position_size_) {
            size = std::copysign(max_position_size_, size);
        }

        // max_leverage_ == 0 means no cap (the sizer defers to RiskLimits on
        // the engine). Derived classes that want an uncapped default set it to
        // 0.0 in their constructor.
        if (max_leverage_ > 0) {
            double max_notional = ctx.current_capital * max_leverage_;
            // Floor max_shares so the resulting notional never exceeds the
            // intended limit due to floating-point rounding.
            double max_shares   = std::floor(max_notional / ctx.current_price);
            if (std::abs(size) > max_shares) {
                size = std::copysign(max_shares, size);
            }
        }

        // Always return a whole number of shares. Fractional shares are not
        // supported and would cause the notional to drift above intended limits.
        return std::floor(std::abs(size)) * (size >= 0.0 ? 1.0 : -1.0);
    }

    double max_position_size_ = 0.0;
    double min_position_size_ = 0.0;
    // Default cap of 1x leverage. Derived classes that support or require
    // leverage above 1x should set this to 0.0 (disabled) in their constructor
    // so the engine's RiskLimits remain the sole ceiling.
    double max_leverage_      = 1.0;
};

class FixedPercentage : public PositionSizer {
public:
    // pct is the fraction of capital to allocate per position.
    // Values above 1.0 imply leverage (e.g. 2.0 = 200% of capital).
    // The sizer's internal max_leverage_ cap is disabled by default so that
    // pct > 1.0 is not silently clipped; the engine's RiskLimits act as the
    // ceiling. Call set_max_leverage() to impose an additional sizer-level cap.
    explicit FixedPercentage(double pct = 0.1) : pct_(pct) {
        if (pct <= 0.0)
            throw std::invalid_argument("Percentage must be positive");
        max_leverage_ = 0.0;
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        // Floor to whole shares first so notional is always <= capital * pct_.
        double shares = std::floor(ctx.current_capital * pct_ * ctx.signal_strength
                                   / ctx.current_price);

        // Belt-and-suspenders: if floating-point arithmetic still produces a
        // notional fractionally above the limit, subtract one share.
        while (shares > 0.0 && shares * ctx.current_price > ctx.current_capital * pct_) {
            shares -= 1.0;
        }

        return apply_constraints(shares, ctx);
    }

    std::string get_name() const override {
        return "FixedPercentage(" + std::to_string(pct_ * 100) + "%)";
    }

private:
    double pct_;
};

class RiskBased : public PositionSizer {
public:
    explicit RiskBased(double risk_per_trade = 0.01) : risk_per_trade_(risk_per_trade) {
        if (risk_per_trade <= 0.0 || risk_per_trade > 0.1)
            throw std::invalid_argument("Risk per trade must be between 0 and 0.1");
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        if (ctx.stop_loss_distance <= 0.0) return 0.0;
        double shares = (ctx.current_capital * risk_per_trade_
                         / (ctx.current_price * ctx.stop_loss_distance))
                        * ctx.signal_strength;
        return apply_constraints(shares, ctx);
    }

    std::string get_name() const override {
        return "RiskBased(" + std::to_string(risk_per_trade_ * 100) + "%)";
    }

private:
    double risk_per_trade_;
};

class KellyCriterion : public PositionSizer {
public:
    KellyCriterion(double win_rate, double avg_win, double avg_loss, double fraction = 1.0)
        : win_rate_(win_rate), avg_win_(avg_win), avg_loss_(avg_loss), kelly_frac_(fraction)
    {
        if (win_rate <= 0.0 || win_rate >= 1.0)
            throw std::invalid_argument("Win rate must be between 0 and 1");
        if (avg_win <= 0.0 || avg_loss <= 0.0)
            throw std::invalid_argument("Average win and loss must be positive");
        if (kelly_frac_ <= 0.0 || kelly_frac_ > 1.0)
            throw std::invalid_argument("Kelly fraction must be between 0 and 1");
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        double wl_ratio  = avg_win_ / avg_loss_;
        double kelly_pct = std::max(0.0,
            (win_rate_ * wl_ratio - (1.0 - win_rate_)) / wl_ratio) * kelly_frac_;
        double shares = (ctx.current_capital * kelly_pct / ctx.current_price)
                        * ctx.signal_strength;
        return apply_constraints(shares, ctx);
    }

    std::string get_name() const override {
        return "KellyCriterion(wr=" + std::to_string(win_rate_)
             + ", f=" + std::to_string(kelly_frac_) + ")";
    }

private:
    double win_rate_;
    double avg_win_;
    double avg_loss_;
    double kelly_frac_;
};

class EqualWeight : public PositionSizer {
public:
    explicit EqualWeight(int n) : n_positions_(n) {
        if (n <= 0) throw std::invalid_argument("Number of positions must be positive");
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        double shares = (ctx.current_capital / n_positions_ / ctx.current_price)
                        * ctx.signal_strength;
        return apply_constraints(shares, ctx);
    }

    std::string get_name() const override {
        return "EqualWeight(n=" + std::to_string(n_positions_) + ")";
    }

private:
    int n_positions_;
};

class VolatilityTargeting : public PositionSizer {
public:
    // target_vol is the annualised portfolio volatility to target as a fraction
    // (e.g. 0.15 = 15%). Values above 1.0 imply leverage; the sizer scales
    // uncapped by default so that the engine's RiskLimits remain the sole ceiling.
    // Call set_max_leverage() to impose an additional sizer-level cap.
    explicit VolatilityTargeting(double target_vol = 0.15) : target_vol_(target_vol) {
        if (target_vol <= 0.0)
            throw std::invalid_argument("Target volatility must be positive");
        max_leverage_ = 0.0;
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        if (ctx.portfolio_volatility <= 0.0) return 0.0;
        double lev    = target_vol_ / ctx.portfolio_volatility;
        double shares = (ctx.current_capital * lev / ctx.current_price)
                        * ctx.signal_strength;
        return apply_constraints(shares, ctx);
    }

    std::string get_name() const override {
        return "VolatilityTargeting(" + std::to_string(target_vol_ * 100) + "%)";
    }

private:
    double target_vol_;
};

class FixedShares : public PositionSizer {
public:
    explicit FixedShares(double n_shares) : n_shares_(n_shares) {
        if (n_shares <= 0.0) throw std::invalid_argument("Number of shares must be positive");
        max_leverage_ = 0.0;
    }

    double calculate_size(const PositionSizingContext& ctx) override {
        return apply_constraints(n_shares_ * ctx.signal_strength, ctx);
    }

    std::string get_name() const override {
        return "FixedShares(" + std::to_string(n_shares_) + ")";
    }

private:
    double n_shares_;
};

using PositionSizerPtr = std::shared_ptr<PositionSizer>;

}