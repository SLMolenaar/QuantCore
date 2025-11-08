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
        double strength = 1.0,
        double capital = 100000.0,
        double price = 100.0,
        double position = 0.0,
        double volatility = 0.02,
        double stop_distance = 0.05
    )
        : signal_strength(strength)
        , current_capital(capital)
        , current_price(price)
        , current_position(position)
        , portfolio_volatility(volatility)
        , stop_loss_distance(stop_distance)
    {
    }
};

class PositionSizer {
public:
    virtual ~PositionSizer() = default;
    
    virtual double calculate_size(const PositionSizingContext& context) = 0;
    
    virtual std::string get_name() const = 0;
    
    void set_max_position_size(double max_size) {
        max_position_size_ = max_size;
    }
    
    void set_min_position_size(double min_size) {
        min_position_size_ = min_size;
    }
    
    void set_max_leverage(double max_leverage) {
        max_leverage_ = max_leverage;
    }
    
protected:
    double apply_constraints(double raw_size, const PositionSizingContext& context) const {
        double size = raw_size;
        
        if (min_position_size_ > 0 && std::abs(size) < min_position_size_) {
            return 0.0;
        }
        
        if (max_position_size_ > 0 && std::abs(size) > max_position_size_) {
            size = std::copysign(max_position_size_, size);
        }
        
        if (max_leverage_ > 0) {
            double max_notional = context.current_capital * max_leverage_;
            double max_shares = max_notional / context.current_price;
            if (std::abs(size) > max_shares) {
                size = std::copysign(max_shares, size);
            }
        }
        
        return size;
    }
    
    double max_position_size_ = 0.0;
    double min_position_size_ = 0.0;
    double max_leverage_ = 1.0;
};

class FixedPercentage : public PositionSizer {
public:
    FixedPercentage(double percentage = 0.1)
        : percentage_(percentage)
    {
        if (percentage <= 0.0 || percentage > 1.0) {
            throw std::invalid_argument("Percentage must be between 0 and 1");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        double allocation = context.current_capital * percentage_;
        
        double shares = allocation / context.current_price;
        
        shares *= context.signal_strength;
        
        return apply_constraints(shares, context);
    }
    
    std::string get_name() const override {
        return "FixedPercentage(" + std::to_string(percentage_ * 100) + "%)";
    }
    
private:
    double percentage_;
};

class RiskBased : public PositionSizer {
public:
    RiskBased(double risk_per_trade = 0.01)
        : risk_per_trade_(risk_per_trade)
    {
        if (risk_per_trade <= 0.0 || risk_per_trade > 0.1) {
            throw std::invalid_argument("Risk per trade must be between 0 and 0.1");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        if (context.stop_loss_distance <= 0.0) {
            return 0.0;
        }
        
        double risk_amount = context.current_capital * risk_per_trade_;
        
        double shares = risk_amount / (context.current_price * context.stop_loss_distance);
        
        shares *= context.signal_strength;
        
        return apply_constraints(shares, context);
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
        : win_rate_(win_rate)
        , avg_win_(avg_win)
        , avg_loss_(avg_loss)
        , kelly_fraction_(fraction)
    {
        if (win_rate <= 0.0 || win_rate >= 1.0) {
            throw std::invalid_argument("Win rate must be between 0 and 1");
        }
        if (avg_win <= 0.0 || avg_loss <= 0.0) {
            throw std::invalid_argument("Average win and loss must be positive");
        }
        if (kelly_fraction <= 0.0 || kelly_fraction > 1.0) {
            throw std::invalid_argument("Kelly fraction must be between 0 and 1");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        double win_loss_ratio = avg_win_ / avg_loss_;
        
        double kelly_pct = (win_rate_ * win_loss_ratio - (1.0 - win_rate_)) / win_loss_ratio;
        
        kelly_pct = std::max(0.0, kelly_pct);
        
        kelly_pct *= kelly_fraction_;
        
        double allocation = context.current_capital * kelly_pct;
        double shares = allocation / context.current_price;
        
        shares *= context.signal_strength;
        
        return apply_constraints(shares, context);
    }
    
    std::string get_name() const override {
        return "KellyCriterion(wr=" + std::to_string(win_rate_) + 
               ", f=" + std::to_string(kelly_fraction_) + ")";
    }
    
private:
    double win_rate_;
    double avg_win_;
    double avg_loss_;
    double kelly_fraction_;
};

class EqualWeight : public PositionSizer {
public:
    EqualWeight(int num_positions)
        : num_positions_(num_positions)
    {
        if (num_positions <= 0) {
            throw std::invalid_argument("Number of positions must be positive");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        double allocation_per_position = context.current_capital / num_positions_;
        
        double shares = allocation_per_position / context.current_price;
        
        shares *= context.signal_strength;
        
        return apply_constraints(shares, context);
    }
    
    std::string get_name() const override {
        return "EqualWeight(n=" + std::to_string(num_positions_) + ")";
    }
    
private:
    int num_positions_;
};

class VolatilityTargeting : public PositionSizer {
public:
    VolatilityTargeting(double target_volatility = 0.15)
        : target_volatility_(target_volatility)
    {
        if (target_volatility <= 0.0 || target_volatility > 1.0) {
            throw std::invalid_argument("Target volatility must be between 0 and 1");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        if (context.portfolio_volatility <= 0.0) {
            return 0.0;
        }
        
        double leverage = target_volatility_ / context.portfolio_volatility;
        
        leverage = std::min(leverage, max_leverage_);
        
        double allocation = context.current_capital * leverage;
        double shares = allocation / context.current_price;
        
        shares *= context.signal_strength;
        
        return apply_constraints(shares, context);
    }
    
    std::string get_name() const override {
        return "VolatilityTargeting(" + std::to_string(target_volatility_ * 100) + "%)";
    }
    
private:
    double target_volatility_;
};

class FixedShares : public PositionSizer {
public:
    FixedShares(double num_shares)
        : num_shares_(num_shares)
    {
        if (num_shares <= 0.0) {
            throw std::invalid_argument("Number of shares must be positive");
        }
    }
    
    double calculate_size(const PositionSizingContext& context) override {
        double shares = num_shares_ * context.signal_strength;
        
        return apply_constraints(shares, context);
    }
    
    std::string get_name() const override {
        return "FixedShares(" + std::to_string(num_shares_) + ")";
    }
    
private:
    double num_shares_;
};

using PositionSizerPtr = std::shared_ptr<PositionSizer>;

}