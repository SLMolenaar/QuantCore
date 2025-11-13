#pragma once

#include "event_queue.h"
#include "market_data_event.h"
#include "signal_event.h"
#include "order_event.h"
#include "fill_event.h"
#include "strategy.h"
#include "bar_data.h"
#include "market_maker.h"
#include "position_sizer.h"
#include "risk_manager.h"
#include "portfolio_context.h"
#include "../Execution.h"
#include <memory>
#include <map>
#include <vector>
#include <stdexcept>

namespace quantcore {

class BacktestEngine {
public:
    BacktestEngine(double initial_capital = 100000.0)
        : init_cap_(initial_capital)
        , curr_cap_(initial_capital)
        , next_oid_(1)
        , sizer_(std::make_shared<FixedPercentage>(0.1))
        , risk_mgr_(std::make_shared<RiskManager>())
        , portfolio_(std::make_shared<PortfolioContext>(initial_capital))
        , mm_levels_(5)
        , mm_spread_(0.0001)
        , mm_depth_(100000)
    {
        if (initial_capital <= 0.0) {
            throw std::invalid_argument("Initial capital must be positive");
        }
        risk_mgr_->set_capital(initial_capital, initial_capital);
    }

    void add_data(const std::string& symbol, const BarSeries& bars) {
        if (bars.empty()) {
            throw std::invalid_argument("Cannot add empty bar series");
        }
        data_[symbol] = bars;
    }

    void set_strategy(std::shared_ptr<Strategy> strat) {
        if (!strat) {
            throw std::invalid_argument("Strategy cannot be null");
        }
        strat_ = strat;
        strat_->portfolio_ = portfolio_.get();
    }

    void set_position_sizer(PositionSizerPtr sizer) {
        if (!sizer) {
            throw std::invalid_argument("Position sizer cannot be null");
        }
        sizer_ = sizer;
    }

    PositionSizerPtr get_position_sizer() const {
        return sizer_;
    }

    void set_risk_limits(const RiskLimits& limits) {
        risk_mgr_->set_limits(limits);
    }

    const RiskLimits& get_risk_limits() const {
        return risk_mgr_->get_limits();
    }

    std::shared_ptr<RiskManager> get_risk_manager() const {
        return risk_mgr_;
    }

    void configure_market_maker(int levels, double spread, Quantity depth) {
        mm_levels_ = levels;
        mm_spread_ = spread;
        mm_depth_ = depth;
    }

    double run() {
        if (!strat_) {
            throw std::runtime_error("No strategy set");
        }
        if (data_.empty()) {
            throw std::runtime_error("No market data loaded");
        }

        eq_.clear();
        engines_.clear();
        mms_.clear();
        last_px_.clear();
        curr_cap_ = init_cap_;
        next_oid_ = 1;
        strat_->reset();
        risk_mgr_->reset();
        risk_mgr_->set_capital(init_cap_, init_cap_);

        portfolio_ = std::make_shared<PortfolioContext>(init_cap_);
        strat_->portfolio_ = portfolio_.get();

        equity_.clear();
        timestamps_.clear();

        for (const auto& [symbol, bars] : data_) {
            engines_[symbol] = std::make_shared<ExecutionEngine>(symbol);
            mms_[symbol] = std::make_shared<MarketMaker>(
                mm_spread_, mm_levels_, mm_depth_
            );
        }

        load_data();

        equity_.push_back(init_cap_);
        timestamps_.push_back(0);

        while (!eq_.empty()) {
            auto event = eq_.pop();

            switch (event->get_type()) {
                case EventType::MARKET_DATA:
                    handle_md(event);
                    break;
                case EventType::SIGNAL:
                    handle_sig(event);
                    break;
                case EventType::ORDER:
                    handle_ord(event);
                    break;
                case EventType::FILL:
                    handle_fill(event);
                    break;
            }

            if (event->get_type() == EventType::MARKET_DATA) {
                update_portfolio();
                double port_val = calc_portfolio_val();
                equity_.push_back(port_val);
                timestamps_.push_back(event->get_timestamp());
                risk_mgr_->set_capital(init_cap_, port_val);
            }
        }

        return calc_portfolio_val();
    }

    double get_total_pnl() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : engines_) {
            total += engine->get_total_pnl();
        }
        return total;
    }

    double get_total_fees() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : engines_) {
            total += engine->get_total_fees();
        }
        return total;
    }

    std::shared_ptr<ExecutionEngine> get_execution_engine(const std::string& symbol) const {
        auto it = engines_.find(symbol);
        return it != engines_.end() ? it->second : nullptr;
    }

    std::shared_ptr<PortfolioContext> get_portfolio_context() const {
        return portfolio_;
    }

    std::vector<double> get_equity_curve() const { return equity_; }
    std::vector<int64_t> get_timestamps() const { return timestamps_; }

private:
    EventQueue eq_;
    std::shared_ptr<Strategy> strat_;
    std::map<std::string, BarSeries> data_;
    std::map<std::string, std::shared_ptr<ExecutionEngine>> engines_;
    std::map<std::string, std::shared_ptr<MarketMaker>> mms_;
    std::map<std::string, double> last_px_;

    double init_cap_;
    double curr_cap_;
    uint64_t next_oid_;

    PositionSizerPtr sizer_;
    std::shared_ptr<RiskManager> risk_mgr_;
    std::shared_ptr<PortfolioContext> portfolio_;

    std::vector<double> equity_;
    std::vector<int64_t> timestamps_;

    int mm_levels_;
    double mm_spread_;
    Quantity mm_depth_;

    void load_data() {
        for (const auto& [symbol, bars] : data_) {
            for (const auto& bar : bars) {
                auto event = std::make_shared<MarketDataEvent>(
                    symbol, bar.timestamp_ns, bar.open, bar.high,
                    bar.low, bar.close, bar.volume
                );
                eq_.push(event);
            }
        }
    }

    void update_portfolio() {
        for (const auto& [symbol, price] : last_px_) {
            portfolio_->update_price(symbol, price);
        }

        for (const auto& [symbol, engine] : engines_) {
            portfolio_->update_position(symbol, engine->get_position());
        }

        double total_pnl = get_total_pnl();
        portfolio_->set_cash(curr_cap_ + total_pnl);
    }

    void handle_md(EventPtr event) {
        auto md = std::static_pointer_cast<MarketDataEvent>(event);
        std::string symbol = md->get_symbol();
        last_px_[symbol] = md->get_close();

        auto ee_it = engines_.find(symbol);
        if (ee_it != engines_.end()) {
            auto mm_it = mms_.find(symbol);
            if (mm_it != mms_.end()) {
                mm_it->second->update_quotes(*ee_it->second, *md);
            }
        }

        strat_->on_data(*md);

        auto signals = strat_->get_signals();
        for (const auto& sig : signals) {
            eq_.push(sig);
        }
    }

    void handle_sig(EventPtr event) {
        auto sig = std::static_pointer_cast<SignalEvent>(event);
        auto px_it = last_px_.find(sig->get_symbol());
        if (px_it == last_px_.end()) return;

        double curr_px = px_it->second;
        if (curr_px == 0.0) return;

        auto ee_it = engines_.find(sig->get_symbol());
        if (ee_it == engines_.end()) return;

        double curr_pos = ee_it->second->get_position();

        PositionSizingContext ctx(
            sig->get_strength(), curr_cap_, curr_px,
            curr_pos, 0.02, 0.05  // TODO: calc actual vol
        );
        double target = sizer_->calculate_size(ctx);

        double spread = mm_spread_;
        auto mm_it = mms_.find(sig->get_symbol());
        if (mm_it != mms_.end()) {
            spread = mm_it->second->get_spread();
        }

        double ord_px;
        Side ord_side;
        double delta;

        if (sig->get_signal_type() == SignalType::BUY) {
            ord_px = curr_px * (1.0 + spread / 2.0);
            delta = target - curr_pos;
            ord_side = Side::Buy;
        } else if (sig->get_signal_type() == SignalType::SELL) {
            ord_px = curr_px * (1.0 - spread / 2.0);
            delta = curr_pos - target;
            ord_side = Side::Sell;
        } else {
            return;
        }

        if (std::abs(delta) < 1.0) return;

        auto risk_check = risk_mgr_->check_order(
            sig->get_symbol(),
            ord_side,
            std::abs(delta),
            ord_px
        );

        if (!risk_check.is_approved()) {
            return;
        }

        auto ord = std::make_shared<OrderEvent>(
            sig->get_symbol(), sig->get_timestamp(),
            ord_side, OrderType::GoodTillCancel,
            std::abs(delta), ord_px
        );
        ord->set_order_id(next_oid_++);
        eq_.push(ord);
    }

    void handle_ord(EventPtr event) {
        auto ord = std::static_pointer_cast<OrderEvent>(event);
        auto it = engines_.find(ord->get_symbol());
        if (it == engines_.end()) {
            throw std::runtime_error("No execution engine for symbol: " + ord->get_symbol());
        }

        auto engine = it->second;
        if (ord->is_cancel()) {
            engine->cancel_order(ord->get_order_id());
            return;
        }

        Price px_cents = static_cast<Price>(ord->get_price() * 100.0);
        Quantity qty = static_cast<Quantity>(ord->get_quantity());

        auto order = std::make_shared<Order>(
            ord->get_order_type(), ord->get_order_id(),
            ord->get_side(), px_cents, qty
        );

        auto trades = engine->execute_order(order);

        for (const auto& trade : trades) {
            auto fill = std::make_shared<FillEvent>(
                ord->get_symbol(), ord->get_timestamp(),
                ord->get_order_id(), ord->get_side(),
                trade.GetBidTrade().quantity_, trade.GetBidTrade().price_ / 100.0, 0.0
            );
            eq_.push(fill);
        }
    }

    void handle_fill(EventPtr event) {
        auto fill = std::static_pointer_cast<FillEvent>(event);
        auto ee_it = engines_.find(fill->get_symbol());
        if (ee_it != engines_.end()) {
            double new_pos = ee_it->second->get_position();
            strat_->set_position(fill->get_symbol(), new_pos);
            risk_mgr_->set_position(fill->get_symbol(), new_pos);
        }

        strat_->on_fill(*fill);
    }

    double calc_portfolio_val() const {
        return curr_cap_ + get_total_pnl();
    }
};

}