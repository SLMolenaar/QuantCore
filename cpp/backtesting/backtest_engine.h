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
#include <unordered_map>
#include <memory_resource>
#include <utility>
#include <vector>
#include <stdexcept>
#include <deque>
#include <numeric>
#include <cmath>
#include <iostream>

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
        , default_volatility_(0.02)
        , default_stop_distance_(0.05)
        , volatility_lookback_(20)
        , bars_per_year_(252)
        , first_bar_timestamp_(0)
    {
        if (initial_capital <= 0.0) {
            throw std::invalid_argument("Initial capital must be positive");
        }
        risk_mgr_->set_capital(initial_capital, initial_capital);
    }

    BacktestEngine(double initial_capital, ExecutionConfig exec_config)
        : BacktestEngine(initial_capital)
    {
        exec_config_ = exec_config;
    }

    void add_data(const std::string& symbol, const BarSeries& bars) {
        if (bars.empty()) {
            throw std::invalid_argument("Cannot add empty bar series");
        }
        data_[symbol] = bars;
        engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
        mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
        price_history_[symbol] = std::deque<double>();
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

    PositionSizerPtr get_position_sizer() const { return sizer_; }

    void set_risk_limits(const RiskLimits& limits) { risk_mgr_->set_limits(limits); }
    const RiskLimits& get_risk_limits()       const { return risk_mgr_->get_limits(); }
    std::shared_ptr<RiskManager> get_risk_manager() const { return risk_mgr_; }

    void configure_market_maker(int levels, double spread, Quantity depth) {
        mm_levels_ = levels;
        mm_spread_ = spread;
        mm_depth_  = depth;
    }

    void set_volatility_params(double default_vol, double stop_distance, size_t lookback) {
        if (default_vol   <= 0.0 || default_vol   > 1.0) throw std::invalid_argument("Default volatility must be between 0 and 1");
        if (stop_distance <= 0.0 || stop_distance > 1.0) throw std::invalid_argument("Stop distance must be between 0 and 1");
        default_volatility_    = default_vol;
        default_stop_distance_ = stop_distance;
        volatility_lookback_   = lookback;
    }

    void set_bars_per_year(size_t bars_per_year) {
        if (bars_per_year == 0) throw std::invalid_argument("Bars per year must be positive");
        bars_per_year_ = bars_per_year;
    }

    size_t get_bars_per_year() const { return bars_per_year_; }

    double run() {
        if (!strat_) throw std::runtime_error("No strategy set");
        if (data_.empty()) throw std::runtime_error("No market data loaded");

        eq_.clear();
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

        // Re-initialize engines and market makers to reset accumulated state between runs.
        for (const auto& [symbol, bars] : data_) {
            engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
            mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
            price_history_[symbol] = std::deque<double>();
        }

        first_bar_timestamp_ = std::numeric_limits<int64_t>::max();
        for (const auto& [symbol, bars] : data_) {
            if (!bars.empty() && bars.front().timestamp_ns < first_bar_timestamp_) {
                first_bar_timestamp_ = bars.front().timestamp_ns;
            }
        }
        if (first_bar_timestamp_ == std::numeric_limits<int64_t>::max()) {
            first_bar_timestamp_ = 0;
        }

        load_data();

        equity_.push_back(init_cap_);
        timestamps_.push_back(first_bar_timestamp_);

        while (!eq_.empty()) {
            auto event = eq_.pop();

            switch (event->get_type()) {
                case EventType::MARKET_DATA: handle_md(event);   break;
                case EventType::SIGNAL:      handle_sig(event);  break;
                case EventType::ORDER:       handle_ord(event);  break;
                case EventType::FILL:        handle_fill(event); break;
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
        for (const auto& [symbol, engine] : engines_) total += engine->get_total_pnl();
        return total;
    }

    double get_total_fees() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : engines_) total += engine->get_total_fees();
        return total;
    }

    std::shared_ptr<ExecutionEngine> get_execution_engine(const std::string& symbol) const {
        auto it = engines_.find(symbol);
        return it != engines_.end() ? it->second : nullptr;
    }

    std::shared_ptr<PortfolioContext> get_portfolio_context() const { return portfolio_; }

    std::vector<double>  get_equity_curve() const { return equity_; }
    std::vector<int64_t> get_timestamps()   const { return timestamps_; }

private:
    // Declared first → destroyed last. All objects allocated from these pools
    // (Orders held in Orderbooks, events in eq_) must be destroyed before the
    // pools themselves, which is guaranteed by this declaration order.
    std::pmr::unsynchronized_pool_resource order_pool_;
    std::pmr::unsynchronized_pool_resource event_pool_;

    EventQueue                eq_;
    std::shared_ptr<Strategy> strat_;

    std::unordered_map<std::string, BarSeries>                         data_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionEngine>>  engines_;
    std::unordered_map<std::string, std::shared_ptr<MarketMaker>>      mms_;
    std::unordered_map<std::string, double>                            last_px_;
    std::unordered_map<std::string, std::deque<double>>                price_history_;

    double   init_cap_;
    double   curr_cap_;
    uint64_t next_oid_;

    PositionSizerPtr              sizer_;
    std::shared_ptr<RiskManager>  risk_mgr_;
    std::shared_ptr<PortfolioContext> portfolio_;

    std::vector<double>  equity_;
    std::vector<int64_t> timestamps_;

    int      mm_levels_;
    double   mm_spread_;
    Quantity mm_depth_;

    double          default_volatility_;
    double          default_stop_distance_;
    size_t          volatility_lookback_;
    size_t          bars_per_year_;
    int64_t         first_bar_timestamp_;
    ExecutionConfig exec_config_;

    static constexpr size_t PRICE_HISTORY_BUFFER = 10;

    template<typename T, typename... Args>
    std::shared_ptr<T> make_event(Args&&... args) {
        std::pmr::polymorphic_allocator<T> alloc{&event_pool_};
        return std::allocate_shared<T>(alloc, std::forward<Args>(args)...);
    }

    void load_data() {
        for (const auto& [symbol, bars] : data_) {
            for (const auto& bar : bars) {
                auto event = make_event<MarketDataEvent>(
                    symbol, bar.timestamp_ns,
                    bar.open, bar.high, bar.low, bar.close, bar.volume
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

        double total_equity = init_cap_ + get_total_pnl();
        curr_cap_ = total_equity;

        double total_position_value = 0.0;
        for (const auto& [symbol, engine] : engines_) {
            double pos = engine->get_position();
            auto px_it = last_px_.find(symbol);
            if (px_it != last_px_.end() && pos != 0.0) {
                total_position_value += std::abs(pos * px_it->second);
            }
        }
        double liquid_cash = total_equity - total_position_value;
        portfolio_->set_cash(liquid_cash);
    }

    double calculate_volatility(const std::string& symbol) const {
        auto it = price_history_.find(symbol);
        if (it == price_history_.end() || it->second.size() < 2) return default_volatility_;

        const auto& prices = it->second;
        size_t n = std::min(prices.size(), volatility_lookback_);
        if (n < 2) return default_volatility_;

        std::vector<double> returns;
        returns.reserve(n - 1);

        size_t start_idx = prices.size() - n;
        for (size_t i = start_idx + 1; i < prices.size(); ++i) {
            if (prices[i-1] <= 0.0 || prices[i] <= 0.0) continue;
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }

        if (returns.empty()) return default_volatility_;

        double mean   = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double sq_sum = 0.0;
        for (double r : returns) sq_sum += (r - mean) * (r - mean);
        double vol = std::sqrt(sq_sum / returns.size());

        vol *= std::sqrt(static_cast<double>(bars_per_year_));
        return std::max(0.001, std::min(1.0, vol));
    }

    void handle_md(EventPtr event) {
        auto md     = std::static_pointer_cast<MarketDataEvent>(event);
        auto symbol = md->get_symbol();
        double price = md->get_close();
        last_px_[symbol] = price;

        auto& hist = price_history_[symbol];
        hist.push_back(price);
        if (hist.size() > volatility_lookback_ + PRICE_HISTORY_BUFFER) hist.pop_front();

        auto ee_it = engines_.find(symbol);
        if (ee_it != engines_.end()) {
            auto mm_it = mms_.find(symbol);
            if (mm_it != mms_.end()) {
                mm_it->second->update_quotes(*ee_it->second, *md);
            }
        }

        strat_->on_data(*md);
        for (const auto& sig : strat_->get_signals()) eq_.push(sig);
    }

    void handle_sig(EventPtr event) {
        auto sig = std::static_pointer_cast<SignalEvent>(event);

        if (sig->get_signal_type() == SignalType::HOLD) return;

        auto px_it = last_px_.find(sig->get_symbol());
        if (px_it == last_px_.end()) return;

        double curr_px = px_it->second;
        if (curr_px <= 0.0) return;

        double strength = sig->get_strength();
        if (strength < 0.01) return;
        if (strength > 2.0) {
            std::cerr << "WARNING: Very high signal strength (" << strength
                      << ") from " << sig->get_strategy_id() << "\n";
        }

        auto ee_it = engines_.find(sig->get_symbol());
        if (ee_it == engines_.end()) return;

        double curr_pos = ee_it->second->get_position();

        double portfolio_val = curr_cap_;
        double portfolio_vol = calculate_volatility(sig->get_symbol());

        double target_pos = 0.0;

        if (sig->get_signal_type() == SignalType::BUY) {
            PositionSizingContext ctx(
                strength, portfolio_val, curr_px,
                curr_pos,
                portfolio_vol, default_stop_distance_
            );
            target_pos = sizer_->calculate_size(ctx);
        } else if (sig->get_signal_type() == SignalType::SELL) {
            PositionSizingContext ctx(
                strength, portfolio_val, curr_px,
                curr_pos,
                portfolio_vol, default_stop_distance_
            );
            target_pos = -sizer_->calculate_size(ctx);
        } else {
            return;
        }

        double delta = target_pos - curr_pos;
        if (std::abs(delta) < 1.0) return;

        Side   ord_side = (delta > 0.0) ? Side::Buy : Side::Sell;
        double ord_qty  = std::abs(delta);

        double spread  = mm_spread_;
        auto   mm_it   = mms_.find(sig->get_symbol());
        if (mm_it != mms_.end()) spread = mm_it->second->get_spread();

        double ord_px = (ord_side == Side::Buy)
            ? curr_px * (1.0 + spread / 2.0)
            : curr_px * (1.0 - spread / 2.0);

        auto risk_check = risk_mgr_->check_order(sig->get_symbol(), ord_side, ord_qty, ord_px);
        if (!risk_check.is_approved()) return;

        int64_t latency = ee_it->second->get_latency_ns();
        auto ord = make_event<OrderEvent>(
            sig->get_symbol(),
            sig->get_timestamp() + latency,
            ord_side, OrderType::GoodTillCancel,
            ord_qty, ord_px
        );
        ord->set_order_id(next_oid_++);
        eq_.push(ord);
    }

    void handle_ord(EventPtr event) {
        auto ord = std::static_pointer_cast<OrderEvent>(event);
        auto it  = engines_.find(ord->get_symbol());
        if (it == engines_.end()) {
            throw std::runtime_error("No execution engine for symbol: " + ord->get_symbol());
        }

        auto engine = it->second;
        if (ord->is_cancel()) {
            engine->cancel_order(ord->get_order_id());
            return;
        }

        Price    px_cents = static_cast<Price>(ord->get_price() * 100.0);
        Quantity qty      = static_cast<Quantity>(ord->get_quantity());

        auto order = std::make_shared<Order>(
            ord->get_order_type(), ord->get_order_id(),
            ord->get_side(), px_cents, qty
        );

        const double slippage_pct = engine->get_slippage_pct();
        auto trades = engine->execute_order(order);
        for (const auto& trade : trades) {
            const TradeInfo& our_trade = (ord->get_side() == Side::Buy)
                ? trade.GetBidTrade()
                : trade.GetAskTrade();

            double raw_price = our_trade.price_ / 100.0;
            double fill_price = (ord->get_side() == Side::Buy)
                ? raw_price * (1.0 + slippage_pct)
                : raw_price * (1.0 - slippage_pct);

            auto fill = make_event<FillEvent>(
                ord->get_symbol(), ord->get_timestamp(),
                ord->get_order_id(), ord->get_side(),
                our_trade.quantity_,
                fill_price,
                0.0
            );
            eq_.push(fill);
        }
    }

    void handle_fill(EventPtr event) {
        auto fill   = std::static_pointer_cast<FillEvent>(event);
        auto ee_it  = engines_.find(fill->get_symbol());
        if (ee_it != engines_.end()) {
            double new_pos = ee_it->second->get_position();
            strat_->set_position(fill->get_symbol(), new_pos);
            risk_mgr_->set_position(fill->get_symbol(), new_pos, fill->get_price());
        }
        strat_->on_fill(*fill);
    }

    double calc_portfolio_val() const { return init_cap_ + get_total_pnl(); }
};

}