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

    void set_volatility_params(double default_vol, double stop_distance, size_t lookback) {
        if (default_vol <= 0.0 || default_vol > 1.0) {
            throw std::invalid_argument("Default volatility must be between 0 and 1");
        }
        if (stop_distance <= 0.0 || stop_distance > 1.0) {
            throw std::invalid_argument("Stop distance must be between 0 and 1");
        }
        default_volatility_ = default_vol;
        default_stop_distance_ = stop_distance;
        volatility_lookback_ = lookback;
    }

    // set number of bars per year for annualizing volatility
    void set_bars_per_year(size_t bars_per_year) {
        if (bars_per_year == 0) {
            throw std::invalid_argument("Bars per year must be positive");
        }
        bars_per_year_ = bars_per_year;
    }

    size_t get_bars_per_year() const {
        return bars_per_year_;
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
        price_history_.clear();
        curr_cap_ = init_cap_;
        next_oid_ = 1;
        strat_->reset();
        risk_mgr_->reset();
        risk_mgr_->set_capital(init_cap_, init_cap_);

        portfolio_ = std::make_shared<PortfolioContext>(init_cap_);
        strat_->portfolio_ = portfolio_.get();

        equity_.clear();
        timestamps_.clear();

        // Set up engine, market maker, and empty price history for each symbol
        for (const auto& [symbol, bars] : data_) {
            engines_[symbol] = std::make_shared<ExecutionEngine>(symbol);
            mms_[symbol] = std::make_shared<MarketMaker>(
                mm_spread_, mm_levels_, mm_depth_
            );
            price_history_[symbol] = std::deque<double>();
        }

        load_data();

        equity_.push_back(init_cap_);
        timestamps_.push_back(0);

        // main loop
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

            // update portfolio every time new data is processed
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

    // unordered_map: O(1) symbol lookups vs O(log n) for std::map
    std::unordered_map<std::string, BarSeries> data_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionEngine>> engines_;
    std::unordered_map<std::string, std::shared_ptr<MarketMaker>> mms_;
    std::unordered_map<std::string, double> last_px_;
    std::unordered_map<std::string, std::deque<double>> price_history_;

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

    double default_volatility_;
    double default_stop_distance_;
    size_t volatility_lookback_;
    size_t bars_per_year_;

    // Pool allocator for events. Avoids repeated heap allocation/deallocation
    // in the hot loop. Events are short-lived so the pool free-list is reused
    // continuously throughout a run.
    std::pmr::unsynchronized_pool_resource event_pool_;

    // Buffer size to avoid frequent reallocation of price history deque
    static constexpr size_t PRICE_HISTORY_BUFFER = 10;

    // Allocate an event from the pool
    template<typename T, typename... Args>
    std::shared_ptr<T> make_event(Args&&... args) {
        std::pmr::polymorphic_allocator<T> alloc{&event_pool_};
        return std::allocate_shared<T>(alloc, std::forward<Args>(args)...);
    }

    void load_data() {
        for (const auto& [symbol, bars] : data_) {
            for (const auto& bar : bars) {
                auto event = make_event<MarketDataEvent>(
                    symbol, bar.timestamp_ns, bar.open, bar.high,
                    bar.low, bar.close, bar.volume
                );
                eq_.push(event); // add event to event queue
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

    double calculate_volatility(const std::string& symbol) const {
        auto it = price_history_.find(symbol);
        if (it == price_history_.end() || it->second.size() < 2) {
            return default_volatility_;
        }

        const auto& prices = it->second;
        if (prices.size() < 2) {
            return default_volatility_;
        }

        size_t n = std::min(prices.size(), volatility_lookback_);

        // Need at least 2 prices to calculate returns
        if (n < 2) return default_volatility_;

        // Calculate returns using the most recent n prices
        std::vector<double> returns;
        returns.reserve(n - 1);

        size_t start_idx = prices.size() - n;
        for (size_t i = start_idx + 1; i < prices.size(); ++i) {
            if (prices[i-1] <= 0.0 || prices[i] <= 0.0) {
                continue; // Skip invalid prices
            }
            double ret = (prices[i] - prices[i-1]) / prices[i-1];
            returns.push_back(ret);
        }

        if (returns.empty()) {
            return default_volatility_;
        }

        // Calculate standard deviation
        double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double sq_sum = 0.0;
        for (double ret : returns) {
            sq_sum += (ret - mean) * (ret - mean);
        }
        double variance = sq_sum / returns.size();
        double vol = std::sqrt(variance);

        // Annualize using configured bars per year
        vol *= std::sqrt(static_cast<double>(bars_per_year_));

        // Clamp to reasonable range [0.1% to 100%]
        return std::max(0.001, std::min(1.0, vol));
    }

    void handle_md(EventPtr event) {
        auto md = std::static_pointer_cast<MarketDataEvent>(event);
        std::string symbol = md->get_symbol();
        double price = md->get_close();
        last_px_[symbol] = price;

        // Update price history for volatility calculation
        auto& hist = price_history_[symbol];
        hist.push_back(price);
        // Keep a bit more than needed for lookback
        if (hist.size() > volatility_lookback_ + PRICE_HISTORY_BUFFER) {
            hist.pop_front();
        }

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

        // validate prereqs
        auto px_it = last_px_.find(sig->get_symbol());
        if (px_it == last_px_.end()) {
            return;
        }

        double curr_px = px_it->second;
        if (curr_px <= 0.0) {
            return;
        }

        // val sig strength
        double strength = sig->get_strength();
        if (strength < 0.0) {
            return;
        }
        if (strength > 2.0) {
            std::cerr << "WARNING: Very high signal strength (" << strength
                      << ") from " << sig->get_strategy_id() << "\n";
        }
        if (strength < 0.01) {
            return;
        }

        auto ee_it = engines_.find(sig->get_symbol());
        if (ee_it == engines_.end()) {
            return;
        }

        double curr_pos = ee_it->second->get_position();

        // calc target position based on signal type
        double target_pos = 0.0;

        if (sig->get_signal_type() == SignalType::BUY) {
            double portfolio_vol = calculate_volatility(sig->get_symbol());
            double stop_distance = default_stop_distance_;

            PositionSizingContext ctx(
                strength, curr_cap_, curr_px,
                0.0, portfolio_vol, stop_distance
            );
            target_pos = sizer_->calculate_size(ctx);

        } else if (sig->get_signal_type() == SignalType::SELL) {
            double portfolio_vol = calculate_volatility(sig->get_symbol());
            double stop_distance = default_stop_distance_;

            PositionSizingContext ctx(
                strength, curr_cap_, curr_px,
                0.0, portfolio_vol, stop_distance
            );
            target_pos = -sizer_->calculate_size(ctx);

        } else if (sig->get_signal_type() == SignalType::HOLD) {
            target_pos = 0.0;
        } else {
            return;
        }

        // calc order qtty (delta from current to target)
        double delta = target_pos - curr_pos;

        // must have meaningful position change
        if (std::abs(delta) < 1.0) {
            return;
        }

        // determine side and qtty
        // If delta > 0, we need to buy (increase position)
        // If delta < 0, we need to sell (decrease position)
        Side ord_side = (delta > 0.0) ? Side::Buy : Side::Sell;
        double ord_qty = std::abs(delta);

        // calc order price with bid/ask spread
        double spread = mm_spread_;
        auto mm_it = mms_.find(sig->get_symbol());
        if (mm_it != mms_.end()) {
            spread = mm_it->second->get_spread();
        }

        double ord_px;
        if (ord_side == Side::Buy) {
            // When buying, pay the ask (market price + half spread)
            ord_px = curr_px * (1.0 + spread / 2.0);
        } else {
            // When selling, hit the bid (market price - half spread)
            ord_px = curr_px * (1.0 - spread / 2.0);
        }

        // pretrade risk check
        auto risk_check = risk_mgr_->check_order(
            sig->get_symbol(),
            ord_side,
            ord_qty,
            ord_px
        );

        if (!risk_check.is_approved()) {
            // Risk limit violation - silently reject order
            return;
        }

        // create & submit order
        int64_t latency = ee_it->second->get_latency_ns();

        auto ord = make_event<OrderEvent>(
            sig->get_symbol(),
            sig->get_timestamp() + latency,  // Order arrives after latency delay
            ord_side,
            OrderType::GoodTillCancel,
            ord_qty,
            ord_px
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
            auto fill = make_event<FillEvent>(
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