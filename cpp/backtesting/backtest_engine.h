#pragma once

#include "event_queue.h"
#include "market_data_event.h"
#include "signal_event.h"
#include "order_event.h"
#include "fill_event.h"
#include "strategy.h"
#include "bar_data.h"
#include "tick_data.h"
#include "market_maker.h"
#include "position_sizer.h"
#include "risk_manager.h"
#include "portfolio_context.h"
#include "../Execution.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <memory_resource>
#include <utility>
#include <vector>
#include <stdexcept>
#include <deque>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace quantcore {

class BacktestEngine {
public:
    explicit BacktestEngine(double initial_capital = 100000.0)
        : init_cap_(initial_capital)
        , curr_cap_(initial_capital)
        , next_oid_(1)
        , halted_(false)
        , sizer_(std::make_shared<FixedPercentage>(0.1))
        , risk_mgr_(std::make_shared<RiskManager>())
        , portfolio_(std::make_shared<PortfolioContext>(initial_capital))
        , mm_levels_(5)
        , mm_spread_(0.0001)
        , mm_depth_(100000)
        , mm_refresh_interval_ns_(0)
        , equity_snapshot_interval_ns_(0)
        , last_equity_snapshot_ns_(-1)
        , default_volatility_(0.02)
        , default_stop_distance_(0.05)
        , volatility_lookback_(20)
        , bars_per_year_(252)
        , first_bar_timestamp_(0)
        , last_event_day_(-1)
    {
        if (initial_capital <= 0.0)
            throw std::invalid_argument("Initial capital must be positive");
        risk_mgr_->set_capital(initial_capital, initial_capital);
    }

    BacktestEngine(double initial_capital, ExecutionConfig exec_config)
        : BacktestEngine(initial_capital)
    {
        exec_config_ = exec_config;
    }

    void add_data(const std::string& symbol, const BarSeries& bars) {
        if (bars.empty())
            throw std::invalid_argument("Cannot add empty bar series");
        if (data_.find(symbol) == data_.end()) {
            symbol_order_index_[symbol] = symbol_order_.size();
            symbol_order_.push_back(symbol);
        }
        data_[symbol]          = bars;
        tick_data_.erase(symbol);
        engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
        mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
        price_history_[symbol] = {};
    }

    void add_tick_data(const std::string& symbol, const TickSeries& ticks) {
        if (ticks.empty())
            throw std::invalid_argument("Cannot add empty tick series");
        if (tick_data_.find(symbol) == tick_data_.end()) {
            if (data_.find(symbol) == data_.end()) {
                symbol_order_index_[symbol] = symbol_order_.size();
                symbol_order_.push_back(symbol);
            }
        }
        tick_data_[symbol]     = ticks;
        data_.erase(symbol);
        engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
        mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
        price_history_[symbol] = {};
    }

    void set_strategy(std::shared_ptr<Strategy> strat) {
        if (!strat)
            throw std::invalid_argument("Strategy cannot be null");
        strat_ = strat;
        strat_->portfolio_ = portfolio_.get();
    }

    void set_position_sizer(PositionSizerPtr sizer) {
        if (!sizer)
            throw std::invalid_argument("Position sizer cannot be null");
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

    void set_mm_refresh_interval(int64_t interval_ns) {
        if (interval_ns < 0)
            throw std::invalid_argument("Market maker refresh interval cannot be negative");
        mm_refresh_interval_ns_ = interval_ns;
    }
    int64_t get_mm_refresh_interval() const { return mm_refresh_interval_ns_; }

    void set_equity_snapshot_interval(int64_t interval_ns) {
        if (interval_ns < 0)
            throw std::invalid_argument("Equity snapshot interval cannot be negative");
        equity_snapshot_interval_ns_ = interval_ns;
    }
    int64_t get_equity_snapshot_interval() const { return equity_snapshot_interval_ns_; }

    void set_volatility_params(double default_vol, double stop_distance, size_t lookback) {
        if (default_vol   <= 0.0 || default_vol   > 1.0)
            throw std::invalid_argument("Default volatility must be between 0 and 1");
        if (stop_distance <= 0.0 || stop_distance > 1.0)
            throw std::invalid_argument("Stop distance must be between 0 and 1");
        default_volatility_    = default_vol;
        default_stop_distance_ = stop_distance;
        volatility_lookback_   = lookback;
    }

    void set_bars_per_year(size_t bars_per_year) {
        if (bars_per_year == 0)
            throw std::invalid_argument("Bars per year must be positive");
        bars_per_year_ = bars_per_year;
    }
    size_t get_bars_per_year() const { return bars_per_year_; }

    double run() {
        if (!strat_)
            throw std::runtime_error("No strategy set");
        if (data_.empty() && tick_data_.empty())
            throw std::runtime_error("No market data loaded");

        eq_.clear();
        last_px_.clear();
        curr_cap_                = init_cap_;
        next_oid_                = 1;
        halted_                  = false;
        pending_buys_.clear();
        gfd_orders_.clear();
        pending_strategy_orders_.clear();
        strategy_order_remaining_.clear();
        last_event_day_          = -1;
        last_equity_snapshot_ns_ = -1;
        mm_last_refresh_.clear();

        strat_->reset();
        risk_mgr_->reset();
        risk_mgr_->set_capital(init_cap_, init_cap_);

        portfolio_         = std::make_shared<PortfolioContext>(init_cap_);
        strat_->portfolio_ = portfolio_.get();

        equity_.clear();
        timestamps_.clear();

        for (const auto& [symbol, bars] : data_) {
            engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
            mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
            price_history_[symbol] = {};
        }
        for (const auto& [symbol, ticks] : tick_data_) {
            engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
            mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
            price_history_[symbol] = {};
        }

        first_bar_timestamp_ = std::numeric_limits<int64_t>::max();
        for (const auto& [symbol, bars] : data_) {
            if (!bars.empty() && bars.front().timestamp_ns < first_bar_timestamp_)
                first_bar_timestamp_ = bars.front().timestamp_ns;
        }
        for (const auto& [symbol, ticks] : tick_data_) {
            if (!ticks.empty() && ticks.front().timestamp_ns < first_bar_timestamp_)
                first_bar_timestamp_ = ticks.front().timestamp_ns;
        }
        if (first_bar_timestamp_ == std::numeric_limits<int64_t>::max())
            first_bar_timestamp_ = 0;

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
                maybe_snapshot_equity(event->get_timestamp());

                if (should_halt())
                    flatten_all_positions(event->get_timestamp());
            }
        }

        return curr_cap_;
    }

    double get_total_pnl() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : engines_)
            total += engine->get_total_pnl();
        return total;
    }

    double get_total_fees() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : engines_)
            total += engine->get_total_fees();
        return total;
    }

    std::vector<double> get_trade_pnls() const {
        std::vector<double> pnls;
        for (const auto& [sym, ee] : engines_)
            for (double p : ee->get_closed_trade_pnls())
                pnls.push_back(p);
        return pnls;
    }

    std::shared_ptr<ExecutionEngine> get_execution_engine(const std::string& symbol) const {
        auto it = engines_.find(symbol);
        return it != engines_.end() ? it->second : nullptr;
    }

    std::shared_ptr<PortfolioContext> get_portfolio_context() const { return portfolio_; }

    std::vector<double>  get_equity_curve() const { return equity_; }
    std::vector<int64_t> get_timestamps()   const { return timestamps_; }

    bool has_tick_data(const std::string& symbol) const {
        return tick_data_.find(symbol) != tick_data_.end();
    }

private:
    // Tracks the signal-time price and remaining unfilled quantity for each
    // open buy order. Available cash is reduced by sum(price * remaining_qty)
    // so subsequent signals on the same bar are sized against realistic cash.
    // Decremented on each fill, erased on full fill, cancel, or GFD expiry.
    struct PendingBuy {
        double price_per_share;
        double remaining_qty;
    };

    // Order pool declared before engines so it outlives all orders placed in it.
    std::pmr::unsynchronized_pool_resource order_pool_;
    std::pmr::unsynchronized_pool_resource event_pool_;

    EventQueue                eq_;
    std::shared_ptr<Strategy> strat_;

    std::unordered_map<std::string, BarSeries>                        data_;
    std::unordered_map<std::string, TickSeries>                       tick_data_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionEngine>> engines_;
    std::unordered_map<std::string, std::shared_ptr<MarketMaker>>     mms_;
    std::unordered_map<std::string, double>                           last_px_;
    std::unordered_map<std::string, std::deque<double>>               price_history_;
    std::vector<std::string>                                          symbol_order_;
    std::unordered_map<std::string, size_t>                           symbol_order_index_;
    std::unordered_map<std::string, int64_t>                          mm_last_refresh_;

    double   init_cap_;
    double   curr_cap_;
    uint64_t next_oid_;
    bool     halted_;

    PositionSizerPtr                  sizer_;
    std::shared_ptr<RiskManager>      risk_mgr_;
    std::shared_ptr<PortfolioContext>  portfolio_;

    std::vector<double>  equity_;
    std::vector<int64_t> timestamps_;

    int      mm_levels_;
    double   mm_spread_;
    Quantity mm_depth_;
    int64_t  mm_refresh_interval_ns_;
    int64_t  equity_snapshot_interval_ns_;
    int64_t  last_equity_snapshot_ns_;

    double  default_volatility_;
    double  default_stop_distance_;
    size_t  volatility_lookback_;
    size_t  bars_per_year_;
    int64_t first_bar_timestamp_;

    ExecutionConfig exec_config_;

    std::unordered_map<uint64_t, PendingBuy>                          pending_buys_;
    std::unordered_map<std::string, std::unordered_set<uint64_t>>     gfd_orders_;

    // Wash trade prevention
    // Tracks all live unfilled strategy order IDs per symbol so they can be
    // cancelled before a new signal in the opposite direction is placed.
    // Without this a stale resting buy could self-match against a new sell
    // if price reverses before the original order fills, triggering the wash
    // trade guard in ExecutionEngine::update_position().
    std::unordered_map<std::string, std::unordered_set<uint64_t>> pending_strategy_orders_;
    std::unordered_map<uint64_t, double> strategy_order_remaining_;

    int64_t last_event_day_; // UTC day of last market-data event; -1 before first event

    static constexpr size_t  PRICE_HISTORY_BUFFER = 10;
    static constexpr double  MIN_ORDER_QTY        = 1e-8;
    static constexpr int64_t NS_PER_DAY           = 86'400LL * 1'000'000'000LL;

    static int64_t to_unix_day(int64_t timestamp_ns) noexcept {
        return timestamp_ns / NS_PER_DAY;
    }

    double pending_buy_notional() const {
        double total = 0.0;
        for (const auto& [id, pb] : pending_buys_)
            total += pb.price_per_share * pb.remaining_qty;
        return total;
    }

    void release_pending_buy(uint64_t order_id, double qty_filled) {
        auto it = pending_buys_.find(order_id);
        if (it == pending_buys_.end()) return;
        it->second.remaining_qty -= qty_filled;
        if (it->second.remaining_qty <= MIN_ORDER_QTY)
            pending_buys_.erase(it);
    }

    void cancel_pending_buy(uint64_t order_id) {
        pending_buys_.erase(order_id);
    }

    // Cancel all live strategy orders for a symbol and clear tracking state.
    void cancel_stale_strategy_orders(const std::string& symbol,
                                      ExecutionEngine& engine) {
        auto it = pending_strategy_orders_.find(symbol);
        if (it == pending_strategy_orders_.end() || it->second.empty())
            return;

        for (uint64_t stale_id : it->second) {
            engine.cancel_order(stale_id);
            cancel_pending_buy(stale_id);
            strategy_order_remaining_.erase(stale_id);
        }
        it->second.clear();
    }

    template<typename T, typename... Args>
    std::shared_ptr<T> make_event(Args&&... args) {
        std::pmr::polymorphic_allocator<T> alloc{&event_pool_};
        return std::allocate_shared<T>(alloc, std::forward<Args>(args)...);
    }

    void load_data() {
        for (const auto& [symbol, bars] : data_) {
            for (const auto& bar : bars) {
                eq_.push(make_event<MarketDataEvent>(
                    symbol, bar.timestamp_ns,
                    bar.open, bar.high, bar.low, bar.close, bar.volume
                ));
            }
        }
        for (const auto& [symbol, ticks] : tick_data_) {
            for (const auto& tick : ticks) {
                eq_.push(make_event<MarketDataEvent>(
                    symbol, tick.timestamp_ns, tick.price, tick.quantity
                ));
            }
        }
    }

    void update_portfolio() {
        for (const auto& [symbol, price] : last_px_)
            portfolio_->update_price(symbol, price);
        for (const auto& [symbol, engine] : engines_)
            portfolio_->update_position(symbol, engine->get_position());

        double total_equity = init_cap_;
        for (const auto& [symbol, engine] : engines_) {
            double pos = engine->get_position();
            double avg = engine->get_average_price();
            auto   px  = last_px_.find(symbol);
            double unrealized = (pos != 0.0 && avg != 0.0 && px != last_px_.end())
                                ? pos * (px->second - avg)
                                : 0.0;
            total_equity += engine->get_realized_pnl() + unrealized;
        }
        curr_cap_ = total_equity;

        double total_position_value = 0.0;
        for (const auto& [symbol, engine] : engines_) {
            double pos = engine->get_position();
            auto px_it = last_px_.find(symbol);
            if (px_it != last_px_.end() && pos != 0.0)
                total_position_value += std::abs(pos * px_it->second);
        }
        portfolio_->set_cash(curr_cap_ - total_position_value);
    }

    void maybe_snapshot_equity(int64_t timestamp) {
        if (equity_snapshot_interval_ns_ == 0) {
            // Bar mode: snapshot once per bar, after all symbols at this timestamp.
            bool is_last = true;
            if (!eq_.empty()) {
                auto next = eq_.peek();
                if (next->get_type() == EventType::MARKET_DATA &&
                    next->get_timestamp() == timestamp) {
                    is_last = false;
                }
            }
            if (is_last) {
                equity_.push_back(curr_cap_);
                timestamps_.push_back(timestamp);
                risk_mgr_->set_capital(init_cap_, curr_cap_);
            }
        } else {
            if (last_equity_snapshot_ns_ < 0 ||
                timestamp - last_equity_snapshot_ns_ >= equity_snapshot_interval_ns_) {
                equity_.push_back(curr_cap_);
                timestamps_.push_back(timestamp);
                risk_mgr_->set_capital(init_cap_, curr_cap_);
                last_equity_snapshot_ns_ = timestamp;
            }
        }
    }

    double calculate_volatility(const std::string& symbol) const {
        auto it = price_history_.find(symbol);
        if (it == price_history_.end() || it->second.size() < 2)
            return default_volatility_;

        const auto& prices = it->second;
        size_t n = std::min(prices.size(), volatility_lookback_);
        if (n < 2) return default_volatility_;

        std::vector<double> returns;
        returns.reserve(n - 1);
        size_t start = prices.size() - n;
        for (size_t i = start + 1; i < prices.size(); ++i) {
            if (prices[i-1] <= 0.0 || prices[i] <= 0.0) continue;
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }
        if (returns.empty()) return default_volatility_;

        double mean   = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double sq_sum = 0.0;
        for (double r : returns) sq_sum += (r - mean) * (r - mean);
        double vol = std::sqrt(sq_sum / returns.size())
                   * std::sqrt(static_cast<double>(bars_per_year_));
        return std::max(0.001, std::min(1.0, vol));
    }

    bool should_halt() const {
        if (halted_) return false;
        const auto& limits = risk_mgr_->get_limits();
        if (!limits.enabled || limits.max_loss_pct <= 0.0) return false;
        return (init_cap_ - curr_cap_) / init_cap_ > limits.max_loss_pct;
    }

    void flatten_all_positions(int64_t timestamp) {
        halted_ = true;

        // Cancel all pending strategy orders before placing close orders.
        for (auto& [sym, ids] : pending_strategy_orders_) {
            auto ee_it = engines_.find(sym);
            if (ee_it != engines_.end()) {
                for (uint64_t id : ids) {
                    ee_it->second->cancel_order(id);
                    cancel_pending_buy(id);
                    strategy_order_remaining_.erase(id);
                }
            }
        }
        pending_strategy_orders_.clear();

        for (const auto& [symbol, ee] : engines_) {
            double pos = ee->get_position();
            if (std::abs(pos) < MIN_ORDER_QTY) continue;
            auto px_it = last_px_.find(symbol);
            if (px_it == last_px_.end()) continue;
            Side   side   = (pos > 0) ? Side::Sell : Side::Buy;
            double ord_px = (side == Side::Buy)
                ? px_it->second * (1.0 + mm_spread_)
                : px_it->second * (1.0 - mm_spread_);
            auto ord = make_event<OrderEvent>(
                symbol, timestamp, side, OrderType::GoodTillCancel,
                std::abs(pos), ord_px
            );
            ord->set_order_id(next_oid_++);
            eq_.push(ord);
        }
    }

    bool should_refresh_mm(const std::string& symbol, int64_t timestamp) {
        if (mm_refresh_interval_ns_ == 0) return true;
        auto it = mm_last_refresh_.find(symbol);
        if (it == mm_last_refresh_.end() ||
            timestamp - it->second >= mm_refresh_interval_ns_) {
            mm_last_refresh_[symbol] = timestamp;
            return true;
        }
        return false;
    }

    void handle_md(EventPtr event) {
        auto md     = std::static_pointer_cast<MarketDataEvent>(event);
        auto symbol = md->get_symbol();
        last_px_[symbol] = md->get_close();

        auto& hist = price_history_[symbol];
        hist.push_back(md->get_close());
        if (hist.size() > volatility_lookback_ + PRICE_HISTORY_BUFFER)
            hist.pop_front();

        // Cancel GFD orders at UTC day boundaries. cancel_order() is a no-op
        // for already-filled orders so this is safe to call unconditionally.
        int64_t event_day = to_unix_day(md->get_timestamp());
        if (last_event_day_ >= 0 && event_day > last_event_day_) {
            for (auto& [sym, ids] : gfd_orders_) {
                auto ee_it = engines_.find(sym);
                if (ee_it != engines_.end()) {
                    for (uint64_t id : ids) {
                        ee_it->second->cancel_order(id);
                        cancel_pending_buy(id);
                        // Also remove from strategy order tracking.
                        strategy_order_remaining_.erase(id);
                        pending_strategy_orders_[sym].erase(id);
                    }
                }
            }
            gfd_orders_.clear();
        }
        last_event_day_ = event_day;

        auto ee_it = engines_.find(symbol);
        if (ee_it != engines_.end()) {
            auto mm_it = mms_.find(symbol);
            if (mm_it != mms_.end() && should_refresh_mm(symbol, md->get_timestamp()))
                mm_it->second->update_quotes(*ee_it->second, *md);
        }

        strat_->on_data(*md);

        // Sort by add_data() insertion order for deterministic signal processing.
        auto signals = strat_->get_signals();
        std::sort(signals.begin(), signals.end(),
            [this](const std::shared_ptr<SignalEvent>& a,
                   const std::shared_ptr<SignalEvent>& b) {
                auto ia = symbol_order_index_.find(a->get_symbol());
                auto ib = symbol_order_index_.find(b->get_symbol());
                size_t idx_a = (ia != symbol_order_index_.end()) ? ia->second : symbol_order_.size();
                size_t idx_b = (ib != symbol_order_index_.end()) ? ib->second : symbol_order_.size();
                return idx_a < idx_b;
            });
        for (const auto& sig : signals)
            eq_.push(sig);
    }

    void handle_sig(EventPtr event) {
        if (halted_) return;

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

        // Cancel any stale unfilled strategy orders for this symbol before placing a new one.
        cancel_stale_strategy_orders(sig->get_symbol(), *ee_it->second);

        double curr_pos      = ee_it->second->get_position();
        double portfolio_vol = calculate_volatility(sig->get_symbol());
        double available_cash = std::max(0.0, portfolio_->get_cash() - pending_buy_notional());

        double target_pos = 0.0;

        if (sig->get_signal_type() == SignalType::BUY) {
            if (curr_pos < 0.0) {
                target_pos = 0.0;
            } else {
                PositionSizingContext ctx(strength, available_cash, curr_px, curr_pos,
                                          portfolio_vol, default_stop_distance_);
                target_pos = sizer_->calculate_size(ctx);
            }
        } else if (sig->get_signal_type() == SignalType::SELL) {
            if (curr_pos > 0.0) {
                target_pos = 0.0;
            } else {
                PositionSizingContext ctx(strength, available_cash, curr_px, curr_pos,
                                          portfolio_vol, default_stop_distance_);
                target_pos = -sizer_->calculate_size(ctx);
            }
        } else {
            return;
        }

        double delta = target_pos - curr_pos;
        if (std::abs(delta) < MIN_ORDER_QTY) return;

        Side   ord_side = (delta > 0.0) ? Side::Buy : Side::Sell;
        double ord_qty  = std::abs(delta);

        double spread = mm_spread_;
        auto   mm_it  = mms_.find(sig->get_symbol());
        if (mm_it != mms_.end()) spread = mm_it->second->get_spread();

        double ord_px = (ord_side == Side::Buy)
            ? curr_px * (1.0 + spread / 2.0)
            : curr_px * (1.0 - spread / 2.0);

        auto risk_check = risk_mgr_->check_order(sig->get_symbol(), ord_side, ord_qty, ord_px);
        if (!risk_check.is_approved()) {
            strat_->on_rejected(sig->get_symbol(), risk_check.reason);
            return;
        }

        int64_t latency = ee_it->second->get_latency_ns();
        auto ord = make_event<OrderEvent>(
            sig->get_symbol(),
            sig->get_timestamp() + latency,
            ord_side, OrderType::GoodTillCancel,
            ord_qty, ord_px
        );
        ord->set_order_id(next_oid_++);

        if (ord_side == Side::Buy)
            pending_buys_[ord->get_order_id()] = {curr_px, ord_qty};

        // Track this order so it can be cancelled if a new signal arrives
        // before it fills.
        pending_strategy_orders_[sig->get_symbol()].insert(ord->get_order_id());
        strategy_order_remaining_[ord->get_order_id()] = ord_qty;

        eq_.push(ord);
    }

    void handle_ord(EventPtr event) {
        auto ord = std::static_pointer_cast<OrderEvent>(event);
        auto it  = engines_.find(ord->get_symbol());
        if (it == engines_.end())
            throw std::runtime_error("No execution engine for symbol: " + ord->get_symbol());

        auto engine = it->second;

        if (ord->is_cancel()) {
            engine->cancel_order(ord->get_order_id());
            cancel_pending_buy(ord->get_order_id());
            return;
        }

        Price    px_cents = static_cast<Price>(ord->get_price() * 100.0);
        Quantity qty      = ord->get_quantity();

        auto order = std::make_shared<Order>(
            ord->get_order_type(), ord->get_order_id(),
            ord->get_side(), px_cents, qty
        );

        if (ord->get_order_type() == OrderType::GoodForDay)
            gfd_orders_[ord->get_symbol()].insert(ord->get_order_id());

        const double slippage_pct = engine->get_slippage_pct();
        auto trades = engine->execute_order(order);

        for (const auto& trade : trades) {
            const TradeInfo& our_trade = (ord->get_side() == Side::Buy)
                ? trade.GetBidTrade()
                : trade.GetAskTrade();

            double raw_price  = our_trade.price_ / 100.0;
            double fill_price = (ord->get_side() == Side::Buy)
                ? raw_price * (1.0 + slippage_pct)
                : raw_price * (1.0 - slippage_pct);

            // Strategy orders are takers; always charge taker_fee.
            double commission = fill_price * our_trade.quantity_ * exec_config_.taker_fee;

            eq_.push(make_event<FillEvent>(
                ord->get_symbol(), ord->get_timestamp(),
                ord->get_order_id(), ord->get_side(),
                our_trade.quantity_, fill_price, commission
            ));
        }
    }

    void handle_fill(EventPtr event) {
        auto fill = std::static_pointer_cast<FillEvent>(event);

        if (fill->get_side() == Side::Buy)
            release_pending_buy(fill->get_order_id(), fill->get_quantity());

        auto rem_it = strategy_order_remaining_.find(fill->get_order_id());
        if (rem_it != strategy_order_remaining_.end()) {
            rem_it->second -= fill->get_quantity();
            if (rem_it->second <= MIN_ORDER_QTY) {
                strategy_order_remaining_.erase(rem_it);
                pending_strategy_orders_[fill->get_symbol()].erase(fill->get_order_id());
            }
        }

        auto ee_it = engines_.find(fill->get_symbol());
        if (ee_it != engines_.end()) {
            double new_pos = ee_it->second->get_position();
            strat_->set_position(fill->get_symbol(), new_pos);
            risk_mgr_->set_position(fill->get_symbol(), new_pos, fill->get_price());
        }

        strat_->on_fill(*fill);
    }
};

} // namespace quantcore