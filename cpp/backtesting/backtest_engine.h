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
#include <algorithm>
#include <iostream>

namespace quantcore {

class BacktestEngine {
public:
    BacktestEngine(double initial_capital = 100000.0)
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
        , default_volatility_(0.02)
        , default_stop_distance_(0.05)
        , volatility_lookback_(20)
        , bars_per_year_(252)
        , first_bar_timestamp_(0)
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
        // Record insertion order on first add; ignore duplicate add_data calls
        // for the same symbol (e.g. re-running with updated data).
        if (data_.find(symbol) == data_.end()) {
            symbol_order_index_[symbol] = symbol_order_.size();
            symbol_order_.push_back(symbol);
        }
        data_[symbol]          = bars;
        engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
        mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
        price_history_[symbol] = std::deque<double>();
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
        halted_   = false;
        intrabar_cash_reserved_ = 0.0;
        intrabar_timestamp_     = -1;
        strat_->reset();
        risk_mgr_->reset();
        risk_mgr_->set_capital(init_cap_, init_cap_);

        portfolio_ = std::make_shared<PortfolioContext>(init_cap_);
        strat_->portfolio_ = portfolio_.get();

        equity_.clear();
        timestamps_.clear();

        for (const auto& [symbol, bars] : data_) {
            engines_[symbol]       = std::make_shared<ExecutionEngine>(symbol, exec_config_);
            mms_[symbol]           = std::make_shared<MarketMaker>(mm_spread_, mm_levels_, mm_depth_, &order_pool_);
            price_history_[symbol] = std::deque<double>();
        }

        first_bar_timestamp_ = std::numeric_limits<int64_t>::max();
        for (const auto& [symbol, bars] : data_) {
            if (!bars.empty() && bars.front().timestamp_ns < first_bar_timestamp_)
                first_bar_timestamp_ = bars.front().timestamp_ns;
        }
        if (first_bar_timestamp_ == std::numeric_limits<int64_t>::max())
            first_bar_timestamp_ = 0;

        load_data();

        // Record the initial state before any bars are processed.
        // The equity curve will then have n_bars + 1 entries total:
        // [initial_capital, after_bar_1, after_bar_2, ..., after_bar_n].
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

                // Only record equity once per bar timestamp. With multiple symbols,
                // there are N MARKET_DATA events per bar (one per symbol). Recording
                // after each one produces N intermediate states per bar where only
                // some symbols have updated prices — this gives wildly wrong values
                // including apparent negative equity. Only snapshot after ALL symbols
                // for this timestamp have been processed, i.e. when the next event
                // in the queue has a different timestamp (or the queue is empty, meaning
                // this was the last market data event of the entire backtest).
                //
                // We compare against the next MARKET_DATA timestamp specifically,
                // not just any event, so that pending SIGNAL/ORDER/FILL events with
                // the same bar timestamp do not falsely trigger an early snapshot.
                bool is_last_md_for_bar = true;
                if (!eq_.empty()) {
                    // Peek ahead: find the next MARKET_DATA event's timestamp.
                    // Since the queue is a min-heap ordered by (timestamp, event_type)
                    // and MARKET_DATA has the lowest EventType value, the next
                    // MARKET_DATA event will always be at or after the current front.
                    // A simpler and correct check: if the very next event is a
                    // MARKET_DATA event with the same timestamp, we are not done yet.
                    auto next = eq_.peek();
                    if (next->get_type() == EventType::MARKET_DATA &&
                        next->get_timestamp() == event->get_timestamp()) {
                        is_last_md_for_bar = false;
                    }
                }

                if (is_last_md_for_bar) {
                    // Use curr_cap_ (computed from last_px_ in update_portfolio)
                    // rather than calc_portfolio_val() which uses orderbook mid prices.
                    // last_px_ is always the authoritative close price for each symbol.
                    equity_.push_back(curr_cap_);
                    timestamps_.push_back(event->get_timestamp());
                    risk_mgr_->set_capital(init_cap_, curr_cap_);

                    if (should_halt())
                        flatten_all_positions(event->get_timestamp());
                }
            }
        }

        return curr_cap_;
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

private:
    // Pools are declared first so they are destroyed last. C++ destructs members
    // in reverse declaration order; pools must outlive all objects allocated from them.
    std::pmr::unsynchronized_pool_resource order_pool_;
    std::pmr::unsynchronized_pool_resource event_pool_;

    EventQueue                eq_;
    std::shared_ptr<Strategy> strat_;

    std::unordered_map<std::string, BarSeries>                         data_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionEngine>>  engines_;
    std::unordered_map<std::string, std::shared_ptr<MarketMaker>>      mms_;
    std::unordered_map<std::string, double>                            last_px_;
    std::unordered_map<std::string, std::deque<double>>                price_history_;
    // Insertion-order record of symbols for deterministic signal processing.
    // symbol_order_ preserves add_data call order; symbol_order_index_ gives
    // O(1) lookup of each symbol's position so the per-bar sort is fast.
    std::vector<std::string>              symbol_order_;
    std::unordered_map<std::string, size_t> symbol_order_index_;

    double   init_cap_;
    double   curr_cap_;
    uint64_t next_oid_;
    bool     halted_;

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

    static constexpr size_t  PRICE_HISTORY_BUFFER = 10;
    // Minimum meaningful order size — rejects floating point noise but
    // allows fractional shares down to 0.00000001.
    static constexpr double  MIN_ORDER_QTY        = 1e-8;

    // Tracks notional reserved by buy signals already processed within the
    // current bar. Reset to zero when the bar timestamp changes.
    double  intrabar_cash_reserved_ = 0.0;
    int64_t intrabar_timestamp_     = -1;

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
        double vol = std::sqrt(sq_sum / returns.size()) * std::sqrt(static_cast<double>(bars_per_year_));
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

    void handle_md(EventPtr event) {
        auto md      = std::static_pointer_cast<MarketDataEvent>(event);
        auto symbol  = md->get_symbol();
        double price = md->get_close();
        last_px_[symbol] = price;

        auto& hist = price_history_[symbol];
        hist.push_back(price);
        if (hist.size() > volatility_lookback_ + PRICE_HISTORY_BUFFER) hist.pop_front();

        auto ee_it = engines_.find(symbol);
        if (ee_it != engines_.end()) {
            auto mm_it = mms_.find(symbol);
            if (mm_it != mms_.end())
                mm_it->second->update_quotes(*ee_it->second, *md);
        }

        strat_->on_data(*md);

        // Sort signals by insertion order (the order add_data was called) so
        // that same-timestamp signals are always processed in a deterministic,
        // user-controlled sequence. symbol_order_index_ gives O(1) position
        // lookup so the sort is O(k log k) on the small signals vector.
        auto signals = strat_->get_signals();
        std::sort(signals.begin(), signals.end(),
            [this](const std::shared_ptr<SignalEvent>& a,
                   const std::shared_ptr<SignalEvent>& b) {
                auto ia = symbol_order_index_.find(a->get_symbol());
                auto ib = symbol_order_index_.find(b->get_symbol());
                size_t idx_a = ia != symbol_order_index_.end() ? ia->second : symbol_order_.size();
                size_t idx_b = ib != symbol_order_index_.end() ? ib->second : symbol_order_.size();
                return idx_a < idx_b;
            });
        for (const auto& sig : signals) eq_.push(sig);
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

        double curr_pos      = ee_it->second->get_position();
        double portfolio_vol = calculate_volatility(sig->get_symbol());

        double target_pos = 0.0;

        // Reset intrabar reservation when we move to a new bar timestamp.
        // Signals for the same bar share a timestamp; signals on the next bar
        // get a fresh reservation budget.
        if (sig->get_timestamp() != intrabar_timestamp_) {
            intrabar_cash_reserved_ = 0.0;
            intrabar_timestamp_     = sig->get_timestamp();
        }

        // Available cash = portfolio cash minus what has already been reserved
        // by earlier signals processed within this same bar. This prevents
        // over-allocation when many symbols signal simultaneously on one bar.
        double available_cash = std::max(0.0, portfolio_->get_cash() - intrabar_cash_reserved_);

        if (sig->get_signal_type() == SignalType::BUY) {
            if (curr_pos < 0.0) {
                // Closing a short position: target is flat (0), not a new long.
                // The delta will be -curr_pos, submitting an exact exit order.
                target_pos = 0.0;
            } else {
                // Flat or already long — open or add to a long position.
                PositionSizingContext ctx(strength, available_cash, curr_px, curr_pos,
                                          portfolio_vol, default_stop_distance_);
                target_pos = sizer_->calculate_size(ctx);
            }
        } else if (sig->get_signal_type() == SignalType::SELL) {
            if (curr_pos > 0.0) {
                // Closing a long position: target is flat (0), not a new short.
                // The delta will be -curr_pos, submitting an exact exit order.
                target_pos = 0.0;
            } else {
                // Flat or already short — open or add to a short position.
                PositionSizingContext ctx(strength, available_cash, curr_px, curr_pos,
                                          portfolio_vol, default_stop_distance_);
                target_pos = -sizer_->calculate_size(ctx);
            }
        } else {
            return;
        }

        double delta = target_pos - curr_pos;
        // Reject floating point noise but allow fractional shares
        if (std::abs(delta) < MIN_ORDER_QTY) return;

        Side   ord_side = (delta > 0.0) ? Side::Buy : Side::Sell;
        double ord_qty  = std::abs(delta);

        // Reserve the notional cost of this buy so subsequent signals on the
        // same bar see reduced available cash and don't over-allocate.
        if (ord_side == Side::Buy) {
            intrabar_cash_reserved_ += ord_qty * curr_px;
        }

        double spread = mm_spread_;
        auto   mm_it  = mms_.find(sig->get_symbol());
        if (mm_it != mms_.end()) spread = mm_it->second->get_spread();

        double ord_px = (ord_side == Side::Buy)
            ? curr_px * (1.0 + spread / 2.0)
            : curr_px * (1.0 - spread / 2.0);

        // Notify the strategy if risk limits reject the order so it can
        // react (e.g. reduce position targets, log, or halt trading).
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
            return;
        }

        Price    px_cents = static_cast<Price>(ord->get_price() * 100.0);
        // Quantity stays as double — no cast to uint32_t, preserving fractional shares
        Quantity qty      = ord->get_quantity();

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

            double raw_price  = our_trade.price_ / 100.0;
            double fill_price = (ord->get_side() == Side::Buy)
                ? raw_price * (1.0 + slippage_pct)
                : raw_price * (1.0 - slippage_pct);

            // Mirror the fee calculation in ExecutionEngine::calculate_fee so
            // the commission field on FillEvent matches what is booked to P&L.
            // Strategy orders always cross the spread and are charged taker fees.
            double commission = fill_price
                                * our_trade.quantity_
                                * exec_config_.taker_fee;

            auto fill = make_event<FillEvent>(
                ord->get_symbol(), ord->get_timestamp(),
                ord->get_order_id(), ord->get_side(),
                our_trade.quantity_,
                fill_price,
                commission
            );
            eq_.push(fill);
        }
    }

    void handle_fill(EventPtr event) {
        auto fill  = std::static_pointer_cast<FillEvent>(event);
        auto ee_it = engines_.find(fill->get_symbol());
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