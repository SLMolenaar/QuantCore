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
#include "../Execution.h"
#include <memory>
#include <map>
#include <vector>

namespace quantcore {

/**
 * Main backtesting loop
 * 
 * Load market data (MarketDataEvents)
 * Strategy sees data, generates SignalEvents
 * Portfolio converts signals (OrderEvents)
 * Execution fills orders )FillEvents)
 * Update positions and PnL
 * Repeat until there's no mor edata
 */
class BacktestEngine {
public:
    BacktestEngine(double initial_capital = 100000.0)
        : initial_capital_(initial_capital)
        , current_capital_(initial_capital)
        , next_order_id_(1)
        , position_sizer_(std::make_shared<FixedPercentage>(0.1))
    {
    }

    /**
     * Add market data to backtest
     */
    void add_data(const std::string& symbol, const BarSeries& bars) {
        market_data_[symbol] = bars;
    }

    void set_strategy(std::shared_ptr<Strategy> strategy) {
        strategy_ = strategy;
    }

    void set_position_sizer(PositionSizerPtr sizer) {
        position_sizer_ = sizer;
    }

    PositionSizerPtr get_position_sizer() const {
        return position_sizer_;
    }

     //Run backtest, returns final portfolio value
    double run() {
        if (!strategy_) {
            throw std::runtime_error("No strategy set");
        }
        if (market_data_.empty()) {
            throw std::runtime_error("No market data loaded");
        }
        
        // Reset state
        event_queue_.clear();
        execution_engines_.clear();
        market_makers_.clear();
        last_prices_.clear();
        current_capital_ = initial_capital_;
        next_order_id_ = 1;
        strategy_->reset();

        // Clear equity tracking
        equity_curve_.clear();
        timestamps_.clear();

        // Initialize execution engines for each symbol
        for (const auto& [symbol, bars] : market_data_) {
            execution_engines_[symbol] = std::make_shared<ExecutionEngine>(symbol);
            market_makers_[symbol] = std::make_shared<MarketMaker>(0.0001, 100000);
        }

        load_market_data();

        // Record initial equity
        equity_curve_.push_back(initial_capital_);
        timestamps_.push_back(0);

        // Main event loop
        while (!event_queue_.empty()) {
            auto event = event_queue_.pop();

            switch (event->get_type()) {
                case EventType::MARKET_DATA:
                    handle_market_data(event);
                    break;

                case EventType::SIGNAL:
                    handle_signal(event);
                    break;

                case EventType::ORDER:
                    handle_order(event);
                    break;

                case EventType::FILL:
                    handle_fill(event);
                    break;
            }

            // After processing event, record equity if it was a market data event
            if (event->get_type() == EventType::MARKET_DATA) {
                double current_equity = calculate_portfolio_value();
                equity_curve_.push_back(current_equity);
                timestamps_.push_back(event->get_timestamp());
            }
        }

        return calculate_portfolio_value();
    }

    double get_total_pnl() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : execution_engines_) {
            total += engine->get_total_pnl();
        }
        return total;
    }

    double get_total_fees() const {
        double total = 0.0;
        for (const auto& [symbol, engine] : execution_engines_) {
            total += engine->get_total_fees();
        }
        return total;
    }

    /**
     * Get execution engine for a symbol (for inspection)
     */
    std::shared_ptr<ExecutionEngine> get_execution_engine(const std::string& symbol) const {
        auto it = execution_engines_.find(symbol);
        return it != execution_engines_.end() ? it->second : nullptr;
    }

    /**
     * Get equity curve (portfolio value over time)
     */
    std::vector<double> get_equity_curve() const {
        return equity_curve_;
    }

    /**
     * Get timestamps corresponding to equity curve
     */
    std::vector<int64_t> get_timestamps() const {
        return timestamps_;
    }

private:
    EventQueue event_queue_;
    std::shared_ptr<Strategy> strategy_;
    std::map<std::string, BarSeries> market_data_;
    std::map<std::string, std::shared_ptr<ExecutionEngine>> execution_engines_;
    std::map<std::string, std::shared_ptr<MarketMaker>> market_makers_;
    std::map<std::string, double> last_prices_;

    double initial_capital_;
    double current_capital_;
    uint64_t next_order_id_;

    PositionSizerPtr position_sizer_;

    // Equity tracking
    std::vector<double> equity_curve_;
    std::vector<int64_t> timestamps_;

    void load_market_data() {
        for (const auto& [symbol, bars] : market_data_) {
            for (const auto& bar : bars) {
                auto event = std::make_shared<MarketDataEvent>(
                    symbol,
                    bar.timestamp_ns,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume
                );
                event_queue_.push(event);
            }
        }
    }

    void handle_market_data(EventPtr event) {
        auto md_event = std::static_pointer_cast<MarketDataEvent>(event);

        std::string symbol = md_event->get_symbol();
        last_prices_[symbol] = md_event->get_close();

        auto mm_it = market_makers_.find(symbol);
        auto ee_it = execution_engines_.find(symbol);

        if (mm_it != market_makers_.end() && ee_it != execution_engines_.end()) {
            mm_it->second->update_quotes(*ee_it->second, *md_event);
        }

        strategy_->on_data(*md_event);

        // Check if strategy generated any signals
        auto signals = strategy_->get_signals();
        for (const auto& signal : signals) {
            event_queue_.push(signal);
        }
    }

    // convert signal event to order
    void handle_signal(EventPtr event) {
        auto signal = std::static_pointer_cast<SignalEvent>(event);

        auto price_it = last_prices_.find(signal->get_symbol());
        if (price_it == last_prices_.end()) {
            return;
        }

        double current_price = price_it->second;
        if (current_price == 0.0) {
            return;
        }

        // Get current position from execution engine
        auto ee_it = execution_engines_.find(signal->get_symbol());
        if (ee_it == execution_engines_.end()) {
            return;
        }

        double current_position = ee_it->second->get_position();

        // Calculate target position size using position sizer
        PositionSizingContext ctx(
            signal->get_strength(),
            current_capital_,
            current_price,
            current_position,
            0.02,  // portfolio_volatility - could be calculated from returns
            0.05   // stop_loss_distance - could come from strategy
        );

        double target_shares = position_sizer_->calculate_size(ctx);

        auto mm_it = market_makers_.find(signal->get_symbol());
        double spread_pct = 0.0001; // Default
        if (mm_it != market_makers_.end()) {
            spread_pct = mm_it->second->get_spread();
        }

        Side order_side;
        double order_price;

        if (signal->get_signal_type() == SignalType::BUY) {
            order_side = Side::Buy;
            order_price = current_price * (1.0 + spread_pct / 2.0);
        } else if (signal->get_signal_type() == SignalType::SELL) {
            order_side = Side::Sell;
            order_price = current_price * (1.0 - spread_pct / 2.0);
            target_shares = -target_shares;
        } else {
            return;
        }

        double position_delta = target_shares - current_position;

        if (std::abs(position_delta) < 1.0) {
            return;
        }

        if (position_delta > 0) {
            order_side = Side::Buy;
        } else {
            order_side = Side::Sell;
            position_delta = -position_delta;
        }

        auto order_event = std::make_shared<OrderEvent>(
            signal->get_symbol(),
            signal->get_timestamp(),
            order_side,
            OrderType::GoodTillCancel,
            std::abs(position_delta),
            order_price
        );
        order_event->set_order_id(next_order_id_++);
        event_queue_.push(order_event);
    }

    // execute through orderbook
    void handle_order(EventPtr event) {
        auto order_event = std::static_pointer_cast<OrderEvent>(event);

        auto it = execution_engines_.find(order_event->get_symbol());
        if (it == execution_engines_.end()) {
            return;
        }

        auto engine = it->second;

        if (order_event->is_cancel()) {
            engine->cancel_order(order_event->get_order_id());
            return;
        }

        Price price_cents = static_cast<Price>(order_event->get_price() * 100.0);
        Quantity quantity = static_cast<Quantity>(order_event->get_quantity());

        auto order = std::make_shared<Order>(
            order_event->get_order_type(),
            order_event->get_order_id(),
            order_event->get_side(),
            price_cents,
            quantity
        );

        auto trades = engine->execute_order(order);

        // Generate fill events for each trade
        for (const auto& trade : trades) {
            auto fill_event = std::make_shared<FillEvent>(
                order_event->get_symbol(),
                order_event->get_timestamp(),
                order_event->get_order_id(),
                order_event->get_side(),
                trade.GetBidTrade().quantity_,
                trade.GetBidTrade().price_ / 100.0,
                0.0
            );
            event_queue_.push(fill_event);
        }
    }

    // update positions
    void handle_fill(EventPtr event) {
        auto fill = std::static_pointer_cast<FillEvent>(event);

        auto ee_it = execution_engines_.find(fill->get_symbol());
        if (ee_it != execution_engines_.end()) {
            double engine_position = ee_it->second->get_position();
            strategy_->set_position(fill->get_symbol(), engine_position);
        }
        
        // Notify strategy
        strategy_->on_fill(*fill);
    }

    double calculate_portfolio_value() const {
        double value = current_capital_;
        value += get_total_pnl();
        return value;
    }
};

}