#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "backtesting/backtest_engine.h"
#include "backtesting/data_loader.h"
#include "backtesting/bar_data.h"
#include "backtesting/market_data_event.h"
#include "backtesting/signal_event.h"
#include "backtesting/order_event.h"
#include "backtesting/fill_event.h"
#include "backtesting/strategy.h"
#include "strategies/buy_and_hold.h"
#include "strategies/sma_crossover.h"
#include "strategies/mean_reversion.h"
#include "Execution.h"
#include "backtesting/position_sizer.h"
#include "backtesting/risk_manager.h"
#include "backtesting/portfolio_context.h"
#include "strategies/pairs_trading.h"

namespace py = pybind11;
using namespace quantcore;

// Python wrapper for Strategy to allow Python subclassing
class PyStrategy : public Strategy {
public:
    using Strategy::Strategy;

    void on_data(const MarketDataEvent& event) override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Strategy,
            on_data,
            event
        );
    }

    void on_fill(const FillEvent& event) override {
        PYBIND11_OVERRIDE(
            void,
            Strategy,
            on_fill,
            event
        );
    }

    void reset() override {
        PYBIND11_OVERRIDE(
            void,
            Strategy,
            reset
        );
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "QuantCore C++ backtesting engine";

    // ============================================================================
    // ENUMS
    // ============================================================================

    py::enum_<SignalType>(m, "SignalType")
        .value("BUY", SignalType::BUY)
        .value("SELL", SignalType::SELL)
        .value("HOLD", SignalType::HOLD)
        .export_values();

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::Buy)
        .value("SELL", Side::Sell)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("GOOD_TILL_CANCEL", OrderType::GoodTillCancel)
        .value("IMMEDIATE_OR_CANCEL", OrderType::ImmediateOrCancel)
        .value("MARKET", OrderType::Market)
        .value("GOOD_FOR_DAY", OrderType::GoodForDay)
        .value("FILL_OR_KILL", OrderType::FillOrKill)
        .export_values();

    // ============================================================================
    // BAR DATA
    // ============================================================================

    py::class_<BarData>(m, "BarData")
        .def(py::init<>())
        .def(py::init<const std::string&, int64_t, double, double, double, double, double>(),
             py::arg("symbol"),
             py::arg("timestamp_ns"),
             py::arg("open"),
             py::arg("high"),
             py::arg("low"),
             py::arg("close"),
             py::arg("volume"))
        .def_readwrite("symbol", &BarData::symbol)
        .def_readwrite("timestamp_ns", &BarData::timestamp_ns)
        .def_readwrite("open", &BarData::open)
        .def_readwrite("high", &BarData::high)
        .def_readwrite("low", &BarData::low)
        .def_readwrite("close", &BarData::close)
        .def_readwrite("volume", &BarData::volume)
        .def("typical_price", &BarData::typical_price)
        .def("range", &BarData::range)
        .def("is_bullish", &BarData::is_bullish)
        .def("is_bearish", &BarData::is_bearish)
        .def("__repr__", [](const BarData& bar) {
            return "<BarData " + bar.symbol + " @ " + std::to_string(bar.close) + ">";
        });

    // ============================================================================
    // DATA LOADER
    // ============================================================================

    py::class_<CSVDataLoader>(m, "CSVDataLoader")
        .def_static("load", &CSVDataLoader::load,
                   py::arg("filepath"),
                   py::arg("symbol") = "",
                   py::arg("has_header") = true,
                   "Load OHLCV data from CSV file");

    // ============================================================================
    // EVENTS
    // ============================================================================

    py::class_<MarketDataEvent, std::shared_ptr<MarketDataEvent>>(m, "MarketDataEvent")
        .def(py::init<const std::string&, int64_t, double, double, double, double, double>(),
             py::arg("symbol"),
             py::arg("timestamp_ns"),
             py::arg("open"),
             py::arg("high"),
             py::arg("low"),
             py::arg("close"),
             py::arg("volume"))
        .def("get_symbol", &MarketDataEvent::get_symbol)
        .def("get_timestamp", &MarketDataEvent::get_timestamp)
        .def("get_open", &MarketDataEvent::get_open)
        .def("get_high", &MarketDataEvent::get_high)
        .def("get_low", &MarketDataEvent::get_low)
        .def("get_close", &MarketDataEvent::get_close)
        .def("get_volume", &MarketDataEvent::get_volume)
        .def("get_price", &MarketDataEvent::get_price)
        .def("__repr__", [](const MarketDataEvent& event) {
            return event.to_string();
        });

    py::class_<SignalEvent, std::shared_ptr<SignalEvent>>(m, "SignalEvent")
        .def(py::init<const std::string&, int64_t, SignalType, double, const std::string&>(),
             py::arg("symbol"),
             py::arg("timestamp_ns"),
             py::arg("signal_type"),
             py::arg("strength") = 1.0,
             py::arg("strategy_id") = "default")
        .def("get_symbol", &SignalEvent::get_symbol)
        .def("get_timestamp", &SignalEvent::get_timestamp)
        .def("get_signal_type", &SignalEvent::get_signal_type)
        .def("get_strength", &SignalEvent::get_strength)
        .def("get_strategy_id", &SignalEvent::get_strategy_id)
        .def("__repr__", [](const SignalEvent& event) {
            return event.to_string();
        });

    py::class_<FillEvent, std::shared_ptr<FillEvent>>(m, "FillEvent")
        .def(py::init<const std::string&, int64_t, uint64_t, Side, double, double, double>(),
             py::arg("symbol"),
             py::arg("timestamp_ns"),
             py::arg("order_id"),
             py::arg("side"),
             py::arg("quantity"),
             py::arg("price"),
             py::arg("commission") = 0.0)
        .def("get_symbol", &FillEvent::get_symbol)
        .def("get_timestamp", &FillEvent::get_timestamp)
        .def("get_order_id", &FillEvent::get_order_id)
        .def("get_side", &FillEvent::get_side)
        .def("get_quantity", &FillEvent::get_quantity)
        .def("get_price", &FillEvent::get_price)
        .def("get_commission", &FillEvent::get_commission)
        .def("get_total_cost", &FillEvent::get_total_cost)
        .def("__repr__", [](const FillEvent& event) {
            return event.to_string();
        });

    // ============================================================================
    // EXECUTION ENGINE
    // ============================================================================

    py::class_<ExecutionConfig>(m, "ExecutionConfig")
        .def(py::init<>())
        .def_readwrite("maker_fee", &ExecutionConfig::maker_fee)
        .def_readwrite("taker_fee", &ExecutionConfig::taker_fee)
        .def_readwrite("latency_ns", &ExecutionConfig::latency_ns)
        .def_readwrite("slippage_pct", &ExecutionConfig::slippage_pct);

    py::class_<ExecutionEngine, std::shared_ptr<ExecutionEngine>>(m, "ExecutionEngine")
        .def(py::init<const std::string&, ExecutionConfig>(),
             py::arg("symbol") = "DEFAULT",
             py::arg("config") = ExecutionConfig())
        .def("get_position", &ExecutionEngine::get_position,
             "Get current position for the symbol")
        .def("get_average_price", &ExecutionEngine::get_average_price,
             "Get average entry price")
        .def("get_realized_pnl", &ExecutionEngine::get_realized_pnl,
             "Get realized PnL")
        .def("get_unrealized_pnl", &ExecutionEngine::get_unrealized_pnl,
             "Get unrealized PnL based on current market price")
        .def("get_total_pnl", &ExecutionEngine::get_total_pnl,
             "Get total PnL (realized + unrealized)")
        .def("get_total_fees", &ExecutionEngine::get_total_fees,
             "Get total fees paid")
        .def("get_best_bid", &ExecutionEngine::get_best_bid,
             "Get best bid price")
        .def("get_best_ask", &ExecutionEngine::get_best_ask,
             "Get best ask price")
        .def("get_mid_price", &ExecutionEngine::get_mid_price,
             "Get mid price (average of best bid and ask)")
        .def("reset", &ExecutionEngine::reset,
             "Reset all state for new backtest run");

    // ============================================================================
    // STRATEGY
    // ============================================================================

    py::class_<Strategy, PyStrategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def(py::init<const std::string&>(),
             py::arg("name") = "Strategy")
        .def("on_data", &Strategy::on_data,
             "Called on each market data update")
        .def("on_fill", &Strategy::on_fill,
             "Called when an order is filled")
        .def("get_name", &Strategy::get_name,
             "Get strategy name")
        .def("get_signals", &Strategy::get_signals,
             "Get and clear pending signals")
        .def("has_signals", &Strategy::has_signals,
             "Check if strategy has pending signals")
        .def("reset", &Strategy::reset,
             "Reset strategy state");

    // ============================================================================
    // BUILT-IN STRATEGIES
    // ============================================================================

    py::class_<BuyAndHold, Strategy, std::shared_ptr<BuyAndHold>>(m, "BuyAndHold")
        .def(py::init<>(),
             "Simple buy and hold strategy - buys once on first bar");

    py::class_<SMACrossover, Strategy, std::shared_ptr<SMACrossover>>(m, "SMACrossover")
        .def(py::init<size_t, size_t>(),
             py::arg("fast_period") = 50,
             py::arg("slow_period") = 200,
             "SMA crossover strategy - buy when fast > slow, sell when fast < slow");

    py::class_<MeanReversion, Strategy, std::shared_ptr<MeanReversion>>(m, "MeanReversion")
        .def(py::init<size_t, double, double>(),
             py::arg("lookback") = 20,
             py::arg("entry_threshold") = 1.5,
             py::arg("exit_threshold") = 0.5,
             "Mean reversion strategy - trade based on z-score")
        .def("get_signal_count", &MeanReversion::get_signal_count,
             "Get number of signals generated");

    py::class_<PairsTrading, Strategy, std::shared_ptr<PairsTrading>>(m, "PairsTrading")
        .def(py::init<std::string, std::string, size_t, double, double>(),
             py::arg("symbol1"),
             py::arg("symbol2"),
             py::arg("lookback") = 20,
             py::arg("entry_zscore") = 2.0,
             py::arg("exit_zscore") = 0.5,
             "Pairs trading strategy - statistical arbitrage between two correlated assets");

    // ============================================================================
    // BACKTEST ENGINE
    // ============================================================================

    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<double>(),
             py::arg("initial_capital") = 100000.0,
             "Create backtest engine with initial capital")
        .def("add_data", &BacktestEngine::add_data,
             py::arg("symbol"),
             py::arg("bars"),
             "Add market data for a symbol")
        .def("set_strategy", &BacktestEngine::set_strategy,
             py::arg("strategy"),
             "Set the trading strategy")
        .def("run", &BacktestEngine::run,
             "Run the backtest and return final portfolio value")
        .def("get_total_pnl", &BacktestEngine::get_total_pnl,
             "Get total PnL across all symbols")
        .def("get_total_fees", &BacktestEngine::get_total_fees,
             "Get total fees paid across all symbols")
        .def("get_execution_engine", &BacktestEngine::get_execution_engine,
             py::arg("symbol"),
             "Get execution engine for a symbol (for inspection)")
        .def("set_position_sizer", &BacktestEngine::set_position_sizer,
             py::arg("sizer"),
             "Set the position sizing method")
        .def("get_position_sizer", &BacktestEngine::get_position_sizer,
             "Get current position sizer")
        .def("get_equity_curve", &BacktestEngine::get_equity_curve,
             "Get portfolio value over time")
        .def("get_timestamps", &BacktestEngine::get_timestamps,
             "Get timestamps corresponding to equity curve")
        .def("set_risk_limits", &BacktestEngine::set_risk_limits)
        .def("get_risk_limits", &BacktestEngine::get_risk_limits)
        .def("get_risk_manager", &BacktestEngine::get_risk_manager);


    // ============================================================================
    // POSITION SIZING
    // ============================================================================

    py::class_<PositionSizingContext>(m, "PositionSizingContext")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("signal_strength") = 1.0,
             py::arg("current_capital") = 100000.0,
             py::arg("current_price") = 100.0,
             py::arg("current_position") = 0.0,
             py::arg("portfolio_volatility") = 0.02,
             py::arg("stop_loss_distance") = 0.05,
             "Context for position sizing calculations")
        .def_readwrite("signal_strength", &PositionSizingContext::signal_strength)
        .def_readwrite("current_capital", &PositionSizingContext::current_capital)
        .def_readwrite("current_price", &PositionSizingContext::current_price)
        .def_readwrite("current_position", &PositionSizingContext::current_position)
        .def_readwrite("portfolio_volatility", &PositionSizingContext::portfolio_volatility)
        .def_readwrite("stop_loss_distance", &PositionSizingContext::stop_loss_distance);

    py::class_<PositionSizer, std::shared_ptr<PositionSizer>>(m, "PositionSizer")
        .def("calculate_size", &PositionSizer::calculate_size,
             py::arg("context"),
             "Calculate position size based on context")
        .def("get_name", &PositionSizer::get_name,
             "Get position sizer name")
        .def("set_max_position_size", &PositionSizer::set_max_position_size,
             py::arg("max_size"),
             "Set maximum position size constraint")
        .def("set_min_position_size", &PositionSizer::set_min_position_size,
             py::arg("min_size"),
             "Set minimum position size constraint")
        .def("set_max_leverage", &PositionSizer::set_max_leverage,
             py::arg("max_leverage"),
             "Set maximum leverage constraint");

    py::class_<FixedPercentage, PositionSizer, std::shared_ptr<FixedPercentage>>(m, "FixedPercentage")
        .def(py::init<double>(),
             py::arg("percentage") = 0.1,
             "Fixed percentage of capital sizing (e.g., 0.1 = 10%)");

    py::class_<RiskBased, PositionSizer, std::shared_ptr<RiskBased>>(m, "RiskBased")
        .def(py::init<double>(),
             py::arg("risk_per_trade") = 0.01,
             "Risk-based sizing using stop-loss distance (e.g., 0.01 = 1% risk)");

    py::class_<KellyCriterion, PositionSizer, std::shared_ptr<KellyCriterion>>(m, "KellyCriterion")
        .def(py::init<double, double, double, double>(),
             py::arg("win_rate"),
             py::arg("avg_win"),
             py::arg("avg_loss"),
             py::arg("fraction") = 1.0,
             "Kelly Criterion sizing based on historical win rate and win/loss ratio");

    py::class_<EqualWeight, PositionSizer, std::shared_ptr<EqualWeight>>(m, "EqualWeight")
        .def(py::init<int>(),
             py::arg("num_positions"),
             "Equal weight across N positions");

    py::class_<VolatilityTargeting, PositionSizer, std::shared_ptr<VolatilityTargeting>>(m, "VolatilityTargeting")
        .def(py::init<double>(),
             py::arg("target_volatility") = 0.15,
             "Volatility targeting position sizing");

    py::class_<FixedShares, PositionSizer, std::shared_ptr<FixedShares>>(m, "FixedShares")
        .def(py::init<double>(),
             py::arg("num_shares"),
             "Fixed number of shares per signal");

    //============================================================================
    // RISK MANAGEMENT
    // ============================================================================

    py::enum_<RiskCheckResult>(m, "RiskCheckResult")
        .value("APPROVED", RiskCheckResult::APPROVED)
        .value("REJECTED_POSITION_LIMIT", RiskCheckResult::REJECTED_POSITION_LIMIT)
        .value("REJECTED_LEVERAGE_LIMIT", RiskCheckResult::REJECTED_LEVERAGE_LIMIT)
        .value("REJECTED_CAPITAL_LIMIT", RiskCheckResult::REJECTED_CAPITAL_LIMIT)
        .value("REJECTED_LOSS_LIMIT", RiskCheckResult::REJECTED_LOSS_LIMIT)
        .value("REJECTED_ORDER_SIZE", RiskCheckResult::REJECTED_ORDER_SIZE);

    py::class_<RiskCheckResponse>(m, "RiskCheckResponse")
        .def(py::init<>())
        .def_readwrite("result", &RiskCheckResponse::result)
        .def_readwrite("reason", &RiskCheckResponse::reason)
        .def("is_approved", &RiskCheckResponse::is_approved);

    py::class_<RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("max_position_pct", &RiskLimits::max_position_pct)
        .def_readwrite("max_leverage", &RiskLimits::max_leverage)
        .def_readwrite("max_loss_pct", &RiskLimits::max_loss_pct)
        .def_readwrite("max_order_value", &RiskLimits::max_order_value)
        .def_readwrite("enabled", &RiskLimits::enabled)
        .def("validate", &RiskLimits::validate);

    py::class_<RiskManager, std::shared_ptr<RiskManager>>(m, "RiskManager")
        .def(py::init<>())
        .def(py::init<const RiskLimits&>())
        .def("set_capital", &RiskManager::set_capital)
        .def("set_position", &RiskManager::set_position)
        .def("get_position", &RiskManager::get_position)
        .def("set_limits", &RiskManager::set_limits)
        .def("get_limits", &RiskManager::get_limits)
        .def("check_order", &RiskManager::check_order)
        .def("update_position", &RiskManager::update_position)
        .def("reset", &RiskManager::reset)
        .def("get_all_positions", &RiskManager::get_all_positions);


    // ============================================================================
    // PORTFOLIO CONTEXT
    // ============================================================================

    py::class_<PortfolioContext, std::shared_ptr<PortfolioContext>>(m, "PortfolioContext")
        .def(py::init<double>(),
             py::arg("initial_capital"),
             "Create portfolio context")
        .def("get_cash", &PortfolioContext::get_cash,
             "Get available cash")
        .def("get_initial_capital", &PortfolioContext::get_initial_capital,
             "Get initial capital")
        .def("get_position", &PortfolioContext::get_position,
             py::arg("symbol"),
             "Get position for a symbol")
        .def("get_price", &PortfolioContext::get_price,
             py::arg("symbol"),
             "Get current price for a symbol")
        .def("get_position_value", &PortfolioContext::get_position_value,
             py::arg("symbol"),
             "Get market value of position")
        .def("get_total_position_value", &PortfolioContext::get_total_position_value,
             "Get total value of all positions")
        .def("get_portfolio_value", &PortfolioContext::get_portfolio_value,
             "Get total portfolio value (cash + positions)")
        .def("get_leverage", &PortfolioContext::get_leverage,
             "Get portfolio leverage ratio")
        .def("get_position_weight", &PortfolioContext::get_position_weight,
             py::arg("symbol"),
             "Get position weight as % of portfolio")
        .def("get_all_positions", &PortfolioContext::get_all_positions,
             "Get all positions as dict")
        .def("get_all_prices", &PortfolioContext::get_all_prices,
             "Get all current prices as dict")
        .def("num_positions", &PortfolioContext::num_positions,
             "Get number of open positions")
        .def("has_position", &PortfolioContext::has_position,
             py::arg("symbol"),
             "Check if has position in symbol");

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    m.def("hello", []() {
        return "Hello from QuantCore C++!";
    }, "A simple test function");

    m.def("version", []() {
        return "0.1.0";
    }, "Get QuantCore version");

    // ============================================================================
    // MODULE METADATA
    // ============================================================================

    m.attr("__version__") = "0.1.0";
}