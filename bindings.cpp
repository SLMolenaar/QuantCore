#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

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

// Trampoline class enabling Python subclasses of Strategy.
class PyStrategy : public Strategy {
public:
    using Strategy::Strategy;

    void on_data(const MarketDataEvent& event) override {
        PYBIND11_OVERRIDE_PURE(void, Strategy, on_data, event);
    }

    void on_fill(const FillEvent& event) override {
        PYBIND11_OVERRIDE(void, Strategy, on_fill, event);
    }

    // Forwards risk-limit rejections to Python subclasses that override on_rejected.
    void on_rejected(const std::string& symbol, const std::string& reason) override {
        PYBIND11_OVERRIDE(void, Strategy, on_rejected, symbol, reason);
    }

    void reset() override {
        PYBIND11_OVERRIDE(void, Strategy, reset);
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "QuantCore C++ backtesting engine";

    // ============================================================================
    // ENUMS
    // ============================================================================

    py::enum_<SignalType>(m, "SignalType")
        .value("BUY",  SignalType::BUY)
        .value("SELL", SignalType::SELL)
        .value("HOLD", SignalType::HOLD)
        .export_values();

    py::enum_<Side>(m, "Side")
        .value("BUY",  Side::Buy)
        .value("SELL", Side::Sell)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("GOOD_TILL_CANCEL",     OrderType::GoodTillCancel)
        .value("IMMEDIATE_OR_CANCEL",  OrderType::ImmediateOrCancel)
        .value("MARKET",               OrderType::Market)
        .value("GOOD_FOR_DAY",         OrderType::GoodForDay)
        .value("FILL_OR_KILL",         OrderType::FillOrKill)
        .export_values();

    // ============================================================================
    // BAR DATA
    // ============================================================================

    py::class_<BarData>(m, "BarData")
        .def(py::init<>())
        .def(py::init<const std::string&, int64_t, double, double, double, double, double>(),
             py::arg("symbol"), py::arg("timestamp_ns"), py::arg("open"),
             py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"))
        .def_readwrite("symbol",       &BarData::symbol)
        .def_readwrite("timestamp_ns", &BarData::timestamp_ns)
        .def_readwrite("open",         &BarData::open)
        .def_readwrite("high",         &BarData::high)
        .def_readwrite("low",          &BarData::low)
        .def_readwrite("close",        &BarData::close)
        .def_readwrite("volume",       &BarData::volume)
        .def("typical_price", &BarData::typical_price)
        .def("range",         &BarData::range)
        .def("is_bullish",    &BarData::is_bullish)
        .def("is_bearish",    &BarData::is_bearish)
        .def("__repr__", [](const BarData& bar) {
            return "<BarData " + bar.symbol + " @ " + std::to_string(bar.close) + ">";
        });

    // ============================================================================
    // DATA LOADER
    // ============================================================================

    py::class_<CSVDataLoader>(m, "CSVDataLoader")
        .def_static("load", &CSVDataLoader::load,
                    py::arg("filepath"),
                    py::arg("symbol")       = "",
                    py::arg("has_header")   = true,
                    py::arg("max_skip_pct") = 0.20);

    // ============================================================================
    // EVENTS
    // ============================================================================

    py::class_<MarketDataEvent, std::shared_ptr<MarketDataEvent>>(m, "MarketDataEvent")
        .def(py::init<const std::string&, int64_t, double, double, double, double, double>(),
             py::arg("symbol"), py::arg("timestamp_ns"), py::arg("open"),
             py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"))
        .def("get_symbol",    &MarketDataEvent::get_symbol)
        .def("get_timestamp", &MarketDataEvent::get_timestamp)
        .def("get_open",      &MarketDataEvent::get_open)
        .def("get_high",      &MarketDataEvent::get_high)
        .def("get_low",       &MarketDataEvent::get_low)
        .def("get_close",     &MarketDataEvent::get_close)
        .def("get_volume",    &MarketDataEvent::get_volume)
        .def("get_price",     &MarketDataEvent::get_price)
        .def_property_readonly("symbol",       &MarketDataEvent::get_symbol)
        .def_property_readonly("timestamp_ns", &MarketDataEvent::get_timestamp)
        .def_property_readonly("open",         &MarketDataEvent::get_open)
        .def_property_readonly("high",         &MarketDataEvent::get_high)
        .def_property_readonly("low",          &MarketDataEvent::get_low)
        .def_property_readonly("close",        &MarketDataEvent::get_close)
        .def_property_readonly("volume",       &MarketDataEvent::get_volume)
        .def("__repr__", [](const MarketDataEvent& e) { return e.to_string(); });

    py::class_<SignalEvent, std::shared_ptr<SignalEvent>>(m, "SignalEvent")
        .def(py::init<const std::string&, int64_t, SignalType, double, const std::string&>(),
             py::arg("symbol"), py::arg("timestamp_ns"), py::arg("signal_type"),
             py::arg("strength") = 1.0, py::arg("strategy_id") = "default")
        .def("get_symbol",      &SignalEvent::get_symbol)
        .def("get_timestamp",   &SignalEvent::get_timestamp)
        .def("get_signal_type", &SignalEvent::get_signal_type)
        .def("get_strength",    &SignalEvent::get_strength)
        .def("get_strategy_id", &SignalEvent::get_strategy_id)
        .def_property_readonly("symbol",       &SignalEvent::get_symbol)
        .def_property_readonly("timestamp_ns", &SignalEvent::get_timestamp)
        .def_property_readonly("signal_type",  &SignalEvent::get_signal_type)
        .def_property_readonly("strength",     &SignalEvent::get_strength)
        .def_property_readonly("strategy_id",  &SignalEvent::get_strategy_id)
        .def("__repr__", [](const SignalEvent& e) { return e.to_string(); });

    py::class_<FillEvent, std::shared_ptr<FillEvent>>(m, "FillEvent")
        .def(py::init<const std::string&, int64_t, uint64_t, Side, double, double, double>(),
             py::arg("symbol"), py::arg("timestamp_ns"), py::arg("order_id"),
             py::arg("side"), py::arg("quantity"), py::arg("price"),
             py::arg("commission") = 0.0)
        .def("get_symbol",     &FillEvent::get_symbol)
        .def("get_timestamp",  &FillEvent::get_timestamp)
        .def("get_order_id",   &FillEvent::get_order_id)
        .def("get_side",       &FillEvent::get_side)
        .def("get_quantity",   &FillEvent::get_quantity)
        .def("get_price",      &FillEvent::get_price)
        .def("get_commission", &FillEvent::get_commission)
        .def("get_total_cost", &FillEvent::get_total_cost)
        .def_property_readonly("symbol",       &FillEvent::get_symbol)
        .def_property_readonly("timestamp_ns", &FillEvent::get_timestamp)
        .def_property_readonly("order_id",     &FillEvent::get_order_id)
        .def_property_readonly("side",         &FillEvent::get_side)
        .def_property_readonly("quantity",     &FillEvent::get_quantity)
        .def_property_readonly("price",        &FillEvent::get_price)
        .def_property_readonly("commission",   &FillEvent::get_commission)
        .def("__repr__", [](const FillEvent& e) { return e.to_string(); });

    // ============================================================================
    // EXECUTION ENGINE
    // ============================================================================

    py::class_<ExecutionConfig>(m, "ExecutionConfig")
        .def(py::init<>())
        .def_readwrite("maker_fee",    &ExecutionConfig::maker_fee)
        .def_readwrite("taker_fee",    &ExecutionConfig::taker_fee)
        .def_readwrite("latency_ns",   &ExecutionConfig::latency_ns)
        .def_readwrite("slippage_pct", &ExecutionConfig::slippage_pct);

    py::class_<ExecutionEngine, std::shared_ptr<ExecutionEngine>>(m, "ExecutionEngine")
        .def(py::init<const std::string&, ExecutionConfig>(),
             py::arg("symbol") = "DEFAULT", py::arg("config") = ExecutionConfig())
        .def("get_position",       &ExecutionEngine::get_position)
        .def("get_average_price",  &ExecutionEngine::get_average_price)
        .def("get_realized_pnl",   &ExecutionEngine::get_realized_pnl)
        .def("get_unrealized_pnl", &ExecutionEngine::get_unrealized_pnl)
        .def("get_total_pnl",      &ExecutionEngine::get_total_pnl)
        .def("get_total_fees",     &ExecutionEngine::get_total_fees)
        // get_best_bid and get_best_ask return Price (int32_t cents) in C++.
        // Divide by 100.0 here so Python callers receive consistent float dollars,
        // matching every other price value in the API.
        .def("get_best_bid", [](const ExecutionEngine& self) -> double {
            return self.get_best_bid() / 100.0;
        })
        .def("get_best_ask", [](const ExecutionEngine& self) -> double {
            return self.get_best_ask() / 100.0;
        })
        .def("get_mid_price",      &ExecutionEngine::get_mid_price)
        .def("get_closed_trade_pnls",   &ExecutionEngine::get_closed_trade_pnls)
        .def("reset",              &ExecutionEngine::reset);

    // ============================================================================
    // STRATEGY
    // ============================================================================

    py::class_<Strategy, PyStrategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def(py::init<const std::string&>(), py::arg("name") = "Strategy")
        .def("on_data",          &Strategy::on_data)
        .def("on_fill",          &Strategy::on_fill)
        .def("on_rejected",      &Strategy::on_rejected,
             py::arg("symbol"), py::arg("reason"))
        .def("get_name",         &Strategy::get_name)
        .def("get_signals",      &Strategy::get_signals)
        .def("has_signals",      &Strategy::has_signals)
        .def("reset",            &Strategy::reset)
        .def("generate_signal",  &Strategy::generate_signal,
             py::arg("symbol"), py::arg("signal_type"),
             py::arg("strength"), py::arg("timestamp_ns"))
        .def("set_position",     &Strategy::set_position,
             py::arg("symbol"), py::arg("quantity"))
        .def("get_position",     &Strategy::get_position,   py::arg("symbol"))
        .def("has_position",     &Strategy::has_position,   py::arg("symbol"))
        // get_portfolio() returns a raw pointer owned by BacktestEngine.
        // The pointer is null until the engine attaches it at run(), so we
        // must return py::none() explicitly — pybind11 does not convert a
        // null raw pointer to None automatically and would raise or segfault.
        .def("get_portfolio",
             [](Strategy& self) -> py::object {
                 PortfolioContext* p = self.get_portfolio();
                 if (!p) return py::none();
                 return py::cast(p, py::return_value_policy::reference);
             });

    // ============================================================================
    // BUILT-IN STRATEGIES
    // ============================================================================

    py::class_<BuyAndHold, Strategy, std::shared_ptr<BuyAndHold>>(m, "BuyAndHold")
        .def(py::init<>());

    py::class_<SMACrossover, Strategy, std::shared_ptr<SMACrossover>>(m, "SMACrossover")
        .def(py::init<size_t, size_t>(),
             py::arg("fast_period") = 50, py::arg("slow_period") = 200);

    py::class_<MeanReversion, Strategy, std::shared_ptr<MeanReversion>>(m, "MeanReversion")
        .def(py::init<size_t, double, double>(),
             py::arg("lookback") = 20, py::arg("entry_threshold") = 1.5,
             py::arg("exit_threshold") = 0.5)
        .def("get_signal_count", &MeanReversion::get_signal_count);

    py::class_<PairsTrading, Strategy, std::shared_ptr<PairsTrading>>(m, "PairsTrading")
        .def(py::init<std::string, std::string, size_t, double, double>(),
             py::arg("symbol1"), py::arg("symbol2"),
             py::arg("lookback") = 20, py::arg("entry_zscore") = 2.0,
             py::arg("exit_zscore") = 0.5)
        .def("in_trade", &PairsTrading::in_trade);

    // ============================================================================
    // BACKTEST ENGINE
    // ============================================================================

    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<double>(),
             py::arg("initial_capital") = 100000.0)
        .def(py::init<double, ExecutionConfig>(),
             py::arg("initial_capital") = 100000.0,
             py::arg("exec_config") = ExecutionConfig())

        // List[BarData] overload — kept for backward compatibility.
        .def("add_data", &BacktestEngine::add_data,
             py::arg("symbol"), py::arg("bars"))

        // (N, 6) float64 numpy array: [timestamp_ns, open, high, low, close, volume].
        // One boundary crossing instead of N individual pybind11 object crossings;
        // roughly 3-5x faster for large datasets.
        //
        // Note: timestamp_ns is cast double→int64. float64 has 53-bit mantissa so
        // timestamps > 2^53 ns (~year 2255) lose sub-microsecond precision — fine
        // for current-era UNIX nanosecond timestamps.
        .def("add_data",
            [](BacktestEngine& self, const std::string& symbol,
               py::array_t<double, py::array::c_style | py::array::forcecast> data) {
                py::buffer_info buf = data.request();
                if (buf.ndim != 2 || buf.shape[1] != 6)
                    throw std::invalid_argument(
                        "data must be shape (N, 6): "
                        "[timestamp_ns, open, high, low, close, volume]"
                    );
                const Py_ssize_t n = buf.shape[0];
                BarSeries bars;
                bars.reserve(static_cast<size_t>(n));
                const double* ptr = static_cast<const double*>(buf.ptr);
                for (Py_ssize_t i = 0; i < n; ++i) {
                    const double* row = ptr + i * 6;
                    bars.emplace_back(symbol,
                        static_cast<int64_t>(row[0]),
                        row[1], row[2], row[3], row[4], row[5]);
                }
                self.add_data(symbol, bars);
            },
            py::arg("symbol"), py::arg("data"))

        .def("set_strategy",         &BacktestEngine::set_strategy,
             py::arg("strategy"), py::keep_alive<1, 2>())
        .def("run",                  &BacktestEngine::run)
        .def("get_total_pnl",        &BacktestEngine::get_total_pnl)
        .def("get_total_fees",       &BacktestEngine::get_total_fees)
        .def("get_trade_pnls",        &BacktestEngine::get_trade_pnls)
        .def("get_execution_engine", &BacktestEngine::get_execution_engine, py::arg("symbol"))
        .def("set_position_sizer",   &BacktestEngine::set_position_sizer,  py::arg("sizer"))
        .def("get_position_sizer",   &BacktestEngine::get_position_sizer)
        .def("get_equity_curve",     &BacktestEngine::get_equity_curve)
        .def("get_timestamps",       &BacktestEngine::get_timestamps)
        .def("set_risk_limits",      &BacktestEngine::set_risk_limits)
        .def("get_risk_limits",      &BacktestEngine::get_risk_limits)
        .def("get_risk_manager",     &BacktestEngine::get_risk_manager)
        .def("get_portfolio_context", &BacktestEngine::get_portfolio_context)
        .def("set_bars_per_year",    &BacktestEngine::set_bars_per_year,    py::arg("bars_per_year"))
        .def("get_bars_per_year",    &BacktestEngine::get_bars_per_year)
        .def("set_volatility_params", &BacktestEngine::set_volatility_params,
             py::arg("default_vol"), py::arg("stop_distance"), py::arg("lookback"))
        .def("configure_market_maker", &BacktestEngine::configure_market_maker,
             py::arg("levels"), py::arg("spread"), py::arg("depth"));

    // ============================================================================
    // POSITION SIZING
    // ============================================================================

    py::class_<PositionSizingContext>(m, "PositionSizingContext")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("signal_strength")    = 1.0,
             py::arg("current_capital")    = 100000.0,
             py::arg("current_price")      = 100.0,
             py::arg("current_position")   = 0.0,
             py::arg("portfolio_volatility") = 0.02,
             py::arg("stop_loss_distance") = 0.05)
        .def_readwrite("signal_strength",     &PositionSizingContext::signal_strength)
        .def_readwrite("current_capital",     &PositionSizingContext::current_capital)
        .def_readwrite("current_price",       &PositionSizingContext::current_price)
        .def_readwrite("current_position",    &PositionSizingContext::current_position)
        .def_readwrite("portfolio_volatility",&PositionSizingContext::portfolio_volatility)
        .def_readwrite("stop_loss_distance",  &PositionSizingContext::stop_loss_distance);

    py::class_<PositionSizer, std::shared_ptr<PositionSizer>>(m, "PositionSizer")
        .def("calculate_size",        &PositionSizer::calculate_size,     py::arg("context"))
        .def("get_name",              &PositionSizer::get_name)
        .def("set_max_position_size", &PositionSizer::set_max_position_size, py::arg("max_size"))
        .def("set_min_position_size", &PositionSizer::set_min_position_size, py::arg("min_size"))
        .def("set_max_leverage",      &PositionSizer::set_max_leverage,   py::arg("max_leverage"));

    py::class_<FixedPercentage, PositionSizer, std::shared_ptr<FixedPercentage>>(m, "FixedPercentage")
        .def(py::init<double>(), py::arg("percentage") = 0.1);

    py::class_<RiskBased, PositionSizer, std::shared_ptr<RiskBased>>(m, "RiskBased")
        .def(py::init<double>(), py::arg("risk_per_trade") = 0.01);

    py::class_<KellyCriterion, PositionSizer, std::shared_ptr<KellyCriterion>>(m, "KellyCriterion")
        .def(py::init<double, double, double, double>(),
             py::arg("win_rate"), py::arg("avg_win"),
             py::arg("avg_loss"), py::arg("fraction") = 1.0);

    py::class_<EqualWeight, PositionSizer, std::shared_ptr<EqualWeight>>(m, "EqualWeight")
        .def(py::init<int>(), py::arg("num_positions"));

    py::class_<VolatilityTargeting, PositionSizer, std::shared_ptr<VolatilityTargeting>>(m, "VolatilityTargeting")
        .def(py::init<double>(), py::arg("target_volatility") = 0.15);

    py::class_<FixedShares, PositionSizer, std::shared_ptr<FixedShares>>(m, "FixedShares")
        .def(py::init<double>(), py::arg("num_shares"));

    // ============================================================================
    // RISK MANAGEMENT
    // ============================================================================

    py::enum_<RiskCheckResult>(m, "RiskCheckResult")
        .value("APPROVED",                  RiskCheckResult::APPROVED)
        .value("REJECTED_POSITION_LIMIT",   RiskCheckResult::REJECTED_POSITION_LIMIT)
        .value("REJECTED_LEVERAGE_LIMIT",   RiskCheckResult::REJECTED_LEVERAGE_LIMIT)
        .value("REJECTED_CAPITAL_LIMIT",    RiskCheckResult::REJECTED_CAPITAL_LIMIT)
        .value("REJECTED_LOSS_LIMIT",       RiskCheckResult::REJECTED_LOSS_LIMIT)
        .value("REJECTED_ORDER_SIZE",       RiskCheckResult::REJECTED_ORDER_SIZE);

    py::class_<RiskCheckResponse>(m, "RiskCheckResponse")
        .def(py::init<>())
        .def_readwrite("result", &RiskCheckResponse::result)
        .def_readwrite("reason", &RiskCheckResponse::reason)
        .def("is_approved",      &RiskCheckResponse::is_approved);

    py::class_<RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("max_position_pct", &RiskLimits::max_position_pct)
        .def_readwrite("max_leverage",     &RiskLimits::max_leverage)
        .def_readwrite("max_loss_pct",     &RiskLimits::max_loss_pct)
        .def_readwrite("max_order_value",  &RiskLimits::max_order_value)
        .def_readwrite("enabled",          &RiskLimits::enabled)
        .def("validate",                   &RiskLimits::validate);

    py::class_<RiskManager, std::shared_ptr<RiskManager>>(m, "RiskManager")
        .def(py::init<>())
        .def(py::init<const RiskLimits&>())
        .def("set_capital",  &RiskManager::set_capital)
        // price has a C++ default of 0.0; pybind11 requires it to be re-declared
        // here or Python callers with 2 args get a "too few arguments" error.
        .def("set_position", &RiskManager::set_position,
             py::arg("symbol"), py::arg("quantity"), py::arg("price") = 0.0)
        .def("get_position", &RiskManager::get_position,  py::arg("symbol"))
        .def("set_limits",   &RiskManager::set_limits)
        .def("get_limits",   &RiskManager::get_limits)
        .def("check_order",  &RiskManager::check_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"), py::arg("price"))
        // .def("update_position",       &RiskManager::update_position)
        .def("reset",                 &RiskManager::reset)
        .def("get_all_positions",     &RiskManager::get_all_positions)
        .def("calculate_total_exposure", static_cast<double(RiskManager::*)() const>(
             &RiskManager::calculate_total_exposure));

    // ============================================================================
    // PORTFOLIO CONTEXT
    // ============================================================================

    py::class_<PortfolioContext, std::shared_ptr<PortfolioContext>>(m, "PortfolioContext")
        .def(py::init<double>(), py::arg("initial_capital"))
        .def("get_cash",                  &PortfolioContext::get_cash)
        .def("get_initial_capital",       &PortfolioContext::get_initial_capital)
        .def("get_position",              &PortfolioContext::get_position,       py::arg("symbol"))
        .def("get_price",                 &PortfolioContext::get_price,          py::arg("symbol"))
        .def("get_position_value",        &PortfolioContext::get_position_value, py::arg("symbol"))
        .def("get_total_position_value",  &PortfolioContext::get_total_position_value)
        .def("get_portfolio_value",       &PortfolioContext::get_portfolio_value)
        .def("get_leverage",              &PortfolioContext::get_leverage)
        .def("get_position_weight",       &PortfolioContext::get_position_weight, py::arg("symbol"))
        .def("get_all_positions",         &PortfolioContext::get_all_positions)
        .def("get_all_prices",            &PortfolioContext::get_all_prices)
        .def("num_positions",             &PortfolioContext::num_positions)
        .def("has_position",              &PortfolioContext::has_position,       py::arg("symbol"));

    // ============================================================================
    // UTILITY
    // ============================================================================

    m.def("hello",   []() { return "Hello from QuantCore C++!"; });
    m.def("version", []() { return "0.1.8.8"; });

    m.attr("__version__") = "0.1.8.8";
}