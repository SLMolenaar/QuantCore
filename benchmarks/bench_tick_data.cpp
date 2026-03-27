/**
 * bench_tick_data.cpp
 *
 * Sections
 * --------
 *  1. Tick throughput        – ticks/s through the engine, varying tick counts
 *  2. MM throttle comparison – no throttle vs 100ms vs 1s refresh interval
 *  3. Aggregation            – ticks/s for aggregate_to_bars, varying bar durations
 *  4. Tick vs bar            – engine throughput: equivalent tick dataset vs bar dataset
 *  5. Equity snapshot cost   – no snapshot vs 1-min vs 1-hr interval
 *  6. Target verification
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "backtesting/backtest_engine.h"
#include "backtesting/tick_data.h"
#include "backtesting/tick_data_loader.h"
#include "backtesting/bar_data.h"
#include "strategies/buy_and_hold.h"
#include "strategies/mean_reversion.h"

using namespace quantcore;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// ============================================================================
// Synthetic data
// ============================================================================

// Generate ticks spaced interval_ns apart, with a simple sine-wave price walk.
static TickSeries make_ticks(const std::string& symbol, size_t n,
                              int64_t interval_ns  = 1'000'000'000LL, // 1s default
                              double  start_price  = 100.0)
{
    TickSeries ticks;
    ticks.reserve(n);

    constexpr int64_t start_ns = 1'577'836'800'000'000'000LL; // 2020-01-01

    double price = start_price;
    for (size_t i = 0; i < n; ++i) {
        price *= 1.0 + 0.00005 * std::sin(static_cast<double>(i) * 0.01);
        Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
        ticks.emplace_back(symbol,
                           start_ns + static_cast<int64_t>(i) * interval_ns,
                           price, 100.0, side);
    }
    return ticks;
}

// One bar per day, same price walk as make_ticks aggregated to daily OHLCV.
static BarSeries make_bars(const std::string& symbol, size_t n,
                            double start_price = 100.0)
{
    BarSeries bars;
    bars.reserve(n);

    constexpr int64_t day_ns   = 86'400'000'000'000LL;
    constexpr int64_t start_ns = 1'577'836'800'000'000'000LL;

    double price = start_price;
    for (size_t i = 0; i < n; ++i) {
        price *= 1.0 + 0.0002 + 0.01 * std::sin(static_cast<double>(i) * 0.05);
        bars.emplace_back(symbol,
                          start_ns + static_cast<int64_t>(i) * day_ns,
                          price, price * 1.005, price * 0.995, price,
                          1'000'000.0);
    }
    return bars;
}

// ============================================================================
// Timing helpers
// ============================================================================

// Returns median wall-clock seconds over `timed` runs (after `warmup`).
template<typename Fn>
static double time_median(Fn fn, int warmup = 2, int timed = 10)
{
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times;
    times.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        auto t0 = Clock::now();
        fn();
        times.push_back(Seconds(Clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// ============================================================================
// Formatting
// ============================================================================

static void sep(char c = '-', int w = 72) { std::cout << std::string(w, c) << '\n'; }

static void col_headers_ticks()
{
    std::cout << std::left  << std::setw(36) << "Scenario"
              << std::right << std::setw(12) << "Ticks"
              << std::setw(16) << "Throughput"
              << std::setw(12) << "Wall time"
              << '\n';
    sep();
}

static void print_tick_row(const std::string& label,
                            size_t n_ticks, double elapsed_s)
{
    double ticks_per_s = static_cast<double>(n_ticks) / elapsed_s;
    std::cout << std::left  << std::setw(36) << label
              << std::right << std::setw(10) << n_ticks << "  "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (ticks_per_s / 1'000.0) << " K ticks/s"
              << "    "
              << std::setprecision(1) << std::setw(7) << (elapsed_s * 1000.0) << " ms\n";
}

// ============================================================================
// 1. Tick throughput, single asset, BuyAndHold
// ============================================================================

static void bench_tick_throughput()
{
    std::cout << "1. Tick throughput \xe2\x80\x94 single asset (BuyAndHold)\n";
    col_headers_ticks();

    // 1-second ticks; varying total counts
    for (size_t n : {1'000UL, 10'000UL, 60'000UL, 360'000UL, 1'000'000UL})
    {
        auto ticks = make_ticks("ASSET", n, 1'000'000'000LL);

        auto elapsed = time_median([&] {
            BacktestEngine eng(100'000.0);
            eng.add_tick_data("ASSET", ticks);
            eng.set_strategy(std::make_shared<BuyAndHold>());
            eng.run();
        });

        std::string label = std::to_string(n / 1000) + "K ticks (1s interval)";
        print_tick_row(label, n, elapsed);
    }
    std::cout << '\n';
}

// ============================================================================
// 2. MM throttle comparison
// ============================================================================

static void bench_mm_throttle()
{
    std::cout << "2. Market maker throttle \xe2\x80\x94 10K ticks, varying refresh interval\n";
    sep();

    constexpr size_t N = 10'000;
    auto ticks = make_ticks("ASSET", N, 100'000'000LL); // 100ms ticks

    std::vector<std::pair<std::string, int64_t>> configs = {
        {"No throttle (every tick)",  0LL},
        {"100ms refresh interval",    100'000'000LL},
        {"1s refresh interval",       1'000'000'000LL},
        {"10s refresh interval",      10'000'000'000LL},
    };

    std::cout << std::left  << std::setw(36) << "Config"
              << std::right << std::setw(16) << "Throughput"
              << std::setw(12) << "Wall time"
              << '\n';
    sep();

    for (const auto& [label, interval] : configs) {
        auto elapsed = time_median([&] {
            BacktestEngine eng(100'000.0);
            eng.add_tick_data("ASSET", ticks);
            eng.set_mm_refresh_interval(interval);
            eng.set_strategy(std::make_shared<BuyAndHold>());
            eng.run();
        });

        double ticks_per_s = static_cast<double>(N) / elapsed;
        std::cout << std::left  << std::setw(36) << label
                  << std::right << std::fixed    << std::setprecision(2)
                  << std::setw(8) << (ticks_per_s / 1'000.0) << " K ticks/s"
                  << "    "
                  << std::setprecision(1) << std::setw(7) << (elapsed * 1000.0) << " ms\n";
    }
    std::cout << '\n';
}

// ============================================================================
// 3. Aggregation throughput, ticks/s for aggregate_to_bars
// ============================================================================

static void bench_aggregation()
{
    std::cout << "3. Aggregation throughput \xe2\x80\x94 aggregate_to_bars\n";
    sep();

    constexpr size_t N = 1'000'000; // 1M ticks
    auto ticks = make_ticks("ASSET", N, 1'000'000'000LL);

    std::vector<std::pair<std::string, int64_t>> durations = {
        {"1-second bars",  1'000'000'000LL},
        {"1-minute bars",  60'000'000'000LL},
        {"1-hour bars",    3'600'000'000'000LL},
        {"1-day bars",     86'400'000'000'000LL},
    };

    std::cout << std::left  << std::setw(36) << "Bar duration"
              << std::right << std::setw(10) << "Bars out"
              << std::setw(16) << "Ticks/s"
              << std::setw(12) << "Wall time"
              << '\n';
    sep();

    for (const auto& [label, dur_ns] : durations) {
        BarSeries bars;
        auto elapsed = time_median([&] {
            bars = TickDataLoader::aggregate_to_bars(ticks, dur_ns);
        });

        double ticks_per_s = static_cast<double>(N) / elapsed;
        std::cout << std::left  << std::setw(36) << label
                  << std::right << std::setw(10) << bars.size()
                  << std::fixed << std::setprecision(2)
                  << "    " << std::setw(8) << (ticks_per_s / 1'000'000.0) << " M ticks/s"
                  << "    "
                  << std::setprecision(1) << std::setw(7) << (elapsed * 1000.0) << " ms\n";
    }
    std::cout << '\n';
}

// ============================================================================
// 4. Tick vs bar, equivalent dataset comparison
// ============================================================================

static void bench_tick_vs_bar()
{
    std::cout << "4. Tick vs bar \xe2\x80\x94 equivalent data, BuyAndHold\n";
    std::cout << "   (252 bars  vs  252 * ticks_per_bar ticks aggregating to the same period)\n";
    col_headers_ticks();

    // bar baseline
    constexpr size_t N_BARS = 252;
    auto bars = make_bars("ASSET", N_BARS);

    auto bar_elapsed = time_median([&] {
        BacktestEngine eng(100'000.0);
        eng.add_data("ASSET", bars);
        eng.set_strategy(std::make_shared<BuyAndHold>());
        eng.run();
    });

    std::cout << std::left  << std::setw(36) << "Bar mode (252 daily bars)"
              << std::right << std::setw(10) << N_BARS << "  "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (static_cast<double>(N_BARS) / bar_elapsed / 1'000.0) << " K bars/s"
              << "    "
              << std::setprecision(1) << std::setw(7) << (bar_elapsed * 1000.0) << " ms\n";

    // tick equivalents: 1 tick/day, 10 ticks/day, 390 ticks/day (1-min bars, US session)
    for (size_t tpb : {1UL, 10UL, 390UL}) {
        size_t n_ticks = N_BARS * tpb;
        int64_t interval_ns = 86'400'000'000'000LL / static_cast<int64_t>(tpb);
        auto ticks = make_ticks("ASSET", n_ticks, interval_ns);

        auto elapsed = time_median([&] {
            BacktestEngine eng(100'000.0);
            eng.add_tick_data("ASSET", ticks);
            eng.set_mm_refresh_interval(86'400'000'000'000LL); // refresh once per day
            eng.set_strategy(std::make_shared<BuyAndHold>());
            eng.run();
        });

        std::string label = "Tick mode (" + std::to_string(tpb) + " ticks/day, "
                          + std::to_string(n_ticks) + " total)";
        print_tick_row(label, n_ticks, elapsed);
    }
    std::cout << '\n';
}

// ============================================================================
// 5. Equity snapshot cost
// ============================================================================

static void bench_snapshot_cost()
{
    std::cout << "5. Equity snapshot cost \xe2\x80\x94 10K ticks, varying snapshot interval\n";
    sep();

    constexpr size_t N = 10'000;
    auto ticks = make_ticks("ASSET", N, 1'000'000'000LL); // 1s ticks

    std::vector<std::pair<std::string, int64_t>> configs = {
        {"Every tick (interval = 0)",  0LL},
        {"Every 1 minute",             60'000'000'000LL},
        {"Every 1 hour",               3'600'000'000'000LL},
    };

    std::cout << std::left  << std::setw(36) << "Snapshot interval"
              << std::right << std::setw(12) << "Equity pts"
              << std::setw(16) << "Throughput"
              << std::setw(12) << "Wall time"
              << '\n';
    sep();

    for (const auto& [label, interval] : configs) {
        std::vector<double> curve;
        auto elapsed = time_median([&] {
            BacktestEngine eng(100'000.0);
            eng.add_tick_data("ASSET", ticks);
            eng.set_mm_refresh_interval(1'000'000'000LL);
            eng.set_equity_snapshot_interval(interval);
            eng.set_strategy(std::make_shared<BuyAndHold>());
            eng.run();
            curve = eng.get_equity_curve();
        });

        double ticks_per_s = static_cast<double>(N) / elapsed;
        std::cout << std::left  << std::setw(36) << label
                  << std::right << std::setw(10) << curve.size()
                  << std::fixed << std::setprecision(2)
                  << "    " << std::setw(8) << (ticks_per_s / 1'000.0) << " K ticks/s"
                  << "    "
                  << std::setprecision(1) << std::setw(7) << (elapsed * 1000.0) << " ms\n";
    }
    std::cout << '\n';
}

// ============================================================================
// 6. Target verification
// ============================================================================

static void bench_targets()
{
    sep('=');
    std::cout << "  Target Verification\n";
    sep('=');

    auto check = [](const std::string& label, bool pass) {
        std::cout << (pass ? "  PASS  " : "  FAIL  ") << label << '\n';
    };

    // 10K ticks with 1s throttle should complete in < 50ms
    {
        constexpr size_t N = 10'000;
        auto ticks = make_ticks("ASSET", N, 1'000'000'000LL);
        auto elapsed = time_median([&] {
            BacktestEngine eng(100'000.0);
            eng.add_tick_data("ASSET", ticks);
            eng.set_mm_refresh_interval(1'000'000'000LL);
            eng.set_strategy(std::make_shared<BuyAndHold>());
            eng.run();
        }, 2, 20);
        double ms = elapsed * 1000.0;
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << ms << "ms";
        check("10K ticks (1s MM throttle) p50 < 50ms  (got " + s.str() + ")", ms < 50.0);
    }

    // aggregate_to_bars: 1M ticks -> 1-min bars in < 100ms
    {
        constexpr size_t N = 1'000'000;
        auto ticks   = make_ticks("ASSET", N, 1'000'000'000LL);
        auto elapsed = time_median([&] {
            TickDataLoader::aggregate_to_bars(ticks, 60'000'000'000LL);
        }, 2, 10);
        double ms = elapsed * 1000.0;
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << ms << "ms";
        check("1M ticks aggregated to 1-min bars < 100ms  (got " + s.str() + ")", ms < 100.0);
    }

    sep('=');
}

// ============================================================================
// main
// ============================================================================

int main()
{
    sep('=');
    std::cout << "  QuantCore \xe2\x80\x94 Tick Data Benchmark\n";
    sep('=');
    std::cout << '\n';

    bench_tick_throughput();
    bench_mm_throttle();
    bench_aggregation();
    bench_tick_vs_bar();
    bench_snapshot_cost();
    bench_targets();

    std::cout << '\n';
    return 0;
}
