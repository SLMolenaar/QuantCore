/**
 * QuantCore — Backtest Engine Benchmark
 *
 * Measures throughput and wall-clock time of the full event-driven loop:
 *   market data → strategy → signal → order → fill → equity update
 *
 * Build (Release):
 *   g++ -std=c++20 -O3 -DNDEBUG -I../cpp -I../cpp/orderbook \
 *       bench_backtest_engine.cpp -o bench_backtest_engine
 *
 * Run:
 *   ./bench_backtest_engine
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "backtesting/backtest_engine.h"
#include "backtesting/bar_data.h"
#include "strategies/buy_and_hold.h"
#include "strategies/mean_reversion.h"
#include "strategies/sma_crossover.h"

using namespace quantcore;
using Clock     = std::chrono::high_resolution_clock;
using Seconds   = std::chrono::duration<double>;
using Nanos     = std::chrono::nanoseconds;

// ============================================================================
// Data generation helpers
// ============================================================================

static BarSeries make_bars(const std::string& symbol, size_t n,
                            double start_price = 100.0)
{
    BarSeries bars;
    bars.reserve(n);

    constexpr int64_t day_ns     = 86'400'000'000'000LL;
    const     int64_t start_ns   = 1'577'836'800'000'000'000LL; // 2020-01-01

    double price = start_price;
    for (size_t i = 0; i < n; ++i) {
        // simple geometric Brownian motion
        double ret   = 0.0002 + 0.01 * std::sin(static_cast<double>(i) * 0.05);
        price       *= (1.0 + ret);
        double high  = price * 1.005;
        double low   = price * 0.995;
        bars.emplace_back(symbol, start_ns + static_cast<int64_t>(i) * day_ns,
                          price, high, low, price, 1'000'000.0);
    }
    return bars;
}

// ============================================================================
// Timing utilities
// ============================================================================

struct BenchResult {
    size_t   n_bars;
    size_t   n_symbols;
    double   elapsed_s;
    double   events_per_sec;
    double   bars_per_sec;
    double   ms_per_bar;
};

static BenchResult run_bench(std::shared_ptr<Strategy> strategy,
                              const std::map<std::string, BarSeries>& data,
                              size_t n_bars, size_t n_symbols,
                              int warmup_runs = 2, int timed_runs = 5)
{
    // warmup — avoid cold-start JIT / branch predictor effects
    for (int w = 0; w < warmup_runs; ++w) {
        BacktestEngine eng(100'000.0);
        for (auto& [sym, bars] : data) eng.add_data(sym, bars);
        eng.set_strategy(strategy);
        eng.run();
        strategy->reset();
    }

    // timed runs
    std::vector<double> times;
    times.reserve(timed_runs);

    for (int r = 0; r < timed_runs; ++r) {
        BacktestEngine eng(100'000.0);
        for (auto& [sym, bars] : data) eng.add_data(sym, bars);
        eng.set_strategy(strategy);

        auto t0 = Clock::now();
        eng.run();
        auto t1 = Clock::now();

        times.push_back(Seconds(t1 - t0).count());
        strategy->reset();
    }

    double median_s = [&] {
        auto t = times;
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }();

    // Each bar generates 1 market data event; a fill path adds signal + order + fill (3 more).
    // Use conservative estimate of 2 events/bar on average across mixed strategies.
    double total_events = static_cast<double>(n_bars * n_symbols) * 2.0;

    return {
        n_bars,
        n_symbols,
        median_s,
        total_events / median_s,
        static_cast<double>(n_bars * n_symbols) / median_s,
        median_s / static_cast<double>(n_bars * n_symbols) * 1000.0,
    };
}

// ============================================================================
// Output helpers
// ============================================================================

static void print_separator(char c = '-', int width = 72)
{
    std::cout << std::string(width, c) << '\n';
}

static void print_header()
{
    print_separator('=');
    std::cout << "  QuantCore — Backtest Engine Benchmark\n";
    print_separator('=');
    std::cout << '\n';
}

static void print_result(const std::string& label, const BenchResult& r)
{
    std::cout << std::left  << std::setw(38) << label
              << std::right << std::setw(8)  << r.n_bars   << " bars   "
              << std::setw(8) << std::fixed << std::setprecision(1)
              << (r.events_per_sec / 1'000'000.0)          << " M ev/s   "
              << std::setw(7) << std::fixed << std::setprecision(1)
              << (r.bars_per_sec  / 1'000.0)               << " K bars/s   "
              << std::setw(7) << std::fixed << std::setprecision(3)
              << r.elapsed_s * 1000.0                      << " ms\n";
}

static void print_column_headers()
{
    std::cout << std::left  << std::setw(38) << "Scenario"
              << std::right << std::setw(8)  << "Bars"    << "         "
              << std::setw(8)                << "Throughput"  << "          "
              << std::setw(7)                << "Speed"       << "        "
              << std::setw(7)                << "Wall time\n";
    print_separator();
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    print_header();

    // -------------------------------------------------------------------------
    // 1. Throughput — single asset, varying bar counts
    // -------------------------------------------------------------------------
    std::cout << "1. Throughput — single asset (BuyAndHold)\n";
    print_column_headers();

    for (size_t n : {252UL, 1'260UL, 2'520UL, 12'600UL, 63'000UL, 252'000UL})
    {
        auto bars = make_bars("ASSET", n);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};
        auto strat = std::make_shared<BuyAndHold>();

        auto label = std::to_string(n / 252) + " yr"
                   + (n % 252 ? " (" + std::to_string(n) + " bars)" : "");
        print_result(label, run_bench(strat, data, n, 1));
    }

    std::cout << '\n';

    // -------------------------------------------------------------------------
    // 2. Strategy comparison — 5 years, single asset
    // -------------------------------------------------------------------------
    constexpr size_t FIVE_YEARS = 1'260;

    std::cout << "2. Strategy comparison — 5 years, 1 asset\n";
    print_column_headers();

    {
        auto bars = make_bars("ASSET", FIVE_YEARS);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};

        struct Case { std::string name; std::shared_ptr<Strategy> strat; };
        std::vector<Case> cases = {
            {"BuyAndHold",            std::make_shared<BuyAndHold>()},
            {"SMACrossover(20/100)",   std::make_shared<SMACrossover>(20, 100)},
            {"MeanReversion(20,1.5)", std::make_shared<MeanReversion>(20, 1.5, 0.5)},
        };

        for (auto& c : cases)
            print_result(c.name, run_bench(c.strat, data, FIVE_YEARS, 1));
    }

    std::cout << '\n';

    // -------------------------------------------------------------------------
    // 3. Multi-asset scaling — 5 years, 1 → 10 symbols
    // -------------------------------------------------------------------------
    std::cout << "3. Multi-asset scaling — 5 years, MeanReversion\n";
    print_column_headers();

    for (size_t n_sym : {1UL, 2UL, 5UL, 10UL})
    {
        std::map<std::string, BarSeries> data;
        for (size_t s = 0; s < n_sym; ++s) {
            std::string sym = "SYM" + std::to_string(s);
            data[sym] = make_bars(sym, FIVE_YEARS, 100.0 + static_cast<double>(s) * 5.0);
        }

        auto strat = std::make_shared<MeanReversion>(20, 1.5, 0.5);
        auto label = std::to_string(n_sym) + " symbol" + (n_sym > 1 ? "s" : "");
        print_result(label, run_bench(strat, data, FIVE_YEARS, n_sym));
    }

    std::cout << '\n';

    // -------------------------------------------------------------------------
    // 4. Target verification
    // -------------------------------------------------------------------------
    print_separator('=');
    std::cout << "  Target Verification\n";
    print_separator('=');

    {
        constexpr size_t N = 252'000; // ~1000 years of daily data
        auto bars = make_bars("ASSET", N);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};
        auto strat = std::make_shared<BuyAndHold>();
        auto r = run_bench(strat, data, N, 1, 1, 3);

        auto check = [](const std::string& label, bool pass) {
            std::cout << (pass ? "  PASS  " : "  FAIL  ") << label << '\n';
        };

        check(">1M events/sec throughput",     r.events_per_sec >= 1'000'000.0);
        check("1 year backtest < 10ms",        run_bench(strat, {{"ASSET", make_bars("ASSET", 252)}},
                                                         252, 1, 1, 3).elapsed_s * 1000.0 < 10.0);
    }

    print_separator('=');
    std::cout << '\n';

    return 0;
}
