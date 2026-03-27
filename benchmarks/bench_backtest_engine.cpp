/**
 * bench_backtest_engine.cpp
 *
 * Sections
 * --------
 *  1. Throughput          – single asset, varying bar counts
 *  2. Strategy comparison – 5 years, 1 asset, 3 strategies
 *  3. Multi-asset scaling – 5 years, 1→10 symbols
 *  4. Order book isolation – raw AddOrder / CancelOrder ops/s (no engine overhead)
 *  5. Latency distribution – p50 / p95 / p99 / max per strategy
 *  6. Memory stability     – RSS before vs after 1000-year run (Linux only)
 *  7. Target verification
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "backtesting/backtest_engine.h"
#include "backtesting/bar_data.h"
#include "orderbook/Orderbook.h"
#include "orderbook/Order.h"
#include "orderbook/Types.h"
#include "strategies/buy_and_hold.h"
#include "strategies/mean_reversion.h"
#include "strategies/sma_crossover.h"

using namespace quantcore;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// ============================================================================
// Synthetic data
// ============================================================================

static BarSeries make_bars(const std::string& symbol, size_t n,
                            double start_price = 100.0)
{
    BarSeries bars;
    bars.reserve(n);

    constexpr int64_t day_ns   = 86'400'000'000'000LL;
    constexpr int64_t start_ns = 1'577'836'800'000'000'000LL; // 2020-01-01

    double price = start_price;
    for (size_t i = 0; i < n; ++i) {
        double ret  = 0.0002 + 0.01 * std::sin(static_cast<double>(i) * 0.05);
        price      *= (1.0 + ret);
        bars.emplace_back(symbol,
                          start_ns + static_cast<int64_t>(i) * day_ns,
                          price, price * 1.005, price * 0.995, price,
                          1'000'000.0);
    }
    return bars;
}

// ============================================================================
// Memory (Linux only)
// ============================================================================

static size_t rss_kb()
{
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            size_t v = 0;
            std::sscanf(line.c_str(), "VmRSS: %zu kB", &v);
            return v;
        }
#endif
    return 0;
}

// ============================================================================
// Timing helpers
// ============================================================================

// Returns per-run wall-clock times in seconds (after warmup).
static std::vector<double> collect_timings(
        std::shared_ptr<Strategy> strategy,
        const std::map<std::string, BarSeries>& data,
        int warmup = 2, int timed = 20)
{
    auto run_once = [&] {
        BacktestEngine eng(100'000.0);
        for (const auto& [sym, bars] : data) eng.add_data(sym, bars);
        eng.set_strategy(strategy);
        eng.run();
        strategy->reset();
    };

    for (int i = 0; i < warmup; ++i) run_once();

    std::vector<double> times;
    times.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        auto t0 = Clock::now();
        run_once();
        times.push_back(Seconds(Clock::now() - t0).count());
    }
    return times;
}

// Median of a vector (non-destructive).
static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ============================================================================
// Formatting
// ============================================================================

static void sep(char c = '-', int w = 72) { std::cout << std::string(w, c) << '\n'; }

static void col_headers()
{
    std::cout << std::left  << std::setw(36) << "Scenario"
              << std::right << std::setw(9)  << "Bars"
              << std::setw(14) << "Throughput"
              << std::setw(14) << "Speed"
              << std::setw(12) << "Wall time"
              << '\n';
    sep();
}

static void print_row(const std::string& label,
                      size_t n_bars, size_t n_syms, double elapsed_s)
{
    double total_events = static_cast<double>(n_bars * n_syms) * 2.0; // market_data + fill
    double ev_per_s     = total_events / elapsed_s;
    double bars_per_s   = static_cast<double>(n_bars * n_syms) / elapsed_s;

    std::cout << std::left  << std::setw(36) << label
              << std::right << std::setw(7)  << (n_bars * n_syms) << " bars"
              << "    "
              << std::fixed << std::setprecision(1)
              << std::setw(5) << (ev_per_s / 1'000'000.0) << " M ev/s"
              << "    "
              << std::setw(7) << (bars_per_s / 1'000.0) << " K bars/s"
              << "    "
              << std::setprecision(3) << std::setw(7) << (elapsed_s * 1000.0) << " ms\n";
}

// ============================================================================
// 1. Throughput, single asset
// ============================================================================

static void bench_throughput()
{
    std::cout << "1. Throughput \xe2\x80\x94 single asset (BuyAndHold)\n";
    col_headers();

    for (size_t n : {252UL, 1'260UL, 2'520UL, 12'600UL, 63'000UL, 252'000UL})
    {
        auto bars  = make_bars("ASSET", n);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};
        auto strat = std::make_shared<BuyAndHold>();
        auto times = collect_timings(strat, data, 2, 20);
        auto label = std::to_string(n / 252) + " yr";
        print_row(label, n, 1, median(times));
    }
    std::cout << '\n';
}

// ============================================================================
// 2. Strategy comparison, 5 years, 1 asset
// ============================================================================

static void bench_strategies()
{
    std::cout << "2. Strategy comparison \xe2\x80\x94 5 years, 1 asset\n";
    col_headers();

    constexpr size_t N = 1'260;
    auto bars = make_bars("ASSET", N);
    std::map<std::string, BarSeries> data = {{"ASSET", bars}};

    struct Case { std::string name; std::shared_ptr<Strategy> strat; };
    for (auto& [name, strat] : std::vector<Case>{
        {"BuyAndHold",             std::make_shared<BuyAndHold>()},
        {"SMACrossover(20/100)",    std::make_shared<SMACrossover>(20, 100)},
        {"MeanReversion(20,1.5)",   std::make_shared<MeanReversion>(20, 1.5, 0.5)},
    }) {
        auto times = collect_timings(strat, data, 2, 20);
        print_row(name, N, 1, median(times));
    }
    std::cout << '\n';
}

// ============================================================================
// 3. Multi-asset scaling, 5 years
// ============================================================================

static void bench_multi_asset()
{
    std::cout << "3. Multi-asset scaling \xe2\x80\x94 5 years, MeanReversion\n";
    col_headers();

    constexpr size_t N = 1'260;
    for (size_t n_sym : {1UL, 2UL, 5UL, 10UL})
    {
        std::map<std::string, BarSeries> data;
        for (size_t s = 0; s < n_sym; ++s) {
            auto sym = "SYM" + std::to_string(s);
            data[sym] = make_bars(sym, N, 100.0 + s * 5.0);
        }
        auto strat = std::make_shared<MeanReversion>(20, 1.5, 0.5);
        auto times = collect_timings(strat, data, 2, 10);
        auto label = std::to_string(n_sym) + " symbol" + (n_sym > 1 ? "s" : "");
        print_row(label, N, n_sym, median(times));
    }
    std::cout << '\n';
}

// ============================================================================
// 4. Order book isolation, raw AddOrder / CancelOrder
// ============================================================================

static void bench_orderbook()
{
    std::cout << "4. Order book isolation \xe2\x80\x94 raw ops/s\n";
    sep();

    auto measure = [](const std::string& label, int n_ops, auto fn) {
        // warmup
        for (int i = 0; i < 500; ++i) fn(i);

        auto t0 = Clock::now();
        for (int i = 0; i < n_ops; ++i) fn(i);
        double secs = Seconds(Clock::now() - t0).count();
        double ops  = static_cast<double>(n_ops * 2) / secs; // 2 ops per iteration

        std::cout << std::left  << std::setw(36) << label
                  << std::right << std::fixed    << std::setprecision(2)
                  << "  " << ops / 1'000'000.0 << " M ops/s"
                  << "  (" << n_ops << " pairs, "
                  << std::setprecision(1) << secs * 1000.0 << " ms)\n";
    };

    // Scenario A: add + cancel (market-maker quote refresh pattern)
    {
        Orderbook ob;
        measure("Add + cancel  (MM quote refresh)", 500'000, [&](int i) {
            auto id = static_cast<OrderId>(1'000'000 + i);
            ob.AddOrder(std::make_shared<Order>(
                OrderType::GoodTillCancel, id,
                (i % 2 == 0) ? Side::Buy : Side::Sell,
                10000 + (i % 10) - 5, 100));
            ob.CancelOrder(id);
        });
    }

    // Scenario B: add + match (liquidity-taking pattern)
    {
        Orderbook ob;
        measure("Add + match   (taker sweep)",      300'000, [&](int i) {
            // resting ask
            ob.AddOrder(std::make_shared<Order>(
                OrderType::GoodTillCancel,
                static_cast<OrderId>(2'000'000 + i * 2),
                Side::Sell, 10000, 1));
            // aggressive buy, matches and clears
            ob.AddOrder(std::make_shared<Order>(
                OrderType::GoodTillCancel,
                static_cast<OrderId>(2'000'000 + i * 2 + 1),
                Side::Buy, 10000, 1));
        });
    }

    std::cout << '\n';
}

// ============================================================================
// 5. Latency distribution, 1-year run, 100 samples
// ============================================================================

static void bench_latency()
{
    std::cout << "5. Latency distribution \xe2\x80\x94 1 year, 100 samples\n";
    sep();

    constexpr size_t N = 252;
    auto bars = make_bars("ASSET", N);
    std::map<std::string, BarSeries> data = {{"ASSET", bars}};

    struct Case { std::string name; std::shared_ptr<Strategy> strat; };
    for (auto& [name, strat] : std::vector<Case>{
        {"BuyAndHold",            std::make_shared<BuyAndHold>()},
        {"SMACrossover(20/100)",   std::make_shared<SMACrossover>(20, 100)},
        {"MeanReversion(20,1.5)",  std::make_shared<MeanReversion>(20, 1.5, 0.5)},
    }) {
        auto raw = collect_timings(strat, data, 5, 100);
        auto t   = raw;
        std::sort(t.begin(), t.end());
        size_t n = t.size();

        auto ms = [](double s) { return s * 1000.0; };
        std::cout << std::left  << std::setw(36) << name
                  << std::right << std::fixed    << std::setprecision(3)
                  << "  p50=" << std::setw(7) << ms(t[n * 50 / 100]) << "ms"
                  << "  p95=" << std::setw(7) << ms(t[n * 95 / 100]) << "ms"
                  << "  p99=" << std::setw(7) << ms(t[n * 99 / 100]) << "ms"
                  << "  max=" << std::setw(7) << ms(t.back())         << "ms\n";
    }
    std::cout << '\n';
}

// ============================================================================
// 6. Memory stability; 1000-year run
// ============================================================================

static void bench_memory()
{
    std::cout << "6. Memory stability \xe2\x80\x94 1000-year run\n";
    sep();

    constexpr size_t N = 252'000;
    auto bars  = make_bars("ASSET", N);
    std::map<std::string, BarSeries> data = {{"ASSET", bars}};
    auto strat = std::make_shared<BuyAndHold>();

    size_t before = rss_kb();

    BacktestEngine eng(100'000.0);
    eng.add_data("ASSET", bars);
    eng.set_strategy(strat);
    eng.run();

    size_t after = rss_kb();

    if (before > 0) {
        long delta = static_cast<long>(after) - static_cast<long>(before);
        std::cout << "  RSS before : " << before << " kB\n"
                  << "  RSS after  : " << after  << " kB\n"
                  << "  Delta      : " << (delta >= 0 ? "+" : "") << delta
                  << " kB  (" << N / 252 << "-year run, "
                  << N << " bars)\n";
    } else {
        std::cout << "  (memory measurement unavailable on this platform)\n";
    }
    std::cout << '\n';
}

// ============================================================================
// 7. Target verification
// ============================================================================

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? "  PASS  " : "  FAIL  ") << label << '\n';
}

static void bench_targets()
{
    sep('=');
    std::cout << "  Target Verification\n";
    sep('=');

    // --- latency: 1-year p99 < 5ms ---
    {
        auto bars  = make_bars("ASSET", 252);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};
        auto strat = std::make_shared<BuyAndHold>();
        auto t     = collect_timings(strat, data, 2, 50);
        std::sort(t.begin(), t.end());
        double p99_ms = t[t.size() * 99 / 100] * 1000.0;
        check("1 year backtest p99 < 5ms  (got " +
              [&]{ std::ostringstream s; s << std::fixed << std::setprecision(2) << p99_ms << "ms"; return s.str(); }() + ")",
              p99_ms < 5.0);
    }

    // --- order book: add+cancel > 500K ops/s ---
    {
        constexpr int N = 300'000;
        Orderbook ob;
        // warmup
        for (int i = 0; i < 1000; ++i) {
            auto id = static_cast<OrderId>(9'000'000 + i);
            ob.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, id, Side::Buy, 10000, 100));
            ob.CancelOrder(id);
        }
        auto t0  = Clock::now();
        for (int i = 0; i < N; ++i) {
            auto id = static_cast<OrderId>(10'000'000 + i);
            ob.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, id, Side::Buy,
                                                10000 - (i % 50), 100));
            ob.CancelOrder(id);
        }
        double ops = (N * 2.0) / Seconds(Clock::now() - t0).count();
        check(">500K order book ops/sec (MM pattern)", ops >= 500'000.0);
    }

    // --- memory: growth < 10 MB over 1000-year run ---
    {
        constexpr size_t N = 252'000;
        auto bars  = make_bars("ASSET", N);
        std::map<std::string, BarSeries> data = {{"ASSET", bars}};
        auto strat = std::make_shared<BuyAndHold>();
        size_t before = rss_kb();
        BacktestEngine eng(100'000.0);
        eng.add_data("ASSET", bars);
        eng.set_strategy(strat);
        eng.run();
        size_t after = rss_kb();
        if (before > 0) {
            long delta_kb = static_cast<long>(after) - static_cast<long>(before);
            check("Memory growth < 10 MB for 1000-year run", delta_kb < 10'240L);
        } else {
            std::cout << "  SKIP   Memory growth (not Linux)\n";
        }
    }

    sep('=');
}

// ============================================================================
// main
// ============================================================================

int main()
{
    sep('=');
    std::cout << "  QuantCore \xe2\x80\x94 Backtest Engine Benchmark\n";
    sep('=');
    std::cout << '\n';

    bench_throughput();
    bench_strategies();
    bench_multi_asset();
    bench_orderbook();
    bench_latency();
    bench_memory();
    bench_targets();

    std::cout << '\n';
    return 0;
}