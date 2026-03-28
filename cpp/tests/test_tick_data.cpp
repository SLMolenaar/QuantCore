#include <gtest/gtest.h>
#include "backtesting/tick_data.h"
#include "backtesting/tick_data_loader.h"
#include "backtesting/backtest_engine.h"
#include "backtesting/strategy.h"
#include "backtesting/market_data_event.h"

#include <fstream>
#include <filesystem>

using namespace quantcore;

TEST(TickData, ConstructorSetsFields) {
    TickData tick("AAPL", 1000000000LL, 150.0, 10.0, Side::Buy);
    EXPECT_EQ(tick.symbol,          "AAPL");
    EXPECT_EQ(tick.timestamp_ns,    1000000000LL);
    EXPECT_DOUBLE_EQ(tick.price,    150.0);
    EXPECT_DOUBLE_EQ(tick.quantity, 10.0);
    EXPECT_EQ(tick.aggressor_side,  Side::Buy);
}

TEST(TickData, DefaultSideIsBuy) {
    TickData tick("AAPL", 1000000000LL, 150.0, 10.0);
    EXPECT_EQ(tick.aggressor_side, Side::Buy);
}

TEST(TickData, ThrowsOnNonPositivePrice) {
    EXPECT_THROW(TickData("AAPL", 1000000000LL,  0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(TickData("AAPL", 1000000000LL, -1.0, 1.0), std::invalid_argument);
}

TEST(TickData, ThrowsOnZeroQuantity) {
    EXPECT_THROW(TickData("AAPL", 1000000000LL, 100.0,  0.0),   std::invalid_argument);
    EXPECT_THROW(TickData("AAPL", 1000000000LL, 100.0, -1e-9),  std::invalid_argument);
}

namespace {

TickSeries make_ticks(const std::string& sym,
                      int64_t base_ts_ns,
                      int64_t interval_ns,
                      const std::vector<double>& prices,
                      double qty = 100.0)
{
    TickSeries ticks;
    for (size_t i = 0; i < prices.size(); ++i)
        ticks.emplace_back(sym, base_ts_ns + static_cast<int64_t>(i) * interval_ns,
                           prices[i], qty);
    return ticks;
}

} // anonymous namespace

TEST(TickDataLoader, AggregateEmpty) {
    TickSeries empty;
    EXPECT_TRUE(TickDataLoader::aggregate_to_bars(empty, 60'000'000'000LL).empty());
}

TEST(TickDataLoader, AggregateInvalidDuration) {
    auto ticks = make_ticks("X", 0, 1'000'000, {100.0});
    EXPECT_THROW(TickDataLoader::aggregate_to_bars(ticks,  0), std::invalid_argument);
    EXPECT_THROW(TickDataLoader::aggregate_to_bars(ticks, -1), std::invalid_argument);
}

TEST(TickDataLoader, AggregateMixedSymbolsThrows) {
    TickSeries mixed;
    mixed.emplace_back("AAPL", 0LL,             100.0, 10.0);
    mixed.emplace_back("MSFT", 1'000'000'000LL, 200.0, 10.0);
    EXPECT_THROW(TickDataLoader::aggregate_to_bars(mixed, 60'000'000'000LL),
                 std::invalid_argument);
}

TEST(TickDataLoader, AggregateSingleBar) {
    int64_t bar_dur = 60'000'000'000LL;
    int64_t base    = 1'000'000'000'000'000'000LL;
    base = (base / bar_dur) * bar_dur;
    auto ticks = make_ticks("AAPL", base, 10'000'000'000LL, {100.0, 105.0, 98.0, 102.0});

    auto bars = TickDataLoader::aggregate_to_bars(ticks, bar_dur);
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_DOUBLE_EQ(bars[0].open,   100.0);
    EXPECT_DOUBLE_EQ(bars[0].high,   105.0);
    EXPECT_DOUBLE_EQ(bars[0].low,     98.0);
    EXPECT_DOUBLE_EQ(bars[0].close,  102.0);
    EXPECT_DOUBLE_EQ(bars[0].volume, 400.0);
}

TEST(TickDataLoader, AggregateMultipleBars) {
    int64_t bar_dur  = 60'000'000'000LL;
    int64_t tick_gap = 20'000'000'000LL;

    auto ticks = make_ticks("MSFT", 0, tick_gap,
                            {10, 12, 11,   // bar 0
                              9, 13, 10}); // bar 1

    auto bars = TickDataLoader::aggregate_to_bars(ticks, bar_dur);
    ASSERT_EQ(bars.size(), 2u);

    EXPECT_DOUBLE_EQ(bars[0].open,  10.0);
    EXPECT_DOUBLE_EQ(bars[0].high,  12.0);
    EXPECT_DOUBLE_EQ(bars[0].low,   10.0);
    EXPECT_DOUBLE_EQ(bars[0].close, 11.0);

    EXPECT_DOUBLE_EQ(bars[1].open,   9.0);
    EXPECT_DOUBLE_EQ(bars[1].high,  13.0);
    EXPECT_DOUBLE_EQ(bars[1].low,    9.0);
    EXPECT_DOUBLE_EQ(bars[1].close, 10.0);
}

namespace {

std::string write_temp_csv(const std::string& content) {
    auto path = std::filesystem::temp_directory_path() / "quantcore_tick_test.csv";
    std::ofstream f(path);
    f << content;
    return path.string();
}

} // anonymous namespace

TEST(TickDataLoader, LoadCSV3Columns) {
    auto path = write_temp_csv(
        "timestamp,price,qty\n"
        "1700000000,100.5,10\n"
        "1700000001,101.0,20\n"
    );
    auto ticks = TickDataLoader::load(path, "TEST");
    ASSERT_EQ(ticks.size(), 2u);
    EXPECT_DOUBLE_EQ(ticks[0].price,    100.5);
    EXPECT_DOUBLE_EQ(ticks[0].quantity, 10.0);
    EXPECT_DOUBLE_EQ(ticks[1].price,    101.0);
    EXPECT_EQ(ticks[0].timestamp_ns, 1700000000LL * 1'000'000'000LL);
}

TEST(TickDataLoader, LoadCSV4ColumnsSide) {
    auto path = write_temp_csv(
        "ts,price,qty,side\n"
        "1700000000,100.0,5,buy\n"
        "1700000001,99.0,3,sell\n"
    );
    auto ticks = TickDataLoader::load(path, "TEST");
    ASSERT_EQ(ticks.size(), 2u);
    EXPECT_EQ(ticks[0].aggressor_side, Side::Buy);
    EXPECT_EQ(ticks[1].aggressor_side, Side::Sell);
}

TEST(TickDataLoader, LoadCSVSortsAscending) {
    auto path = write_temp_csv(
        "ts,price,qty\n"
        "1700000002,102.0,1\n"
        "1700000001,101.0,1\n"
        "1700000000,100.0,1\n"
    );
    auto ticks = TickDataLoader::load(path, "TEST");
    ASSERT_EQ(ticks.size(), 3u);
    EXPECT_LT(ticks[0].timestamp_ns, ticks[1].timestamp_ns);
    EXPECT_LT(ticks[1].timestamp_ns, ticks[2].timestamp_ns);
}

TEST(TickDataLoader, LoadCSVThrowsOnEmpty) {
    auto path = write_temp_csv("ts,price,qty\n");
    EXPECT_THROW(TickDataLoader::load(path, "TEST"), std::runtime_error);
}

namespace {

class PriceRecorder : public Strategy {
public:
    PriceRecorder() : Strategy("PriceRecorder") {}

    void on_data(const MarketDataEvent& event) override {
        prices.push_back(event.get_close());
    }

    std::vector<double> prices;
};

} // anonymous namespace

TEST(BacktestEngine, AddTickDataBasicRun) {
    auto strat = std::make_shared<PriceRecorder>();
    BacktestEngine engine(10000.0);

    TickSeries ticks;
    ticks.emplace_back("AAPL", 1'000'000'000LL, 100.0, 10.0);
    ticks.emplace_back("AAPL", 2'000'000'000LL, 101.0, 10.0);
    ticks.emplace_back("AAPL", 3'000'000'000LL, 102.0, 10.0);

    engine.add_tick_data("AAPL", ticks);
    engine.set_strategy(strat);

    double final_val = engine.run();

    EXPECT_GT(final_val, 0.0);
    ASSERT_EQ(strat->prices.size(), 3u);
    EXPECT_DOUBLE_EQ(strat->prices[0], 100.0);
    EXPECT_DOUBLE_EQ(strat->prices[1], 101.0);
    EXPECT_DOUBLE_EQ(strat->prices[2], 102.0);
}

TEST(BacktestEngine, AddTickDataClearsBarData) {
    BacktestEngine engine(10000.0);

    BarSeries bars;
    bars.emplace_back("AAPL", 1'000'000'000LL, 99.0, 101.0, 98.0, 100.0, 1000.0);
    engine.add_data("AAPL", bars);

    TickSeries ticks;
    ticks.emplace_back("AAPL", 2'000'000'000LL, 105.0, 10.0);
    engine.add_tick_data("AAPL", ticks);

    EXPECT_TRUE(engine.has_tick_data("AAPL"));
}

TEST(BacktestEngine, AddBarDataClearsTickData) {
    BacktestEngine engine(10000.0);

    TickSeries ticks;
    ticks.emplace_back("AAPL", 1'000'000'000LL, 100.0, 10.0);
    engine.add_tick_data("AAPL", ticks);

    BarSeries bars;
    bars.emplace_back("AAPL", 2'000'000'000LL, 99.0, 101.0, 98.0, 100.0, 1000.0);
    engine.add_data("AAPL", bars);

    EXPECT_FALSE(engine.has_tick_data("AAPL"));
}

TEST(BacktestEngine, MmRefreshIntervalAccepted) {
    BacktestEngine engine(10000.0);
    engine.set_mm_refresh_interval(1'000'000'000LL);
    EXPECT_EQ(engine.get_mm_refresh_interval(), 1'000'000'000LL);
}

TEST(BacktestEngine, MmRefreshIntervalNegativeThrows) {
    BacktestEngine engine(10000.0);
    EXPECT_THROW(engine.set_mm_refresh_interval(-1), std::invalid_argument);
}

TEST(BacktestEngine, EquitySnapshotIntervalAccepted) {
    BacktestEngine engine(10000.0);
    engine.set_equity_snapshot_interval(60'000'000'000LL);
    EXPECT_EQ(engine.get_equity_snapshot_interval(), 60'000'000'000LL);
}

TEST(BacktestEngine, EquitySnapshotIntervalNegativeThrows) {
    BacktestEngine engine(10000.0);
    EXPECT_THROW(engine.set_equity_snapshot_interval(-1), std::invalid_argument);
}

TEST(BacktestEngine, EquitySnapshotIntervalReducesSnapshots) {
    auto strat = std::make_shared<PriceRecorder>();
    BacktestEngine engine(10000.0);

    TickSeries ticks;
    for (int i = 0; i < 100; ++i)
        ticks.emplace_back("X", static_cast<int64_t>(i) * 1'000'000'000LL, 100.0, 1.0);

    engine.add_tick_data("X", ticks);
    engine.set_strategy(strat);
    engine.set_equity_snapshot_interval(10'000'000'000LL);
    engine.run();

    EXPECT_LT(engine.get_equity_curve().size(), 20u);
}

TEST(BacktestEngine, EmptyTickSeriesThrows) {
    BacktestEngine engine(10000.0);
    EXPECT_THROW(engine.add_tick_data("AAPL", {}), std::invalid_argument);
}
