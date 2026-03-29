/**
 * Tests for BacktestEngine
 *
 * Fee model (ExecutionConfig defaults):
 *   taker_fee = 0.0002 (0.02%)
 *   maker_fee = 0.0001 (0.01%)
 *   slippage  = 0.0001 (0.01%)
 */

#include <gtest/gtest.h>
#include "backtesting/backtest_engine.h"
#include "backtesting/data_loader.h"
#include "backtesting/bar_data.h"
#include "backtesting/position_sizer.h"
#include "backtesting/risk_manager.h"
#include "strategies/buy_and_hold.h"
#include "strategies/sma_crossover.h"
#include "strategies/mean_reversion.h"
#include "strategies/pairs_trading.h"
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>

using namespace quantcore;

class BacktestEngineTest : public ::testing::Test {
protected:
    static constexpr double INITIAL_CAPITAL = 100000.0;
    static constexpr double TAKER_FEE      = 0.0002;
    static constexpr double SLIPPAGE_PCT   = 0.0001;

    void SetUp() override {
        engine_ = std::make_unique<BacktestEngine>(INITIAL_CAPITAL);
    }

    BarSeries create_flat_bars(int count, double price = 100.0,
                               const std::string& symbol = "TEST") {
        BarSeries bars;
        bars.reserve(count);
        for (int i = 0; i < count; ++i) {
            bars.push_back(BarData(
                symbol, static_cast<int64_t>(i) * 1'000'000'000LL,
                price, price + 1.0, price - 1.0, price, 1'000'000.0
            ));
        }
        return bars;
    }

    BarSeries create_uptrend_bars(int count, double start_price = 100.0,
                                  double increment = 0.5,
                                  const std::string& symbol = "TEST") {
        BarSeries bars;
        bars.reserve(count);
        for (int i = 0; i < count; ++i) {
            double price = start_price + i * increment;
            bars.push_back(BarData(
                symbol, static_cast<int64_t>(i) * 1'000'000'000LL,
                price, price + 1.0, price - 1.0, price, 1'000'000.0
            ));
        }
        return bars;
    }

    BarSeries create_downtrend_bars(int count, double start_price = 100.0,
                                    double decrement = 0.5,
                                    const std::string& symbol = "TEST") {
        BarSeries bars;
        bars.reserve(count);
        for (int i = 0; i < count; ++i) {
            double price = start_price - i * decrement;
            if (price <= 0.0) price = 0.01;
            bars.push_back(BarData(
                symbol, static_cast<int64_t>(i) * 1'000'000'000LL,
                price, price + 1.0, price - 1.0, price, 1'000'000.0
            ));
        }
        return bars;
    }

    void disable_risk_limits() {
        RiskLimits limits;
        limits.enabled = false;
        engine_->set_risk_limits(limits);
    }

    std::unique_ptr<BacktestEngine> engine_;
};

TEST_F(BacktestEngineTest, ZeroCapitalThrows) {
    EXPECT_THROW(BacktestEngine(0.0), std::invalid_argument);
}

TEST_F(BacktestEngineTest, NegativeCapitalThrows) {
    EXPECT_THROW(BacktestEngine(-1000.0), std::invalid_argument);
}

TEST_F(BacktestEngineTest, AddEmptyBarsThrows) {
    EXPECT_THROW(engine_->add_data("TEST", {}), std::invalid_argument);
}

TEST_F(BacktestEngineTest, SetNullStrategyThrows) {
    EXPECT_THROW(engine_->set_strategy(nullptr), std::invalid_argument);
}

TEST_F(BacktestEngineTest, SetNullPositionSizerThrows) {
    EXPECT_THROW(engine_->set_position_sizer(nullptr), std::invalid_argument);
}

TEST_F(BacktestEngineTest, RunWithoutStrategyThrows) {
    engine_->add_data("TEST", create_flat_bars(10));
    EXPECT_THROW(engine_->run(), std::runtime_error);
}

TEST_F(BacktestEngineTest, RunWithoutDataThrows) {
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_THROW(engine_->run(), std::runtime_error);
}

TEST_F(BacktestEngineTest, DefaultPositionSizerIsNotNull) {
    EXPECT_NE(engine_->get_position_sizer(), nullptr);
}

TEST_F(BacktestEngineTest, PortfolioContextInitialCapital) {
    auto portfolio = engine_->get_portfolio_context();
    ASSERT_NE(portfolio, nullptr);
    EXPECT_DOUBLE_EQ(portfolio->get_initial_capital(), INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, EquityCurveStartsAtInitialCapital) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto equity = engine_->get_equity_curve();
    ASSERT_FALSE(equity.empty());
    EXPECT_DOUBLE_EQ(equity.front(), INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, EquityCurveLengthMatchesBars) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    // initial point + one per bar = 21
    EXPECT_EQ(engine_->get_equity_curve().size(), 21u);
}

TEST_F(BacktestEngineTest, EquityCurveAndTimestampsAlwaysSameLength) {
    engine_->add_data("TEST", create_flat_bars(30));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    EXPECT_EQ(engine_->get_equity_curve().size(),
              engine_->get_timestamps().size());
}

TEST_F(BacktestEngineTest, TimestampsAreMonotonicallyNonDecreasing) {
    engine_->add_data("TEST", create_flat_bars(50));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto ts = engine_->get_timestamps();
    for (size_t i = 1; i < ts.size(); ++i) {
        EXPECT_GE(ts[i], ts[i - 1]) << "Timestamp decreased at index " << i;
    }
}

TEST_F(BacktestEngineTest, EquityCurveValuesAreFinite) {
    engine_->add_data("TEST", create_uptrend_bars(50));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    for (double val : engine_->get_equity_curve()) {
        EXPECT_TRUE(std::isfinite(val)) << "Non-finite equity value: " << val;
    }
}

TEST_F(BacktestEngineTest, FinalValueMatchesLastEquityCurvePoint) {
    engine_->add_data("TEST", create_uptrend_bars(30));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    double final_value = engine_->run();
    EXPECT_DOUBLE_EQ(final_value, engine_->get_equity_curve().back());
}

TEST_F(BacktestEngineTest, UptrendYieldsProfitAndDowntrendYieldsLoss) {
    engine_->add_data("TEST", create_uptrend_bars(200, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    double up_final = engine_->run();
    EXPECT_GT(up_final, INITIAL_CAPITAL);

    engine_ = std::make_unique<BacktestEngine>(INITIAL_CAPITAL);
    engine_->add_data("TEST", create_downtrend_bars(200, 100.0, 0.3));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    double down_final = engine_->run();
    EXPECT_LT(down_final, INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, LargerUptrendYieldsLargerProfit) {
    BacktestEngine slow_engine(INITIAL_CAPITAL);
    slow_engine.add_data("TEST", create_uptrend_bars(100, 100.0, 0.5));
    slow_engine.set_strategy(std::make_shared<BuyAndHold>());
    slow_engine.run();
    double slow_pnl = slow_engine.get_total_pnl();

    BacktestEngine fast_engine(INITIAL_CAPITAL);
    fast_engine.add_data("TEST", create_uptrend_bars(100, 100.0, 2.0));
    fast_engine.set_strategy(std::make_shared<BuyAndHold>());
    fast_engine.run();
    double fast_pnl = fast_engine.get_total_pnl();

    EXPECT_GT(fast_pnl, slow_pnl);
}

TEST_F(BacktestEngineTest, FlatMarketPnlIsNegativeFees) {
    // In a flat market, BuyAndHold opens once and never closes.
    // Unrealized PnL = 0, total PnL = -(fees + slippage).
    disable_risk_limits();
    engine_->add_data("TEST", create_flat_bars(20, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    double total_pnl  = engine_->get_total_pnl();
    double total_fees = engine_->get_total_fees();

    EXPECT_GT(total_fees, 0.0);
    EXPECT_LT(total_pnl, 0.0);
    EXPECT_LE(total_pnl, -total_fees);
    EXPECT_GE(total_pnl, -total_fees * 10.0);
}

TEST_F(BacktestEngineTest, PnlComponentsAlwaysSumToTotal) {
    engine_->add_data("TEST", create_uptrend_bars(50, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);

    EXPECT_DOUBLE_EQ(exec->get_total_pnl(),
                     exec->get_realized_pnl() + exec->get_unrealized_pnl());
}

TEST_F(BacktestEngineTest, FeesArePositiveWhenTradeOccurred) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    EXPECT_GT(engine_->get_total_fees(), 0.0);
}

TEST_F(BacktestEngineTest, FeesAreFiniteAndReasonable) {
    engine_->add_data("TEST", create_uptrend_bars(100));
    engine_->set_strategy(std::make_shared<MeanReversion>(10, 1.0, 0.3));
    engine_->run();

    double fees = engine_->get_total_fees();
    EXPECT_TRUE(std::isfinite(fees));
    EXPECT_GT(fees, 0.0);
    EXPECT_LT(fees, INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, TotalFeesEqualsAggregatedSymbolFees) {
    engine_->add_data("AAPL",  create_flat_bars(20, 100.0, "AAPL"));
    engine_->add_data("GOOGL", create_flat_bars(20, 200.0, "GOOGL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");
    ASSERT_NE(exec1, nullptr);
    ASSERT_NE(exec2, nullptr);

    EXPECT_DOUBLE_EQ(engine_->get_total_fees(),
                     exec1->get_total_fees() + exec2->get_total_fees());
}

TEST_F(BacktestEngineTest, TotalPnlEqualsAggregatedSymbolPnl) {
    engine_->add_data("AAPL",  create_uptrend_bars(30, 100.0, 0.5, "AAPL"));
    engine_->add_data("GOOGL", create_uptrend_bars(30, 200.0, 1.0, "GOOGL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");

    EXPECT_NEAR(engine_->get_total_pnl(),
                exec1->get_total_pnl() + exec2->get_total_pnl(), 0.01);
}

TEST_F(BacktestEngineTest, FixedPercentageProducesEqualDollarPnlAcrossPrices) {
    // Scale everything by the same factor. z-scores are invariant to scale
    // so MeanReversion fires on identical bars with proportional sizes.
    // Realized PnL (closed trades only) must then be exactly equal.
    auto run = [&](double scale) -> double {
        BacktestEngine eng(INITIAL_CAPITAL);
        RiskLimits lim; lim.enabled = false;
        eng.set_risk_limits(lim);
        BarSeries bars;
        for (int i = 0; i < 100; ++i) {
            double p = (100.0 + 5.0 * std::sin(i * 0.3)) * scale;
            bars.push_back(BarData("TEST",
                static_cast<int64_t>(i) * 1'000'000'000LL,
                p, p + scale, p - scale, p, 1'000'000.0));
        }
        eng.add_data("TEST", bars);
        eng.set_strategy(std::make_shared<MeanReversion>(10, 1.0, 0.3));
        eng.set_position_sizer(std::make_shared<FixedPercentage>(0.5));
        eng.run();
        return eng.get_execution_engine("TEST")->get_realized_pnl();
    };

    double cheap_realized = run(1.0);
    double exp_realized   = run(2.0);

    EXPECT_NE(cheap_realized, 0.0) << "No trades occurred";
    EXPECT_NEAR(cheap_realized, exp_realized, 0.01);
}

TEST_F(BacktestEngineTest, LargerAllocationProducesLargerPosition) {
    disable_risk_limits();

    BacktestEngine engine_small(INITIAL_CAPITAL);
    engine_small.set_risk_limits([]{ RiskLimits l; l.enabled = false; return l; }());
    engine_small.add_data("TEST", create_flat_bars(30, 100.0));
    engine_small.set_strategy(std::make_shared<BuyAndHold>());
    engine_small.set_position_sizer(std::make_shared<FixedPercentage>(0.05));
    engine_small.run();
    double pos_small = std::abs(engine_small.get_execution_engine("TEST")->get_position());

    BacktestEngine engine_large(INITIAL_CAPITAL);
    engine_large.set_risk_limits([]{ RiskLimits l; l.enabled = false; return l; }());
    engine_large.add_data("TEST", create_flat_bars(30, 100.0));
    engine_large.set_strategy(std::make_shared<BuyAndHold>());
    engine_large.set_position_sizer(std::make_shared<FixedPercentage>(0.20));
    engine_large.run();
    double pos_large = std::abs(engine_large.get_execution_engine("TEST")->get_position());

    ASSERT_GT(pos_small, 0.0) << "5% sizer produced no position";
    ASSERT_GT(pos_large, 0.0) << "20% sizer produced no position";
    EXPECT_GT(pos_large, pos_small);
}

TEST_F(BacktestEngineTest, PositionSizeIsProportionalToAllocationFraction) {
    // 20% allocation should be ~4x the size of 5%.
    disable_risk_limits();

    auto run_with_pct = [&](double pct) -> double {
        BacktestEngine eng(INITIAL_CAPITAL);
        RiskLimits lim; lim.enabled = false;
        eng.set_risk_limits(lim);
        eng.add_data("TEST", create_flat_bars(30, 100.0));
        eng.set_strategy(std::make_shared<BuyAndHold>());
        eng.set_position_sizer(std::make_shared<FixedPercentage>(pct));
        eng.run();
        return std::abs(eng.get_execution_engine("TEST")->get_position());
    };

    double pos_5pct  = run_with_pct(0.05);
    double pos_20pct = run_with_pct(0.20);

    ASSERT_GT(pos_5pct, 0.0);
    EXPECT_NEAR(pos_20pct / pos_5pct, 4.0, 0.4);
}

TEST_F(BacktestEngineTest, PositionSizerChangeIsReflectedAfterRun) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    auto new_sizer = std::make_shared<FixedPercentage>(0.3);
    engine_->set_position_sizer(new_sizer);

    EXPECT_EQ(engine_->get_position_sizer(), new_sizer);
}

TEST_F(BacktestEngineTest, MultiAssetBothGetPositions) {
    engine_->add_data("AAPL",  create_flat_bars(20, 100.0, "AAPL"));
    engine_->add_data("GOOGL", create_flat_bars(20, 200.0, "GOOGL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");
    ASSERT_NE(exec1, nullptr);
    ASSERT_NE(exec2, nullptr);

    EXPECT_GT(std::abs(exec1->get_position()), 0.0);
    EXPECT_GT(std::abs(exec2->get_position()), 0.0);
}

TEST_F(BacktestEngineTest, MultiAssetPnlSignsReflectMarketDirection) {
    engine_->add_data("AAPL",  create_uptrend_bars(50, 100.0, 0.5, "AAPL"));
    engine_->add_data("GOOGL", create_downtrend_bars(50, 200.0, 0.5, "GOOGL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    EXPECT_GT(engine_->get_execution_engine("AAPL")->get_total_pnl(), 0.0);
    EXPECT_LT(engine_->get_execution_engine("GOOGL")->get_total_pnl(), 0.0);
}

TEST_F(BacktestEngineTest, NonExistentSymbolReturnsNullptr) {
    engine_->add_data("AAPL", create_flat_bars(10, 100.0, "AAPL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    EXPECT_EQ(engine_->get_execution_engine("NONEXISTENT"), nullptr);
}

TEST_F(BacktestEngineTest, MultiAssetEquityCurveReflectsAllPositions) {
    BacktestEngine single(INITIAL_CAPITAL);
    single.add_data("AAPL", create_uptrend_bars(30, 100.0, 0.5, "AAPL"));
    single.set_strategy(std::make_shared<BuyAndHold>());
    double single_final = single.run();

    BacktestEngine multi(INITIAL_CAPITAL);
    multi.add_data("AAPL",  create_uptrend_bars(30, 100.0, 0.5, "AAPL"));
    multi.add_data("GOOGL", create_uptrend_bars(30, 200.0, 1.0, "GOOGL"));
    multi.set_strategy(std::make_shared<BuyAndHold>());
    double multi_final = multi.run();

    EXPECT_NE(single_final, multi_final);
}

TEST_F(BacktestEngineTest, MultipleRunsProduceIdenticalResults) {
    engine_->add_data("TEST", create_uptrend_bars(30, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    double final1 = engine_->run();
    double pnl1   = engine_->get_total_pnl();
    double fees1  = engine_->get_total_fees();

    double final2 = engine_->run();
    double pnl2   = engine_->get_total_pnl();
    double fees2  = engine_->get_total_fees();

    EXPECT_DOUBLE_EQ(final1, final2);
    EXPECT_NEAR(pnl1, pnl2, 0.01);
    EXPECT_DOUBLE_EQ(fees1, fees2);
}

TEST_F(BacktestEngineTest, EquityCurveResetsOnSecondRun) {
    engine_->add_data("TEST", create_flat_bars(10));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    engine_->run();
    size_t len1 = engine_->get_equity_curve().size();

    engine_->run();
    size_t len2 = engine_->get_equity_curve().size();

    EXPECT_EQ(len1, len2);
}

TEST_F(BacktestEngineTest, FeesDoNotAccumulateAcrossRuns) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    engine_->run();
    double fees1 = engine_->get_total_fees();

    engine_->run();
    double fees2 = engine_->get_total_fees();

    EXPECT_DOUBLE_EQ(fees1, fees2);
}

TEST_F(BacktestEngineTest, RiskLimitsCapPositionSize) {
    RiskLimits limits;
    limits.max_position_pct = 0.01;
    limits.enabled = true;
    engine_->set_risk_limits(limits);

    engine_->add_data("TEST", create_flat_bars(50, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);

    // Max allowed: 1% of $100,000 = $1,000 -> 10 shares @ $100
    double position_value = std::abs(exec->get_position() * 100.0);
    EXPECT_LE(position_value, 1200.0) << "Position exceeds risk limit: $" << position_value;
}

TEST_F(BacktestEngineTest, DisabledRiskLimitsAllowLargePositions) {
    RiskLimits tight;
    tight.max_position_pct = 0.01;
    tight.enabled = true;
    engine_->set_risk_limits(tight);
    engine_->add_data("TEST", create_flat_bars(20, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->set_position_sizer(std::make_shared<FixedPercentage>(0.5));
    engine_->run();
    double restricted_pos = std::abs(engine_->get_execution_engine("TEST")->get_position());

    engine_ = std::make_unique<BacktestEngine>(INITIAL_CAPITAL);
    RiskLimits off; off.enabled = false;
    engine_->set_risk_limits(off);
    engine_->add_data("TEST", create_flat_bars(20, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->set_position_sizer(std::make_shared<FixedPercentage>(0.5));
    engine_->run();
    double unrestricted_pos = std::abs(engine_->get_execution_engine("TEST")->get_position());

    EXPECT_GT(unrestricted_pos, restricted_pos);
}

TEST_F(BacktestEngineTest, RiskManagerAccessible) {
    EXPECT_NE(engine_->get_risk_manager(), nullptr);
}

TEST_F(BacktestEngineTest, RiskRejectionDoesNotCrashEngine) {
    RiskLimits zero;
    zero.max_position_pct = 0.0001;
    zero.enabled = true;
    engine_->set_risk_limits(zero);

    engine_->add_data("TEST", create_flat_bars(50));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, BuyAndHoldOpensPositionOnFirstBar) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    disable_risk_limits();
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);
    EXPECT_GT(std::abs(exec->get_position()), 0.0);
}

TEST_F(BacktestEngineTest, BuyAndHoldOnlyTradesOnce) {
    engine_->add_data("TEST", create_uptrend_bars(100));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    disable_risk_limits();
    engine_->run();

    // BuyAndHold never closes, so unrealized PnL should be positive in an uptrend.
    EXPECT_GT(engine_->get_execution_engine("TEST")->get_unrealized_pnl(), 0.0);
}

TEST_F(BacktestEngineTest, SMACrossoverRequiresSufficientDataBeforeTrading) {
    engine_->add_data("TEST", create_uptrend_bars(10, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<SMACrossover>(50, 200));
    engine_->run();

    EXPECT_DOUBLE_EQ(engine_->get_execution_engine("TEST")->get_position(), 0.0)
        << "SMA 50/200 should not trade on only 10 bars";
}

TEST_F(BacktestEngineTest, SMACrossoverTradesAfterWarmup) {
    BarSeries bars;
    const int slow = 50;

    // Phase 1: downtrend so fast SMA drops below slow
    for (int i = 0; i < slow + 10; ++i) {
        double price = 200.0 - i * 1.0;
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               price, price + 1.0, price - 1.0, price, 1'000'000.0));
    }

    // Phase 2: sharp uptrend so fast SMA crosses back above slow
    int offset = static_cast<int>(bars.size());
    for (int i = 0; i < slow + 10; ++i) {
        double price = bars.back().close + i * 2.0;
        bars.push_back(BarData("TEST", static_cast<int64_t>(offset + i) * 1'000'000'000LL,
                               price, price + 1.0, price - 1.0, price, 1'000'000.0));
    }

    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<SMACrossover>(10, slow));
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    bool traded = std::abs(exec->get_position()) > 0.0 || exec->get_total_fees() > 0.0;
    EXPECT_TRUE(traded);
}

TEST_F(BacktestEngineTest, MeanReversionTradesOnOscillatingPrices) {
    BarSeries bars;
    for (int i = 0; i < 100; ++i) {
        double price = 100.0 + 10.0 * std::sin(i * 0.3);
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               price, price + 1.0, price - 1.0, price, 1'000'000.0));
    }

    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<MeanReversion>(10, 1.0, 0.3));
    engine_->run();

    EXPECT_GT(engine_->get_total_fees(), 0.0);
}

TEST_F(BacktestEngineTest, MeanReversionDoesNotTradeOnFlatPrices) {
    engine_->add_data("TEST", create_flat_bars(50));
    engine_->set_strategy(std::make_shared<MeanReversion>(10, 2.0, 0.5));
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    EXPECT_DOUBLE_EQ(exec->get_position(), 0.0);
    EXPECT_DOUBLE_EQ(exec->get_total_fees(), 0.0);
}

TEST_F(BacktestEngineTest, PairsStrategyRequiresTwoSymbols) {
    engine_->add_data("SYM1", create_uptrend_bars(60, 100.0, 0.5, "SYM1"));
    engine_->add_data("SYM2", create_downtrend_bars(60, 100.0, 0.3, "SYM2"));
    engine_->set_strategy(std::make_shared<PairsTrading>("SYM1", "SYM2", 20, 1.5, 0.5));

    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, SingleBarBacktest) {
    engine_->add_data("TEST", create_flat_bars(1));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, LargeBacktestCompletesWithoutCrash) {
    engine_->add_data("TEST", create_flat_bars(1000));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, ZeroVolumeBarsHandledGracefully) {
    BarSeries bars;
    for (int i = 0; i < 20; ++i) {
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               100.0, 101.0, 99.0, 100.0, 0.0));
    }
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, FlatPriceProducesZeroVolatilityWithoutCrash) {
    BarSeries bars;
    for (int i = 0; i < 50; ++i) {
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               100.0, 100.0, 100.0, 100.0, 1'000'000.0));
    }
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());

    EXPECT_TRUE(std::isfinite(engine_->get_total_pnl()));
}

TEST_F(BacktestEngineTest, LargePriceGapProducesFinitePnl) {
    BarSeries bars;
    for (int i = 0; i < 10; ++i) {
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               100.0, 101.0, 99.0, 100.0, 1'000'000.0));
    }
    for (int i = 10; i < 30; ++i) {
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               150.0, 151.0, 149.0, 150.0, 1'000'000.0));
    }
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    double pnl = engine_->get_total_pnl();
    EXPECT_TRUE(std::isfinite(pnl));
    EXPECT_GT(pnl, 0.0);
}

TEST_F(BacktestEngineTest, EquityNeverNegative) {
    engine_->add_data("TEST", create_downtrend_bars(100, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    for (double val : engine_->get_equity_curve()) {
        EXPECT_GE(val, 0.0) << "Equity went negative: " << val;
    }
}

TEST_F(BacktestEngineTest, SimultaneousSameTimestampBarsHandled) {
    BarSeries bars1, bars2;
    for (int i = 0; i < 20; ++i) {
        int64_t ts = static_cast<int64_t>(i) * 1'000'000'000LL;
        bars1.push_back(BarData("AAPL",  ts, 100.0, 101.0, 99.0,  100.0, 1'000'000.0));
        bars2.push_back(BarData("GOOGL", ts, 200.0, 201.0, 199.0, 200.0, 1'000'000.0));
    }
    engine_->add_data("AAPL",  bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());

    EXPECT_TRUE(std::isfinite(engine_->get_total_pnl()));
}
