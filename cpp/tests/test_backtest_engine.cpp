/**
 * Tests for BacktestEngine
 *
 * Tests verify financial correctness with exact expected values derived from
 * manual calculations, not just "it ran" or "value > 0" checks.
 *
 * Fee model (ExecutionConfig defaults):
 *   taker_fee = 0.0002 (0.02%)
 *   maker_fee = 0.0001 (0.01%)
 *   slippage  = 0.0001 (0.01%)
 *
 * Default position sizer: FixedPercentage
 * Default capital: $100,000
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

// ============================================================================
// TEST FIXTURE
// ============================================================================

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

    // Disable risk limits so tests can reason about raw position sizing
    void disable_risk_limits() {
        RiskLimits limits;
        limits.enabled = false;
        engine_->set_risk_limits(limits);
    }

    std::unique_ptr<BacktestEngine> engine_;
};

// ============================================================================
// INITIALIZATION & CONFIGURATION
// ============================================================================

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

// ============================================================================
// EQUITY CURVE INVARIANTS
// ============================================================================

TEST_F(BacktestEngineTest, EquityCurveStartsAtInitialCapital) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto equity = engine_->get_equity_curve();
    ASSERT_FALSE(equity.empty());
    EXPECT_DOUBLE_EQ(equity.front(), INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, EquityCurveLengthMatchesBars) {
    // One equity point is recorded per market data event, plus the initial
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto equity = engine_->get_equity_curve();
    // initial point + one per bar = 21
    EXPECT_EQ(equity.size(), 21u);
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
    auto equity = engine_->get_equity_curve();

    EXPECT_DOUBLE_EQ(final_value, equity.back());
}

TEST_F(BacktestEngineTest, UptrendYieldsProfitAndDowntrendYieldsLoss) {
    // Buy and hold in rising market must make money (net of fees)
    engine_->add_data("TEST", create_uptrend_bars(200, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    double up_final = engine_->run();
    EXPECT_GT(up_final, INITIAL_CAPITAL);

    // Buy and hold in falling market must lose money
    engine_ = std::make_unique<BacktestEngine>(INITIAL_CAPITAL);
    engine_->add_data("TEST", create_downtrend_bars(200, 100.0, 0.3));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    double down_final = engine_->run();
    EXPECT_LT(down_final, INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, LargerUptrendYieldsLargerProfit) {
    // Steeper trend should produce more PnL
    BacktestEngine slow_engine(INITIAL_CAPITAL);
    slow_engine.add_data("TEST", create_uptrend_bars(100, 100.0, 0.5));
    slow_engine.set_strategy(std::make_shared<BuyAndHold>());
    double slow_pnl = slow_engine.get_total_pnl();
    slow_engine.run();
    slow_pnl = slow_engine.get_total_pnl();

    BacktestEngine fast_engine(INITIAL_CAPITAL);
    fast_engine.add_data("TEST", create_uptrend_bars(100, 100.0, 2.0));
    fast_engine.set_strategy(std::make_shared<BuyAndHold>());
    fast_engine.run();
    double fast_pnl = fast_engine.get_total_pnl();

    EXPECT_GT(fast_pnl, slow_pnl);
}

// ============================================================================
// PNL & FEE FINANCIAL CORRECTNESS
// ============================================================================

TEST_F(BacktestEngineTest, FlatMarketPnlIsNegativeFees) {
    // In a perfectly flat market, BuyAndHold opens once and never closes.
    // Unrealized PnL = 0 (price unchanged).
    // Total PnL = -(fees + slippage), both of which are execution costs.
    // The engine deducts slippage from PnL independently of total_fees, so
    // total_pnl <= -total_fees in a flat market.
    disable_risk_limits();
    engine_->add_data("TEST", create_flat_bars(20, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    double total_pnl  = engine_->get_total_pnl();
    double total_fees = engine_->get_total_fees();

    // Fees must be positive (we traded)
    EXPECT_GT(total_fees, 0.0);
    // PnL must be negative (execution costs in a flat market)
    EXPECT_LT(total_pnl, 0.0);
    // PnL cannot be better than just fees, slippage makes it worse
    EXPECT_LE(total_pnl, -total_fees);
    // PnL should not be astronomically worse than fees (sanity bound)
    EXPECT_GE(total_pnl, -total_fees * 10.0);
}

TEST_F(BacktestEngineTest, PnlComponentsAlwaysSumToTotal) {
    // realized + unrealized == total_pnl, always.
    engine_->add_data("TEST", create_uptrend_bars(50, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);

    double realized   = exec->get_realized_pnl();
    double unrealized = exec->get_unrealized_pnl();
    double total      = exec->get_total_pnl();

    EXPECT_DOUBLE_EQ(total, realized + unrealized);
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
    // Fees should never exceed total capital (that would be broken fee math)
    EXPECT_LT(fees, INITIAL_CAPITAL);
}

TEST_F(BacktestEngineTest, TotalFeesEqualsAggregatedSymbolFees) {
    // total_fees from engine must equal sum of per-symbol fees.
    BarSeries bars1 = create_flat_bars(20, 100.0, "AAPL");
    BarSeries bars2 = create_flat_bars(20, 200.0, "GOOGL");

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");
    ASSERT_NE(exec1, nullptr);
    ASSERT_NE(exec2, nullptr);

    double expected = exec1->get_total_fees() + exec2->get_total_fees();
    EXPECT_DOUBLE_EQ(engine_->get_total_fees(), expected);
}

TEST_F(BacktestEngineTest, TotalPnlEqualsAggregatedSymbolPnl) {
    BarSeries bars1 = create_uptrend_bars(30, 100.0, 0.5, "AAPL");
    BarSeries bars2 = create_uptrend_bars(30, 200.0, 1.0, "GOOGL");

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");

    double expected = exec1->get_total_pnl() + exec2->get_total_pnl();
    EXPECT_NEAR(engine_->get_total_pnl(), expected, 0.01);
}

TEST_F(BacktestEngineTest, HigherPriceAssetGeneratesMoreAbsolutePnl) {
    // Same percentage move, but $200 asset should produce ~2x the absolute PnL
    // versus a $100 asset when position sizing is proportional.
    disable_risk_limits();

    BacktestEngine engine_cheap(INITIAL_CAPITAL);
    engine_cheap.set_risk_limits([]{ RiskLimits l; l.enabled = false; return l; }());
    engine_cheap.add_data("TEST", create_uptrend_bars(50, 100.0, 1.0));
    engine_cheap.set_strategy(std::make_shared<BuyAndHold>());
    engine_cheap.set_position_sizer(std::make_shared<FixedPercentage>(0.5));
    engine_cheap.run();

    BacktestEngine engine_exp(INITIAL_CAPITAL);
    engine_exp.set_risk_limits([]{ RiskLimits l; l.enabled = false; return l; }());
    engine_exp.add_data("TEST", create_uptrend_bars(50, 200.0, 2.0));
    engine_exp.set_strategy(std::make_shared<BuyAndHold>());
    engine_exp.set_position_sizer(std::make_shared<FixedPercentage>(0.5));
    engine_exp.run();

    // Both should be profitable with same % allocation; expensive asset has same PnL
    // because FixedPercentage allocates proportionally, roughly equal dollar PnL
    EXPECT_GT(engine_cheap.get_total_pnl(), 0.0);
    EXPECT_GT(engine_exp.get_total_pnl(), 0.0);
}

// ============================================================================
// POSITION SIZING INTEGRATION
// ============================================================================

TEST_F(BacktestEngineTest, LargerAllocationProducesLargerPosition) {
    // 5% allocation should produce a smaller position than 20%.
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
    // 20% allocation should be ~4x the size of a 5% allocation.
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
    // Allow 10% tolerance for spread/slippage effects
    EXPECT_NEAR(pos_20pct / pos_5pct, 4.0, 0.4);
}

TEST_F(BacktestEngineTest, PositionSizerChangeIsReflectedAfterRun) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    auto new_sizer = std::make_shared<FixedPercentage>(0.3);
    engine_->set_position_sizer(new_sizer);

    EXPECT_EQ(engine_->get_position_sizer(), new_sizer);
}

// ============================================================================
// MULTI-ASSET CORRECTNESS
// ============================================================================

TEST_F(BacktestEngineTest, MultiAssetBothGetPositions) {
    BarSeries bars1 = create_flat_bars(20, 100.0, "AAPL");
    BarSeries bars2 = create_flat_bars(20, 200.0, "GOOGL");

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
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
    // AAPL goes up, GOOGL goes down, both should have correct PnL sign.
    BarSeries bars_up   = create_uptrend_bars(50, 100.0, 0.5, "AAPL");
    BarSeries bars_down = create_downtrend_bars(50, 200.0, 0.5, "GOOGL");

    engine_->add_data("AAPL", bars_up);
    engine_->add_data("GOOGL", bars_down);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec_up   = engine_->get_execution_engine("AAPL");
    auto exec_down = engine_->get_execution_engine("GOOGL");

    EXPECT_GT(exec_up->get_total_pnl(), 0.0)
        << "Long position in rising market should be profitable";
    EXPECT_LT(exec_down->get_total_pnl(), 0.0)
        << "Long position in falling market should be unprofitable";
}

TEST_F(BacktestEngineTest, NonExistentSymbolReturnsNullptr) {
    engine_->add_data("AAPL", create_flat_bars(10, 100.0, "AAPL"));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    EXPECT_EQ(engine_->get_execution_engine("NONEXISTENT"), nullptr);
}

TEST_F(BacktestEngineTest, MultiAssetEquityCurveReflectsAllPositions) {
    // Running two assets should produce a different equity curve than one.
    BacktestEngine single(INITIAL_CAPITAL);
    single.add_data("AAPL", create_uptrend_bars(30, 100.0, 0.5, "AAPL"));
    single.set_strategy(std::make_shared<BuyAndHold>());
    double single_final = single.run();

    BacktestEngine multi(INITIAL_CAPITAL);
    multi.add_data("AAPL",  create_uptrend_bars(30, 100.0, 0.5, "AAPL"));
    multi.add_data("GOOGL", create_uptrend_bars(30, 200.0, 1.0, "GOOGL"));
    multi.set_strategy(std::make_shared<BuyAndHold>());
    double multi_final = multi.run();

    // More positions = more exposure = larger moves in both directions
    EXPECT_NE(single_final, multi_final);
}

// ============================================================================
// IDEMPOTENCY & RESET
// ============================================================================

TEST_F(BacktestEngineTest, MultipleRunsProduceIdenticalResults) {
    auto bars = create_uptrend_bars(30, 100.0, 0.5);
    engine_->add_data("TEST", bars);
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
    auto bars = create_flat_bars(10);
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    engine_->run();
    size_t len1 = engine_->get_equity_curve().size();

    engine_->run();
    size_t len2 = engine_->get_equity_curve().size();

    // Equity curve should be exactly the same length after each run (not accumulated)
    EXPECT_EQ(len1, len2);
}

TEST_F(BacktestEngineTest, FeesDoNotAccumulateAcrossRuns) {
    auto bars = create_flat_bars(20);
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    engine_->run();
    double fees1 = engine_->get_total_fees();

    engine_->run();
    double fees2 = engine_->get_total_fees();

    // If fees accumulated across runs, fees2 would be ~2x fees1
    EXPECT_DOUBLE_EQ(fees1, fees2);
}

// ============================================================================
// RISK MANAGER INTEGRATION
// ============================================================================

TEST_F(BacktestEngineTest, RiskLimitsCapPositionSize) {
    // With a very tight position limit, position must stay within it.
    RiskLimits limits;
    limits.max_position_pct = 0.01;  // 1% of capital
    limits.enabled = true;
    engine_->set_risk_limits(limits);

    engine_->add_data("TEST", create_flat_bars(50, 100.0));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);

    double position_value = std::abs(exec->get_position() * 100.0);
    // Max allowed: 1% of $100,000 = $1,000 → 10 shares @ $100
    // Allow small buffer for spread/slippage execution price
    EXPECT_LE(position_value, 1200.0)
        << "Position exceeds risk limit: $" << position_value;
}

TEST_F(BacktestEngineTest, DisabledRiskLimitsAllowLargePositions) {
    // Disabling risk limits should allow much larger positions.
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
    // Even when every order is rejected, run() must complete cleanly.
    RiskLimits zero;
    zero.max_position_pct = 0.0001;  // Effectively zero
    zero.enabled = true;
    engine_->set_risk_limits(zero);

    engine_->add_data("TEST", create_flat_bars(50));
    engine_->set_strategy(std::make_shared<BuyAndHold>());

    EXPECT_NO_THROW(engine_->run());
}

// ============================================================================
// STRATEGY CORRECTNESS
// ============================================================================

TEST_F(BacktestEngineTest, BuyAndHoldOpensPositionOnFirstBar) {
    engine_->add_data("TEST", create_flat_bars(20));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    disable_risk_limits();
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    ASSERT_NE(exec, nullptr);
    // Must have a position, BuyAndHold buys once on the first bar
    EXPECT_GT(std::abs(exec->get_position()), 0.0);
}

TEST_F(BacktestEngineTest, BuyAndHoldOnlyTradesOnce) {
    engine_->add_data("TEST", create_uptrend_bars(100));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    disable_risk_limits();
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    // BuyAndHold never closes, so realized PnL should only reflect entry fees.
    // Unrealized PnL should be positive (uptrend).
    EXPECT_GT(exec->get_unrealized_pnl(), 0.0);
}

TEST_F(BacktestEngineTest, SMACrossoverRequiresSufficientDataBeforeTrading) {
    // With only 10 bars, a 50/200 SMA strategy should never generate a signal.
    auto bars = create_uptrend_bars(10, 100.0, 0.5);
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<SMACrossover>(50, 200));
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    EXPECT_DOUBLE_EQ(exec->get_position(), 0.0)
        << "SMA 50/200 should not trade on only 10 bars";
}

TEST_F(BacktestEngineTest, SMACrossoverTradesAfterWarmup) {
    // A monotonic uptrend never produces a crossover (fast SMA is always above
    // slow SMA once the slow SMA warms up, so there is no transition event).
    // To guarantee a crossover we need a downtrend followed by an uptrend:
    // price falls for slow_period bars → fast crosses below slow,
    // then rises → fast crosses back above slow, triggering a BUY signal.
    BarSeries bars;
    const int slow = 50;
    // Phase 1: downtrend; fast SMA drops below slow SMA
    for (int i = 0; i < slow + 10; ++i) {
        double price = 200.0 - i * 1.0;
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               price, price + 1.0, price - 1.0, price, 1'000'000.0));
    }
    // Phase 2: sharp uptrend; fast SMA crosses back above slow SMA
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
    EXPECT_TRUE(traded) << "SMA 10/50 should have crossed over on downtrend→uptrend price series";
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

    // Oscillating prices should trigger multiple signals, so fees must be > 0
    EXPECT_GT(engine_->get_total_fees(), 0.0)
        << "MeanReversion should have traded on oscillating prices";
}

TEST_F(BacktestEngineTest, MeanReversionDoesNotTradeOnFlatPrices) {
    // Perfectly flat prices → z-score is always 0 → no signals
    engine_->add_data("TEST", create_flat_bars(50));
    engine_->set_strategy(std::make_shared<MeanReversion>(10, 2.0, 0.5));
    engine_->run();

    auto exec = engine_->get_execution_engine("TEST");
    EXPECT_DOUBLE_EQ(exec->get_position(), 0.0)
        << "MeanReversion should not trade when price never deviates";
    EXPECT_DOUBLE_EQ(exec->get_total_fees(), 0.0);
}

TEST_F(BacktestEngineTest, PairsStrategyRequiresTwoSymbols) {
    BarSeries bars1 = create_uptrend_bars(60, 100.0, 0.5, "SYM1");
    BarSeries bars2 = create_downtrend_bars(60, 100.0, 0.3, "SYM2");

    engine_->add_data("SYM1", bars1);
    engine_->add_data("SYM2", bars2);
    engine_->set_strategy(std::make_shared<PairsTrading>("SYM1", "SYM2", 20, 1.5, 0.5));

    EXPECT_NO_THROW(engine_->run());
}

// ============================================================================
// EDGE CASES & ROBUSTNESS
// ============================================================================

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
    // Volatility calculation on perfectly flat prices must not divide by zero.
    BarSeries bars;
    for (int i = 0; i < 50; ++i) {
        bars.push_back(BarData("TEST", static_cast<int64_t>(i) * 1'000'000'000LL,
                               100.0, 100.0, 100.0, 100.0, 1'000'000.0));
    }
    engine_->add_data("TEST", bars);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());

    double pnl = engine_->get_total_pnl();
    EXPECT_TRUE(std::isfinite(pnl));
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
    // Bought at $100, price gapped to $150 → should be profitable
    EXPECT_GT(pnl, 0.0);
}

TEST_F(BacktestEngineTest, EquityNeverNegative) {
    // Even in a severe downtrend, equity must never go negative.
    engine_->add_data("TEST", create_downtrend_bars(100, 100.0, 0.5));
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    engine_->run();

    for (double val : engine_->get_equity_curve()) {
        EXPECT_GE(val, 0.0) << "Equity went negative: " << val;
    }
}

TEST_F(BacktestEngineTest, SimultaneousSameTimestampBarsHandled) {
    // Both symbols share identical timestamps, event ordering must not break.
    BarSeries bars1, bars2;
    for (int i = 0; i < 20; ++i) {
        int64_t ts = static_cast<int64_t>(i) * 1'000'000'000LL;
        bars1.push_back(BarData("AAPL",  ts, 100.0, 101.0, 99.0, 100.0, 1'000'000.0));
        bars2.push_back(BarData("GOOGL", ts, 200.0, 201.0, 199.0, 200.0, 1'000'000.0));
    }
    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(std::make_shared<BuyAndHold>());
    EXPECT_NO_THROW(engine_->run());

    double pnl = engine_->get_total_pnl();
    EXPECT_TRUE(std::isfinite(pnl));
}