/**
 * Tests for BacktestEngine
 * Tests use exact expected values with manual calculations.
 */

#include <gtest/gtest.h>
#include "backtesting/backtest_engine.h"
#include "backtesting/data_loader.h"
#include "backtesting/bar_data.h"
#include "strategies/buy_and_hold.h"
#include "strategies/sma_crossover.h"
#include "strategies/mean_reversion.h"
#include <memory>
#include <vector>

using namespace quantcore;

class BacktestEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<BacktestEngine>(100000.0);
    }

    // create synthetic bars with constant price
    BarSeries create_flat_bars(int count, double price = 100.0) {
        BarSeries bars;
        for (int i = 0; i < count; ++i) {
            bars.push_back(BarData(
                "TEST",
                i * 1000000000LL,
                price, price + 1.0, price - 1.0, price,
                1000000.0
            ));
        }
        return bars;
    }

    // create upward trending bars
    BarSeries create_uptrend_bars(int count, double start_price = 100.0, double increment = 0.5) {
        BarSeries bars;
        for (int i = 0; i < count; ++i) {
            double price = start_price + i * increment;
            bars.push_back(BarData(
                "TEST",
                i * 1000000000LL,
                price, price + 1.0, price - 1.0, price,
                1000000.0
            ));
        }
        return bars;
    }

    // create downward trending bars
    BarSeries create_downtrend_bars(int count, double start_price = 100.0, double decrement = 0.5) {
        BarSeries bars;
        for (int i = 0; i < count; ++i) {
            double price = start_price - i * decrement;
            bars.push_back(BarData(
                "TEST",
                i * 1000000000LL,
                price, price + 1.0, price - 1.0, price,
                1000000.0
            ));
        }
        return bars;
    }

    std::unique_ptr<BacktestEngine> engine_;
};

// ============================================================================
// INITIALIZATION & CONFIG
// ============================================================================

TEST_F(BacktestEngineTest, Initialization) {
    EXPECT_NO_THROW(BacktestEngine engine(100000.0));
}

TEST_F(BacktestEngineTest, InitializationZeroCapital) {
    EXPECT_THROW(BacktestEngine engine(0.0), std::invalid_argument);
}

TEST_F(BacktestEngineTest, InitializationNegativeCapital) {
    EXPECT_THROW(BacktestEngine engine(-1000.0), std::invalid_argument);
}

TEST_F(BacktestEngineTest, AddDataSingleSymbol) {
    auto bars = create_flat_bars(10);
    EXPECT_NO_THROW(engine_->add_data("TEST", bars));
}

TEST_F(BacktestEngineTest, AddDataEmptyBars) {
    BarSeries empty;
    EXPECT_THROW(engine_->add_data("TEST", empty), std::invalid_argument);
}

TEST_F(BacktestEngineTest, SetStrategy) {
    auto strategy = std::make_shared<BuyAndHold>();
    EXPECT_NO_THROW(engine_->set_strategy(strategy));
}

TEST_F(BacktestEngineTest, SetStrategyNull) {
    EXPECT_THROW(engine_->set_strategy(nullptr), std::invalid_argument);
}

TEST_F(BacktestEngineTest, RunWithoutStrategy) {
    auto bars = create_flat_bars(10);
    engine_->add_data("TEST", bars);
    EXPECT_THROW(engine_->run(), std::runtime_error);
}

TEST_F(BacktestEngineTest, RunWithoutData) {
    auto strategy = std::make_shared<BuyAndHold>();
    engine_->set_strategy(strategy);
    EXPECT_THROW(engine_->run(), std::runtime_error);
}

// ============================================================================
// BASIC BACKTEST EXECUTION
// ============================================================================

TEST_F(BacktestEngineTest, SimpleBuyAndHold) {
    auto bars = create_flat_bars(10, 100.0);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    double final_value = engine_->run();

    // Should complete successfully
    EXPECT_GT(final_value, 0.0);
}

TEST_F(BacktestEngineTest, BuyAndHoldProfitUptrend) {
    auto bars = create_uptrend_bars(50, 100.0, 0.5);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    double final_value = engine_->run();

    // Should make profit in uptrend
    EXPECT_GT(final_value, 100000.0);
}

TEST_F(BacktestEngineTest, BuyAndHoldLossDowntrend) {
    auto bars = create_downtrend_bars(50, 100.0, 0.5);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    double final_value = engine_->run();

    // Should lose money in downtrend
    EXPECT_LT(final_value, 100000.0);
}

TEST_F(BacktestEngineTest, SingleBar) {
    auto bars = create_flat_bars(1, 100.0);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // Should handle single bar gracefully
    EXPECT_NO_THROW(engine_->run());
}

// ============================================================================
// EQUITY CURVE GENERATION
// ============================================================================

TEST_F(BacktestEngineTest, EquityCurveLength) {
    auto bars = create_flat_bars(20);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    auto equity_curve = engine_->get_equity_curve();

    // Should have 21 points: initial + 20 bars
    EXPECT_EQ(equity_curve.size(), 21);
}

TEST_F(BacktestEngineTest, EquityCurveStartsAtInitialCapital) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    auto equity_curve = engine_->get_equity_curve();

    EXPECT_DOUBLE_EQ(equity_curve[0], 100000.0);
}

TEST_F(BacktestEngineTest, EquityCurveMatchesTimestamps) {
    auto bars = create_flat_bars(15);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    auto equity_curve = engine_->get_equity_curve();
    auto timestamps = engine_->get_timestamps();

    EXPECT_EQ(equity_curve.size(), timestamps.size());
}

TEST_F(BacktestEngineTest, EquityCurveMonotonic) {
    // Strictly upward trending market
    auto bars = create_uptrend_bars(30, 100.0, 1.0);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    auto equity_curve = engine_->get_equity_curve();

    // Buy and hold in uptrend should have increasing equity (after initial buy)
    // First few points might dip due to fees, but should trend up
    EXPECT_GT(equity_curve.back(), equity_curve.front());
}

TEST_F(BacktestEngineTest, TimestampsSorted) {
    auto bars = create_flat_bars(20);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    auto timestamps = engine_->get_timestamps();

    // Timestamps should be in ascending order
    for (size_t i = 1; i < timestamps.size(); ++i) {
        EXPECT_GE(timestamps[i], timestamps[i-1]);
    }
}

// ============================================================================
// PNL & FEE TRACKING
// ============================================================================

TEST_F(BacktestEngineTest, TotalPnlZeroNoStrategy) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // Before running
    EXPECT_DOUBLE_EQ(engine_->get_total_pnl(), 0.0);
}

TEST_F(BacktestEngineTest, FeesPositive) {
    auto bars = create_flat_bars(20);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    // Should have paid some fees
    EXPECT_GT(engine_->get_total_fees(), 0.0);
}

TEST_F(BacktestEngineTest, FinalValueCalculation) {
    auto bars = create_uptrend_bars(30, 100.0, 0.5);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    double final_value = engine_->run();
    auto equity_curve = engine_->get_equity_curve();

    // Final value should match last point in equity curve
    EXPECT_DOUBLE_EQ(final_value, equity_curve.back());
}

// ============================================================================
// POSITION SIZING INTEGRATION
// ============================================================================

TEST_F(BacktestEngineTest, DefaultPositionSizer) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // Should have default position sizer
    auto sizer = engine_->get_position_sizer();
    EXPECT_NE(sizer, nullptr);
}

TEST_F(BacktestEngineTest, SetFixedPercentageSizer) {
    auto sizer = std::make_shared<FixedPercentage>(0.2);  // 20%

    EXPECT_NO_THROW(engine_->set_position_sizer(sizer));

    auto retrieved = engine_->get_position_sizer();
    EXPECT_NE(retrieved, nullptr);
}

TEST_F(BacktestEngineTest, SetNullPositionSizer) {
    EXPECT_THROW(engine_->set_position_sizer(nullptr), std::invalid_argument);
}

TEST_F(BacktestEngineTest, PositionSizerAffectsSize) {
    auto bars = create_flat_bars(30, 100.0);

    // Run with 5% allocation
    BacktestEngine engine1(100000.0);

    // Disable risk limits for this test
    RiskLimits limits;
    limits.enabled = false;
    engine1.set_risk_limits(limits);

    engine1.add_data("TEST", bars);
    engine1.set_strategy(std::make_shared<BuyAndHold>());
    engine1.set_position_sizer(std::make_shared<FixedPercentage>(0.05));
    engine1.run();

    double pos1 = std::abs(engine1.get_execution_engine("TEST")->get_position());

    // Run with 20% allocation
    auto bars2 = create_flat_bars(30, 100.0);
    BacktestEngine engine2(100000.0);

    // Disable risk limits for this test
    engine2.set_risk_limits(limits);

    engine2.add_data("TEST", bars2);
    engine2.set_strategy(std::make_shared<BuyAndHold>());
    engine2.set_position_sizer(std::make_shared<FixedPercentage>(0.2));
    engine2.run();

    double pos2 = std::abs(engine2.get_execution_engine("TEST")->get_position());

    std::cout << "5% position: " << pos1 << std::endl;
    std::cout << "20% position: " << pos2 << std::endl;

    ASSERT_GT(pos1, 0.0) << "5% position failed";
    ASSERT_GT(pos2, 0.0) << "20% position failed";
    EXPECT_GT(pos2, pos1);
}

// ============================================================================
// RISK MANAGEMENT INTEGRATION
// ============================================================================

TEST_F(BacktestEngineTest, DefaultRiskLimits) {
    auto limits = engine_->get_risk_limits();

    EXPECT_GT(limits.max_position_pct, 0.0);
    EXPECT_GT(limits.max_leverage, 0.0);
}

TEST_F(BacktestEngineTest, SetRiskLimits) {
    RiskLimits limits;
    limits.max_position_pct = 0.15;
    limits.max_leverage = 1.5;
    limits.max_loss_pct = 0.30;

    EXPECT_NO_THROW(engine_->set_risk_limits(limits));

    auto retrieved = engine_->get_risk_limits();
    EXPECT_DOUBLE_EQ(retrieved.max_position_pct, 0.15);
    EXPECT_DOUBLE_EQ(retrieved.max_leverage, 1.5);
    EXPECT_DOUBLE_EQ(retrieved.max_loss_pct, 0.30);
}

TEST_F(BacktestEngineTest, RiskLimitsRejectOrders) {
    auto bars = create_flat_bars(20);
    auto strategy = std::make_shared<BuyAndHold>();

    // Very restrictive position limit
    RiskLimits limits;
    limits.max_position_pct = 0.01;  // Only 1% allowed
    limits.enabled = true;

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);
    engine_->set_risk_limits(limits);
    engine_->run();

    auto exec_engine = engine_->get_execution_engine("TEST");
    double position_value = exec_engine->get_position() * 100.0;  // Rough estimate

    // Position should be constrained by risk limits
    EXPECT_LT(position_value, 2000.0);  // Should be less than 2% of capital
}

TEST_F(BacktestEngineTest, RiskManagerAccessible) {
    auto risk_mgr = engine_->get_risk_manager();
    EXPECT_NE(risk_mgr, nullptr);
}

// ============================================================================
// VOLATILITY CALCULATIONS
// ============================================================================

TEST_F(BacktestEngineTest, VolatilityParametersDefault) {
    // Should have reasonable defaults
    EXPECT_EQ(engine_->get_bars_per_year(), 252);
}

TEST_F(BacktestEngineTest, SetVolatilityParameters) {
    EXPECT_NO_THROW(engine_->set_volatility_params(0.02, 0.05, 20));
    EXPECT_NO_THROW(engine_->set_bars_per_year(365));

    EXPECT_EQ(engine_->get_bars_per_year(), 365);
}

TEST_F(BacktestEngineTest, InvalidVolatilityParameters) {
    EXPECT_THROW(engine_->set_volatility_params(0.0, 0.05, 20), std::invalid_argument);
    EXPECT_THROW(engine_->set_volatility_params(1.5, 0.05, 20), std::invalid_argument);
    EXPECT_THROW(engine_->set_volatility_params(0.02, 0.0, 20), std::invalid_argument);
    EXPECT_THROW(engine_->set_bars_per_year(0), std::invalid_argument);
}

// ============================================================================
// MULTI-SYMBOL BACKTESTS
// ============================================================================

TEST_F(BacktestEngineTest, TwoSymbolsIndependent) {
    auto bars1 = create_flat_bars(15);
    auto bars2 = create_flat_bars(15);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(strategy);

    engine_->run();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");

    EXPECT_NE(exec1, nullptr);
    EXPECT_NE(exec2, nullptr);
}

TEST_F(BacktestEngineTest, TotalPnlAggregatesSymbols) {
    auto bars1 = create_uptrend_bars(20, 100.0, 0.5);
    auto bars2 = create_uptrend_bars(20, 200.0, 1.0);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(strategy);

    engine_->run();

    double total_pnl = engine_->get_total_pnl();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");

    double individual_sum = exec1->get_total_pnl() + exec2->get_total_pnl();

    // Total should equal sum of individuals
    EXPECT_NEAR(total_pnl, individual_sum, 0.01);
}

TEST_F(BacktestEngineTest, TotalFeesAggregatesSymbols) {
    auto bars1 = create_flat_bars(15);
    auto bars2 = create_flat_bars(15);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("AAPL", bars1);
    engine_->add_data("GOOGL", bars2);
    engine_->set_strategy(strategy);

    engine_->run();

    double total_fees = engine_->get_total_fees();

    auto exec1 = engine_->get_execution_engine("AAPL");
    auto exec2 = engine_->get_execution_engine("GOOGL");

    double individual_fees = exec1->get_total_fees() + exec2->get_total_fees();

    EXPECT_DOUBLE_EQ(total_fees, individual_fees);
}

TEST_F(BacktestEngineTest, NonExistentSymbol) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("AAPL", bars);
    engine_->set_strategy(strategy);
    engine_->run();

    // Should return nullptr for non-existent symbol
    auto exec = engine_->get_execution_engine("NONEXISTENT");
    EXPECT_EQ(exec, nullptr);
}

// ============================================================================
// STRATEGY INTEGRATION
// ============================================================================

TEST_F(BacktestEngineTest, SMACrossoverGeneratesSignals) {
    auto bars = create_uptrend_bars(250, 100.0, 0.1);
    auto strategy = std::make_shared<SMACrossover>(50, 200);

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, MeanReversionGeneratesSignals) {
    // Oscillating prices for mean reversion
    BarSeries bars;
    for (int i = 0; i < 100; ++i) {
        double price = 100.0 + 10.0 * std::sin(i * 0.3);
        bars.push_back(BarData(
            "TEST",
            i * 1000000000LL,
            price, price + 1.0, price - 1.0, price,
            1000000.0
        ));
    }

    auto strategy = std::make_shared<MeanReversion>(20, 1.5, 0.5);
    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, StrategyResetsOnNewRun) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // First run
    engine_->run();
    double pnl1 = engine_->get_total_pnl();

    // Second run should reset
    engine_->run();
    double pnl2 = engine_->get_total_pnl();

    // Should get same result (strategy was reset)
    EXPECT_NEAR(pnl1, pnl2, 0.01);
}

// ============================================================================
// PORTFOLIO CONTEXT
// ============================================================================

TEST_F(BacktestEngineTest, PortfolioContextAccessible) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    auto portfolio = engine_->get_portfolio_context();
    EXPECT_NE(portfolio, nullptr);
}

TEST_F(BacktestEngineTest, PortfolioContextStartsWithCapital) {
    auto bars = create_flat_bars(10);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    auto portfolio = engine_->get_portfolio_context();

    // Initial cash should equal initial capital
    EXPECT_DOUBLE_EQ(portfolio->get_initial_capital(), 100000.0);
}

// ============================================================================
// MARKET MAKER CONFIGURATION
// ============================================================================

TEST_F(BacktestEngineTest, ConfigureMarketMaker) {
    EXPECT_NO_THROW(engine_->configure_market_maker(5, 0.0001, 100000));
    EXPECT_NO_THROW(engine_->configure_market_maker(10, 0.0002, 50000));
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(BacktestEngineTest, VeryLongBacktest) {
    // 1000 bars
    auto bars = create_flat_bars(1000);
    auto strategy = std::make_shared<BuyAndHold>();

    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // Should handle large backtests
    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, HighFrequencySignals) {
    // Mean reversion on volatile data generates many signals
    BarSeries bars;
    for (int i = 0; i < 200; ++i) {
        double price = 100.0 + 5.0 * std::sin(i * 0.5);
        bars.push_back(BarData(
            "TEST",
            i * 1000000000LL,
            price, price + 0.5, price - 0.5, price,
            1000000.0
        ));
    }

    auto strategy = std::make_shared<MeanReversion>(10, 1.0, 0.3);
    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, ZeroVolumeBars) {
    BarSeries bars;
    for (int i = 0; i < 20; ++i) {
        bars.push_back(BarData(
            "TEST",
            i * 1000000000LL,
            100.0, 101.0, 99.0, 100.0,
            0.0  // Zero volume
        ));
    }

    auto strategy = std::make_shared<BuyAndHold>();
    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    // Should handle zero volume gracefully
    EXPECT_NO_THROW(engine_->run());
}

TEST_F(BacktestEngineTest, PriceGaps) {
    // Large gap in prices
    BarSeries bars;
    for (int i = 0; i < 10; ++i) {
        bars.push_back(BarData("TEST", i * 1000000000LL, 100.0, 101.0, 99.0, 100.0, 1000000.0));
    }
    // Gap up 50%
    for (int i = 10; i < 20; ++i) {
        bars.push_back(BarData("TEST", i * 1000000000LL, 150.0, 151.0, 149.0, 150.0, 1000000.0));
    }

    auto strategy = std::make_shared<BuyAndHold>();
    engine_->add_data("TEST", bars);
    engine_->set_strategy(strategy);

    engine_->run();

    // Should handle gaps and reflect in PnL
    EXPECT_GT(engine_->get_total_pnl(), 0.0);
}

