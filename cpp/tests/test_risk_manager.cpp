/** Tests for PositionSizer classes */

#include <gtest/gtest.h>
#include "backtesting/position_sizer.h"
#include <cmath>

using namespace quantcore;

class PositionSizerTest : public ::testing::Test {
protected:
    PositionSizingContext create_context(
        double strength = 1.0,
        double capital = 100000.0,
        double price = 100.0,
        double position = 0.0,
        double volatility = 0.02,
        double stop_distance = 0.05
    ) {
        return PositionSizingContext(strength, capital, price, position, volatility, stop_distance);
    }
};

TEST_F(PositionSizerTest, FixedPercentageBasic) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context();

    // 10% of $100,000 = $10,000 / $100 = 100 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 100.0);
}

TEST_F(PositionSizerTest, FixedPercentageWithStrength) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(0.5);

    // 100 shares * 0.5 = 50 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 50.0);
}

TEST_F(PositionSizerTest, FixedPercentageSmallAllocation) {
    FixedPercentage sizer(0.01);
    auto ctx = create_context();

    // 1% of $100,000 = $1,000 / $100 = 10 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 10.0);
}

TEST_F(PositionSizerTest, FixedPercentageFullAllocation) {
    FixedPercentage sizer(1.0);
    auto ctx = create_context();

    // 100% of $100,000 = $100,000 / $100 = 1000 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 1000.0);
}

TEST_F(PositionSizerTest, FixedPercentageHighPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 500.0);

    // 10% of $100,000 = $10,000 / $500 = 20 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 20.0);
}

TEST_F(PositionSizerTest, FixedPercentageInvalidPctZero) {
    EXPECT_THROW(FixedPercentage(0.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedPercentageInvalidPctNegative) {
    EXPECT_THROW(FixedPercentage(-0.1), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedPercentageGetName) {
    FixedPercentage sizer(0.1);
    std::string name = sizer.get_name();

    EXPECT_TRUE(name.find("FixedPercentage") != std::string::npos);
    EXPECT_TRUE(name.find("10") != std::string::npos);
}

TEST_F(PositionSizerTest, RiskBasedBasic) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.05);

    // risk amount: $100,000 * 0.01 = $1,000
    // risk per share: $100 * 0.05 = $5
    // shares: $1,000 / $5 = 200
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 200.0);
}

TEST_F(PositionSizerTest, RiskBasedTightStop) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.02);

    // risk per share: $100 * 0.02 = $2; shares: $1,000 / $2 = 500
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 500.0);
}

TEST_F(PositionSizerTest, RiskBasedWideStop) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.10);

    // risk per share: $100 * 0.10 = $10; shares: $1,000 / $10 = 100
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 100.0);
}

TEST_F(PositionSizerTest, RiskBasedWithStrength) {
    RiskBased sizer(0.01);
    auto ctx = create_context(0.5, 100000.0, 100.0, 0.0, 0.02, 0.05);

    // 200 shares * 0.5 = 100 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 100.0);
}

TEST_F(PositionSizerTest, RiskBasedZeroStopDistance) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.0);

    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 0.0);
}

TEST_F(PositionSizerTest, RiskBasedInvalidRiskZero) {
    EXPECT_THROW(RiskBased(0.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, RiskBasedInvalidRiskTooHigh) {
    EXPECT_THROW(RiskBased(0.15), std::invalid_argument);
}

TEST_F(PositionSizerTest, RiskBasedGetName) {
    RiskBased sizer(0.02);
    EXPECT_TRUE(sizer.get_name().find("RiskBased") != std::string::npos);
}

TEST_F(PositionSizerTest, KellyCriterionBasic) {
    // win rate 60%, avg win $2, avg loss $1, full Kelly
    // Kelly %: (0.6 * 2 - 0.4) / 2 = 0.4 = 40%
    // allocation: $100,000 * 0.4 = $40,000 / $100 = 400 shares
    KellyCriterion sizer(0.6, 2.0, 1.0, 1.0);
    EXPECT_NEAR(sizer.calculate_size(create_context()), 400.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionHalfKelly) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 0.5);
    EXPECT_NEAR(sizer.calculate_size(create_context()), 200.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionQuarterKelly) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 0.25);
    EXPECT_NEAR(sizer.calculate_size(create_context()), 100.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionLosingStrategy) {
    // win rate 30%, avg win $1, avg loss $1: negative edge -> 0
    KellyCriterion sizer(0.3, 1.0, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 0.0);
}

TEST_F(PositionSizerTest, KellyCriterionBreakeven) {
    // win rate 50%, avg win $1, avg loss $1: zero edge -> 0
    KellyCriterion sizer(0.5, 1.0, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 0.0);
}

TEST_F(PositionSizerTest, KellyCriterionHighWinRate) {
    // win rate 80%, avg win $1.5, avg loss $1
    // Kelly %: (0.8 * 1.5 - 0.2) / 1.5 = 0.667
    KellyCriterion sizer(0.8, 1.5, 1.0, 1.0);
    EXPECT_NEAR(sizer.calculate_size(create_context()), 666.67, 0.1);
}

TEST_F(PositionSizerTest, KellyCriterionWithStrength) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 1.0);
    auto ctx = create_context(0.5);
    EXPECT_NEAR(sizer.calculate_size(ctx), 200.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidWinRate) {
    EXPECT_THROW(KellyCriterion(0.0,  2.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(1.0,  2.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(-0.1, 2.0, 1.0, 1.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidAvgWinLoss) {
    EXPECT_THROW(KellyCriterion(0.6, 0.0,  1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, 2.0,  0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, -1.0, 1.0, 1.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidFraction) {
    EXPECT_THROW(KellyCriterion(0.6, 2.0, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, 2.0, 1.0, 1.5), std::invalid_argument);
}

TEST_F(PositionSizerTest, EqualWeightBasic) {
    EqualWeight sizer(5);
    // $100,000 / 5 = $20,000 / $100 = 200 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 200.0);
}

TEST_F(PositionSizerTest, EqualWeightTenPositions) {
    EqualWeight sizer(10);
    // $100,000 / 10 = $10,000 / $100 = 100 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 100.0);
}

TEST_F(PositionSizerTest, EqualWeightSinglePosition) {
    EqualWeight sizer(1);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 1000.0);
}

TEST_F(PositionSizerTest, EqualWeightWithStrength) {
    EqualWeight sizer(5);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context(0.5)), 100.0);
}

TEST_F(PositionSizerTest, EqualWeightInvalidZero) {
    EXPECT_THROW(EqualWeight(0), std::invalid_argument);
}

TEST_F(PositionSizerTest, EqualWeightInvalidNegative) {
    EXPECT_THROW(EqualWeight(-5), std::invalid_argument);
}

TEST_F(PositionSizerTest, EqualWeightGetName) {
    EqualWeight sizer(10);
    std::string name = sizer.get_name();
    EXPECT_TRUE(name.find("EqualWeight") != std::string::npos);
    EXPECT_TRUE(name.find("10") != std::string::npos);
}

TEST_F(PositionSizerTest, VolatilityTargetingBasic) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.15, 0.05);

    // leverage: 0.15 / 0.15 = 1.0; allocation: $100,000 * 1.0 / $100 = 1000 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 1000.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingLowVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.05, 0.05);

    // leverage: 0.15 / 0.05 = 3.0; 3000 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 3000.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingHighVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.30, 0.05);

    // leverage: 0.15 / 0.30 = 0.5; 500 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 500.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingWithStrength) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(0.5, 100000.0, 100.0, 0.0, 0.15, 0.05);

    // 1000 * 0.5 = 500 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 500.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingZeroVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.0, 0.05);

    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 0.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingInvalidTarget) {
    EXPECT_THROW(VolatilityTargeting(0.0), std::invalid_argument);
    EXPECT_THROW(VolatilityTargeting(-0.1), std::invalid_argument);
    EXPECT_NO_THROW(VolatilityTargeting(1.5));
}

TEST_F(PositionSizerTest, FixedSharesBasic) {
    FixedShares sizer(100);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 100.0);
}

TEST_F(PositionSizerTest, FixedSharesWithStrength) {
    FixedShares sizer(100);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context(0.5)), 50.0);
}

TEST_F(PositionSizerTest, FixedSharesLargePosition) {
    FixedShares sizer(10000);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 10000.0);
}

TEST_F(PositionSizerTest, FixedSharesInvalidZero) {
    EXPECT_THROW(FixedShares(0), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedSharesInvalidNegative) {
    EXPECT_THROW(FixedShares(-100), std::invalid_argument);
}

TEST_F(PositionSizerTest, MaxPositionSizeConstraint) {
    FixedPercentage sizer(0.5);  // 50% = 500 shares
    sizer.set_max_position_size(200.0);

    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 200.0);
}

TEST_F(PositionSizerTest, MinPositionSizeConstraint) {
    FixedPercentage sizer(0.01);  // 1% = 10 shares
    sizer.set_min_position_size(50.0);

    // below minimum -> return 0
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 0.0);
}

TEST_F(PositionSizerTest, MaxLeverageConstraint) {
    FixedPercentage sizer(0.5);
    sizer.set_max_leverage(0.1);  // max notional: $10,000 -> 100 shares

    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 100.0);
}

TEST_F(PositionSizerTest, MultipleConstraints) {
    FixedPercentage sizer(0.5);
    sizer.set_max_position_size(300.0);
    sizer.set_max_leverage(0.2);  // 200 shares

    // leverage constraint (200) is tighter
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context()), 200.0);
}

TEST_F(PositionSizerTest, VerySmallCapital) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100.0, 100.0);

    // 10% of $100 = $10 / $100 = 0.1 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 0.1);
}

TEST_F(PositionSizerTest, VeryHighPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 50000.0);

    // 10% of $100,000 = $10,000 / $50,000 = 0.2 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 0.2);
}

TEST_F(PositionSizerTest, VeryLowPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 0.01);

    // 10% of $100,000 = $10,000 / $0.01 = 1,000,000 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(ctx), 1000000.0);
}

TEST_F(PositionSizerTest, ZeroStrength) {
    FixedPercentage sizer(0.1);
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context(0.0)), 0.0);
}

TEST_F(PositionSizerTest, HighStrength) {
    FixedPercentage sizer(0.1);
    // 100 shares * 2.0 = 200 shares
    EXPECT_DOUBLE_EQ(sizer.calculate_size(create_context(2.0)), 200.0);
}
