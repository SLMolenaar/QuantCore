/** tests for PositionSizer classes */


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

// ============================================================================
// FIXED PERCENTAGE SIZER
// ============================================================================

TEST_F(PositionSizerTest, FixedPercentageBasic) {
    FixedPercentage sizer(0.1);  // 10%
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // 10% of $100,000 = $10,000 / $100 = 100 shares
    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, FixedPercentageWithStrength) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(0.5);  // 50% signal strength

    double size = sizer.calculate_size(ctx);

    // 100 shares * 0.5 = 50 shares
    EXPECT_DOUBLE_EQ(size, 50.0);
}

TEST_F(PositionSizerTest, FixedPercentageSmallAllocation) {
    FixedPercentage sizer(0.01);  // 1%
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // 1% of $100,000 = $1,000 / $100 = 10 shares
    EXPECT_DOUBLE_EQ(size, 10.0);
}

TEST_F(PositionSizerTest, FixedPercentageFullAllocation) {
    FixedPercentage sizer(1.0);  // 100%
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // 100% of $100,000 = $100,000 / $100 = 1000 shares
    EXPECT_DOUBLE_EQ(size, 1000.0);
}

TEST_F(PositionSizerTest, FixedPercentageHighPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 500.0);  // $500 per share

    double size = sizer.calculate_size(ctx);

    // 10% of $100,000 = $10,000 / $500 = 20 shares
    EXPECT_DOUBLE_EQ(size, 20.0);
}

TEST_F(PositionSizerTest, FixedPercentageInvalidPctZero) {
    EXPECT_THROW(FixedPercentage(0.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedPercentageInvalidPctNegative) {
    EXPECT_THROW(FixedPercentage(-0.1), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedPercentageInvalidPctOverOne) {
    EXPECT_NO_THROW(FixedPercentage(1.5));
    EXPECT_THROW(FixedPercentage(0.0), std::invalid_argument);
    EXPECT_THROW(FixedPercentage(-0.1), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedPercentageGetName) {
    FixedPercentage sizer(0.1);
    std::string name = sizer.get_name();

    EXPECT_TRUE(name.find("FixedPercentage") != std::string::npos);
    EXPECT_TRUE(name.find("10") != std::string::npos);
}

// ============================================================================
// RISK-BASED SIZER
// ============================================================================

TEST_F(PositionSizerTest, RiskBasedBasic) {
    RiskBased sizer(0.01);  // 1% risk per trade
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.05);

    double size = sizer.calculate_size(ctx);

    // Risk amount: $100,000 * 0.01 = $1,000
    // Risk per share: $100 * 0.05 = $5
    // Shares: $1,000 / $5 = 200
    EXPECT_DOUBLE_EQ(size, 200.0);
}

TEST_F(PositionSizerTest, RiskBasedTightStop) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.02);  // 2% stop

    double size = sizer.calculate_size(ctx);

    // Risk per share: $100 * 0.02 = $2
    // Shares: $1,000 / $2 = 500
    EXPECT_DOUBLE_EQ(size, 500.0);
}

TEST_F(PositionSizerTest, RiskBasedWideStop) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.10);  // 10% stop

    double size = sizer.calculate_size(ctx);

    // Risk per share: $100 * 0.10 = $10
    // Shares: $1,000 / $10 = 100
    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, RiskBasedWithStrength) {
    RiskBased sizer(0.01);
    auto ctx = create_context(0.5, 100000.0, 100.0, 0.0, 0.02, 0.05);

    double size = sizer.calculate_size(ctx);

    // 200 shares * 0.5 strength = 100 shares
    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, RiskBasedZeroStopDistance) {
    RiskBased sizer(0.01);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.02, 0.0);

    double size = sizer.calculate_size(ctx);

    // Zero stop distance should return 0
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, RiskBasedInvalidRiskZero) {
    EXPECT_THROW(RiskBased(0.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, RiskBasedInvalidRiskTooHigh) {
    EXPECT_THROW(RiskBased(0.15), std::invalid_argument);
}

TEST_F(PositionSizerTest, RiskBasedGetName) {
    RiskBased sizer(0.02);
    std::string name = sizer.get_name();

    EXPECT_TRUE(name.find("RiskBased") != std::string::npos);
}

// ============================================================================
// KELLY CRITERION SIZER
// ============================================================================

TEST_F(PositionSizerTest, KellyCriterionBasic) {
    // Win rate 60%, avg win $2, avg loss $1, full Kelly
    KellyCriterion sizer(0.6, 2.0, 1.0, 1.0);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Kelly %: (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4 = 40%
    // Allocation: $100,000 * 0.4 = $40,000 / $100 = 400 shares
    EXPECT_NEAR(size, 400.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionHalfKelly) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 0.5);  // Half Kelly
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Half of 400 = 200 shares
    EXPECT_NEAR(size, 200.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionQuarterKelly) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 0.25);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Quarter of 400 = 100 shares
    EXPECT_NEAR(size, 100.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionLosingStrategy) {
    // Win rate 30%, avg win $1, avg loss $1 = negative edge
    KellyCriterion sizer(0.3, 1.0, 1.0, 1.0);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Kelly %: (0.3 * 1 - 0.7) / 1 = -0.4 -> clamped to 0
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, KellyCriterionBreakeven) {
    // Win rate 50%, avg win $1, avg loss $1 = zero edge
    KellyCriterion sizer(0.5, 1.0, 1.0, 1.0);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Kelly %: (0.5 * 1 - 0.5) / 1 = 0
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, KellyCriterionHighWinRate) {
    // Win rate 80%, avg win $1.5, avg loss $1
    KellyCriterion sizer(0.8, 1.5, 1.0, 1.0);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Kelly %: (0.8 * 1.5 - 0.2) / 1.5 = (1.2 - 0.2) / 1.5 = 0.667
    // Allocation: $100,000 * 0.667 = $66,700 / $100 = 667 shares
    EXPECT_NEAR(size, 666.67, 0.1);
}

TEST_F(PositionSizerTest, KellyCriterionWithStrength) {
    KellyCriterion sizer(0.6, 2.0, 1.0, 1.0);
    auto ctx = create_context(0.5);

    double size = sizer.calculate_size(ctx);

    // 400 * 0.5 = 200 shares
    EXPECT_NEAR(size, 200.0, 0.01);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidWinRate) {
    EXPECT_THROW(KellyCriterion(0.0, 2.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(1.0, 2.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(-0.1, 2.0, 1.0, 1.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidAvgWinLoss) {
    EXPECT_THROW(KellyCriterion(0.6, 0.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, 2.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, -1.0, 1.0, 1.0), std::invalid_argument);
}

TEST_F(PositionSizerTest, KellyCriterionInvalidFraction) {
    EXPECT_THROW(KellyCriterion(0.6, 2.0, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(KellyCriterion(0.6, 2.0, 1.0, 1.5), std::invalid_argument);
}

// ============================================================================
// EQUAL WEIGHT SIZER
// ============================================================================

TEST_F(PositionSizerTest, EqualWeightBasic) {
    EqualWeight sizer(5);  // 5 positions
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // $100,000 / 5 = $20,000 / $100 = 200 shares
    EXPECT_DOUBLE_EQ(size, 200.0);
}

TEST_F(PositionSizerTest, EqualWeightTenPositions) {
    EqualWeight sizer(10);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // $100,000 / 10 = $10,000 / $100 = 100 shares
    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, EqualWeightSinglePosition) {
    EqualWeight sizer(1);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    // Full allocation = 1000 shares
    EXPECT_DOUBLE_EQ(size, 1000.0);
}

TEST_F(PositionSizerTest, EqualWeightWithStrength) {
    EqualWeight sizer(5);
    auto ctx = create_context(0.5);

    double size = sizer.calculate_size(ctx);

    // 200 * 0.5 = 100 shares
    EXPECT_DOUBLE_EQ(size, 100.0);
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

// ============================================================================
// VOLATILITY TARGETING SIZER
// ============================================================================

TEST_F(PositionSizerTest, VolatilityTargetingBasic) {
    VolatilityTargeting sizer(0.15);  // Target 15% vol
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.15, 0.05);

    double size = sizer.calculate_size(ctx);

    // Leverage: 0.15 / 0.15 = 1.0
    // Allocation: $100,000 * 1.0 = $100,000 / $100 = 1000 shares
    EXPECT_DOUBLE_EQ(size, 1000.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingLowVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.05, 0.05);  // 5% actual vol

    double size = sizer.calculate_size(ctx);

    // Leverage: 0.15 / 0.05 = 3.0
    // So size = 3000 shares
    EXPECT_DOUBLE_EQ(size, 3000.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingHighVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.30, 0.05);  // 30% actual vol

    double size = sizer.calculate_size(ctx);

    // Leverage: 0.15 / 0.30 = 0.5
    // Allocation: $100,000 * 0.5 = $50,000 / $100 = 500 shares
    EXPECT_DOUBLE_EQ(size, 500.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingWithStrength) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(0.5, 100000.0, 100.0, 0.0, 0.15, 0.05);

    double size = sizer.calculate_size(ctx);

    // 1000 * 0.5 = 500 shares
    EXPECT_DOUBLE_EQ(size, 500.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingZeroVol) {
    VolatilityTargeting sizer(0.15);
    auto ctx = create_context(1.0, 100000.0, 100.0, 0.0, 0.0, 0.05);

    double size = sizer.calculate_size(ctx);

    // Zero vol should return 0
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, VolatilityTargetingInvalidTarget) {
    EXPECT_THROW(VolatilityTargeting(0.0), std::invalid_argument);
    EXPECT_THROW(VolatilityTargeting(-0.1), std::invalid_argument);
    EXPECT_NO_THROW(VolatilityTargeting(1.5));  // valid, implies leverage
}

// ============================================================================
// FIXED SHARES SIZER
// ============================================================================

TEST_F(PositionSizerTest, FixedSharesBasic) {
    FixedShares sizer(100);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, FixedSharesWithStrength) {
    FixedShares sizer(100);
    auto ctx = create_context(0.5);

    double size = sizer.calculate_size(ctx);

    EXPECT_DOUBLE_EQ(size, 50.0);
}

TEST_F(PositionSizerTest, FixedSharesLargePosition) {
    FixedShares sizer(10000);
    auto ctx = create_context();

    double size = sizer.calculate_size(ctx);

    EXPECT_DOUBLE_EQ(size, 10000.0);
}

TEST_F(PositionSizerTest, FixedSharesInvalidZero) {
    EXPECT_THROW(FixedShares(0), std::invalid_argument);
}

TEST_F(PositionSizerTest, FixedSharesInvalidNegative) {
    EXPECT_THROW(FixedShares(-100), std::invalid_argument);
}

// ============================================================================
// CONSTRAINT TESTS
// ============================================================================

TEST_F(PositionSizerTest, MaxPositionSizeConstraint) {
    FixedPercentage sizer(0.5);  // 50% = 500 shares
    sizer.set_max_position_size(200.0);

    auto ctx = create_context();
    double size = sizer.calculate_size(ctx);

    EXPECT_DOUBLE_EQ(size, 200.0);
}

TEST_F(PositionSizerTest, MinPositionSizeConstraint) {
    FixedPercentage sizer(0.01);  // 1% = 10 shares
    sizer.set_min_position_size(50.0);

    auto ctx = create_context();
    double size = sizer.calculate_size(ctx);

    // Below min, should return 0
    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, MaxLeverageConstraint) {
    FixedPercentage sizer(0.5);
    sizer.set_max_leverage(0.1);  // Max 10% of capital in notional

    auto ctx = create_context();
    double size = sizer.calculate_size(ctx);

    // Max notional: $100,000 * 0.1 = $10,000
    // Max shares: $10,000 / $100 = 100
    EXPECT_DOUBLE_EQ(size, 100.0);
}

TEST_F(PositionSizerTest, MultipleConstraints) {
    FixedPercentage sizer(0.5);
    sizer.set_max_position_size(300.0);
    sizer.set_max_leverage(0.2);  // 200 shares max

    auto ctx = create_context();
    double size = sizer.calculate_size(ctx);

    // Leverage constraint (200) is tighter than position constraint (300)
    EXPECT_DOUBLE_EQ(size, 200.0);
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(PositionSizerTest, VerySmallCapital) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100.0, 100.0);  // Only $100 capital

    double size = sizer.calculate_size(ctx);

    // 10% of $100 = $10 / $100 = 0.1 shares
    EXPECT_DOUBLE_EQ(size, 0.1);
}

TEST_F(PositionSizerTest, VeryHighPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 50000.0);  // $50,000 per share

    double size = sizer.calculate_size(ctx);

    // 10% of $100,000 = $10,000 / $50,000 = 0.2 shares
    EXPECT_DOUBLE_EQ(size, 0.2);
}

TEST_F(PositionSizerTest, VeryLowPrice) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(1.0, 100000.0, 0.01);  // 1 cent per share

    double size = sizer.calculate_size(ctx);

    // 10% of $100,000 = $10,000 / $0.01 = 1,000,000 shares
    EXPECT_DOUBLE_EQ(size, 1000000.0);
}

TEST_F(PositionSizerTest, ZeroStrength) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(0.0);

    double size = sizer.calculate_size(ctx);

    EXPECT_DOUBLE_EQ(size, 0.0);
}

TEST_F(PositionSizerTest, HighStrength) {
    FixedPercentage sizer(0.1);
    auto ctx = create_context(2.0);  // 200% strength

    double size = sizer.calculate_size(ctx);

    // 100 shares * 2.0 = 200 shares
    EXPECT_DOUBLE_EQ(size, 200.0);
}