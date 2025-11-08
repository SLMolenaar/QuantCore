"""
Position sizing utilities for strategy development

Provides common position sizing methods:
- Fixed percentage of capital
- Risk-based sizing (using stop-loss distance)
- Kelly criterion
- Equal weight allocation
- Volatility-adjusted sizing
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    quantity: float
    notional_value: float
    percent_of_capital: float
    reasoning: str


class PositionSizer:
    """
    Calculate position sizes for orders

    Different sizing methods for different trading styles:
    - Fixed % for simplicity
    - Risk-based for defined risk per trade
    - Kelly for optimal growth (theoretical)
    - Equal weight for diversification
    - Volatility-adjusted for risk parity
    """

    def __init__(self, capital: float, max_position_pct: float = 0.2):
        """
        Initialize position sizer

        Args:
            capital: Total capital available for trading
            max_position_pct: Maximum position size as % of capital (default 20%)
        """
        self.capital = capital
        self.max_position_pct = max_position_pct

    def update_capital(self, new_capital: float):
        """Update available capital"""
        self.capital = new_capital

    def fixed_percentage(
            self,
            price: float,
            percentage: float = 0.1,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Fixed percentage of capital

        Simple and commonly used. Allocate fixed % of capital to each position.

        Args:
            price: Current price per unit
            percentage: Percentage of capital to use (default 10%)
            min_quantity: Minimum quantity to trade

        Returns:
            PositionSizeResult with calculated quantity
        """
        percentage = min(percentage, self.max_position_pct)
        notional = self.capital * percentage
        quantity = notional / price

        quantity = max(quantity, min_quantity)
        actual_notional = quantity * price
        actual_pct = actual_notional / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_pct,
            reasoning=f"Fixed {percentage * 100:.1f}% allocation"
        )

    def risk_based(
            self,
            price: float,
            stop_loss_price: float,
            risk_per_trade: float = 0.01,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Risk-based position sizing

        Size position so that if stop-loss is hit, loss = risk_per_trade * capital.
        Popular for systematic traders.

        Args:
            price: Entry price
            stop_loss_price: Stop-loss price
            risk_per_trade: Max risk as % of capital (default 1%)
            min_quantity: Minimum quantity

        Returns:
            PositionSizeResult with calculated quantity
        """
        if price == stop_loss_price:
            raise ValueError("Entry price cannot equal stop-loss price")

        risk_amount = self.capital * risk_per_trade
        risk_per_unit = abs(price - stop_loss_price)

        quantity = risk_amount / risk_per_unit
        quantity = max(quantity, min_quantity)

        notional = quantity * price

        if notional > self.capital * self.max_position_pct:
            max_notional = self.capital * self.max_position_pct
            quantity = max_notional / price
            notional = quantity * price

        actual_risk_pct = (quantity * risk_per_unit) / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=notional,
            percent_of_capital=notional / self.capital,
            reasoning=f"Risk {actual_risk_pct * 100:.2f}% if stop at ${stop_loss_price:.2f}"
        )

    def kelly_criterion(
            self,
            price: float,
            win_rate: float,
            avg_win: float,
            avg_loss: float,
            kelly_fraction: float = 0.25,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Kelly criterion position sizing

        Optimal sizing for maximum geometric growth (theoretically).
        Most traders use a fraction of full Kelly to reduce volatility.

        Formula: f = (p*W - (1-p)*L) / (W*L)
        where p=win_rate, W=avg_win/avg_loss

        Args:
            price: Current price
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
            min_quantity: Minimum quantity

        Returns:
            PositionSizeResult with calculated quantity
        """
        if avg_loss == 0:
            raise ValueError("Average loss cannot be zero")

        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        kelly_pct = max(0, kelly_pct)

        adjusted_kelly = kelly_pct * kelly_fraction

        adjusted_kelly = min(adjusted_kelly, self.max_position_pct)

        notional = self.capital * adjusted_kelly
        quantity = notional / price
        quantity = max(quantity, min_quantity)

        actual_notional = quantity * price
        actual_pct = actual_notional / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_pct,
            reasoning=f"Kelly {kelly_pct * 100:.1f}% (using {kelly_fraction * 100:.0f}% of full Kelly)"
        )

    def equal_weight(
            self,
            price: float,
            num_positions: int,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Equal weight allocation

        Divide capital equally among N positions.
        Simple diversification strategy.

        Args:
            price: Current price
            num_positions: Total number of positions in portfolio
            min_quantity: Minimum quantity

        Returns:
            PositionSizeResult with calculated quantity
        """
        if num_positions <= 0:
            raise ValueError("Number of positions must be positive")

        weight_per_position = 1.0 / num_positions
        weight_per_position = min(weight_per_position, self.max_position_pct)

        notional = self.capital * weight_per_position
        quantity = notional / price
        quantity = max(quantity, min_quantity)

        actual_notional = quantity * price
        actual_pct = actual_notional / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_pct,
            reasoning=f"Equal weight: 1/{num_positions} positions"
        )

    def volatility_adjusted(
            self,
            price: float,
            volatility: float,
            target_volatility: float = 0.15,
            base_allocation: float = 0.1,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Volatility-adjusted position sizing

        Scale position size inversely with volatility.
        Used in risk parity strategies.

        Args:
            price: Current price
            volatility: Asset's annualized volatility (e.g., 0.25 = 25%)
            target_volatility: Target portfolio volatility (default 15%)
            base_allocation: Base allocation at target volatility
            min_quantity: Minimum quantity

        Returns:
            PositionSizeResult with calculated quantity
        """
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        vol_scalar = target_volatility / volatility

        allocation = base_allocation * vol_scalar
        allocation = min(allocation, self.max_position_pct)

        notional = self.capital * allocation
        quantity = notional / price
        quantity = max(quantity, min_quantity)

        actual_notional = quantity * price
        actual_pct = actual_notional / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_pct,
            reasoning=f"Vol-adjusted: {volatility * 100:.1f}% vol → {allocation * 100:.1f}% allocation"
        )

    def leveraged(
            self,
            price: float,
            leverage: float,
            base_percentage: float = 0.1,
            min_quantity: int = 1
    ) -> PositionSizeResult:
        """
        Leveraged position sizing

        Apply leverage to base allocation. Use with caution.

        Args:
            price: Current price
            leverage: Leverage multiplier (e.g., 2.0 = 2x leverage)
            base_percentage: Base allocation before leverage
            min_quantity: Minimum quantity

        Returns:
            PositionSizeResult with calculated quantity
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        target_notional = self.capital * base_percentage * leverage

        max_notional = self.capital * self.max_position_pct * leverage
        target_notional = min(target_notional, max_notional)

        quantity = target_notional / price
        quantity = max(quantity, min_quantity)

        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"{leverage}x leverage on {base_percentage * 100:.1f}% allocation"
        )


class PortfolioPositionSizer:
    """
    Multi-asset position sizing

    Manage position sizes across multiple assets simultaneously.
    Ensures total exposure doesn't exceed limits.
    """

    def __init__(
            self,
            capital: float,
            max_total_exposure: float = 1.0,
            max_single_position: float = 0.2
    ):
        """
        Initialize portfolio position sizer

        Args:
            capital: Total capital
            max_total_exposure: Max total exposure as multiple of capital (1.0 = 100%, no leverage)
            max_single_position: Max single position as % of capital
        """
        self.capital = capital
        self.max_total_exposure = max_total_exposure
        self.max_single_position = max_single_position
        self.current_positions: Dict[str, float] = {}

    def update_position(self, symbol: str, notional_value: float):
        """Update position value for a symbol"""
        if notional_value == 0:
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = notional_value

    def get_total_exposure(self) -> float:
        """Get current total exposure as multiple of capital"""
        total = sum(abs(v) for v in self.current_positions.values())
        return total / self.capital if self.capital > 0 else 0

    def get_available_capital(self) -> float:
        """Get remaining capital available for new positions"""
        used = sum(abs(v) for v in self.current_positions.values())
        max_allowed = self.capital * self.max_total_exposure
        return max(0, max_allowed - used)

    def can_add_position(
            self,
            symbol: str,
            notional_value: float
    ) -> tuple[bool, str]:
        """
        Check if new position can be added

        Returns:
            (can_add, reason)
        """
        current_notional = self.current_positions.get(symbol, 0)
        new_total = abs(current_notional) + abs(notional_value)

        if new_total > self.capital * self.max_single_position:
            return False, f"Would exceed single position limit ({self.max_single_position * 100:.0f}%)"

        total_exposure = self.get_total_exposure()
        additional_exposure = abs(notional_value) / self.capital

        if total_exposure + additional_exposure > self.max_total_exposure:
            return False, f"Would exceed total exposure limit ({self.max_total_exposure * 100:.0f}%)"

        return True, "Position allowed"

    def size_new_position(
            self,
            symbol: str,
            price: float,
            desired_percentage: float,
            sizing_method: str = "fixed"
    ) -> Optional[PositionSizeResult]:
        """
        Size new position respecting portfolio constraints

        Args:
            symbol: Asset symbol
            price: Current price
            desired_percentage: Desired allocation %
            sizing_method: Method to use

        Returns:
            PositionSizeResult or None if not possible
        """
        available = self.get_available_capital()

        if available <= 0:
            return None

        max_for_asset = min(
            available,
            self.capital * self.max_single_position
        )

        desired_notional = self.capital * desired_percentage
        actual_notional = min(desired_notional, max_for_asset)

        if actual_notional <= 0:
            return None

        quantity = actual_notional / price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"Portfolio-constrained: {self.get_total_exposure() * 100:.1f}% exposure used"
        )