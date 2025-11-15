"""
Position sizing utilities for strategy development

Common methods for calculating position sizes based on capital, risk, and strategy parameters.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result from position size calculation"""
    quantity: float
    notional_value: float
    percent_of_capital: float
    reasoning: str


class PositionSizer:
    """
    Calculate position sizes for orders

    Supports multiple sizing methods depending on strategy needs.
    """

    def __init__(self, capital: float, max_position_pct: float = 0.2):
        """
        Args:
            capital: Total capital available
            max_position_pct: Maximum position size as % of capital
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
        Allocate a fixed % of capital to the position
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
        Size position based on risk per trade

        If stop loss is hit, loss will equal risk_per_trade * capital
        """
        if price == stop_loss_price:
            raise ValueError("Entry price cannot equal stop-loss price")

        risk_amount = self.capital * risk_per_trade
        risk_per_unit = abs(price - stop_loss_price)

        quantity = risk_amount / risk_per_unit
        quantity = max(quantity, min_quantity)

        notional = quantity * price

        # cap at max position size
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
        Kelly criterion sizing for optimal growth

        Usually use a fraction of full Kelly to reduce volatility.
        Formula: f = (p*W - (1-p)*L) / (W*L) where p=win_rate, W=avg_win/avg_loss
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
        Divide capital equally among N positions
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
        Scale position inversely with volatility for risk parity
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
        Apply leverage to base allocation
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
    Multi-asset position sizing with portfolio-level constraints

    Tracks exposure across assets to ensure limits aren't exceeded.
    """

    def __init__(
            self,
            capital: float,
            max_total_exposure: float = 1.0,
            max_single_position: float = 0.2
    ):
        """
        Args:
            capital: Total capital
            max_total_exposure: Max total exposure as multiple of capital
            max_single_position: Max single position as % of capital
        """
        self.capital = capital
        self.max_total_exposure = max_total_exposure
        self.max_single_position = max_single_position
        self.current_positions: Dict[str, float] = {}

    def update_position(self, symbol: str, notional_value: float):
        """Update or remove position for a symbol"""
        if notional_value == 0:
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = notional_value

    def get_total_exposure(self) -> float:
        """Current total exposure as multiple of capital"""
        total = sum(abs(v) for v in self.current_positions.values())
        return total / self.capital if self.capital > 0 else 0

    def get_available_capital(self) -> float:
        """Remaining capital for new positions"""
        used = sum(abs(v) for v in self.current_positions.values())
        max_allowed = self.capital * self.max_total_exposure
        return max(0, max_allowed - used)

    def can_add_position(
            self,
            symbol: str,
            notional_value: float
    ) -> tuple[bool, str]:
        """
        Check if new position fits within constraints

        Returns (can_add, reason)
        """
        current_notional = self.current_positions.get(symbol, 0)
        new_total = abs(current_notional) + abs(notional_value)

        # check single position limit
        if new_total > self.capital * self.max_single_position:
            return False, f"Would exceed single position limit ({self.max_single_position * 100:.0f}%)"

        # check total exposure limit
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

        Returns None if position can't be added
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