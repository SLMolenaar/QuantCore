"""
Position sizing utilities for strategy development.

These are standalone Python helpers for calculating order sizes before
submitting them to the engine. They are separate from the C++ PositionSizer
hierarchy (exposed as quantcore.PositionSizer) which is used internally by
BacktestEngine.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result from a position size calculation."""
    quantity:            float
    notional_value:      float
    percent_of_capital:  float
    reasoning:           str


class PositionCalculator:
    """
    Calculate position sizes for orders.

    Supports multiple sizing methods depending on strategy needs.
    All methods respect a configurable maximum position size as a
    percentage of total capital.
    """

    def __init__(self, capital: float, max_position_pct: float = 0.2):
        """
        Args:
            capital:          Total capital available.
            max_position_pct: Hard cap on any single position as a fraction of capital.
        """
        if capital <= 0:
            raise ValueError("Capital must be positive")
        self.capital          = capital
        self.max_position_pct = max_position_pct

    def update_capital(self, new_capital: float) -> None:
        """Refresh the capital figure used for sizing calculations."""
        if new_capital <= 0:
            raise ValueError("Capital must be positive")
        self.capital = new_capital

    def _validate_price(self, price: float) -> None:
        if price <= 0:
            raise ValueError("Price must be positive")

    def fixed_percentage(
            self,
            price:        float,
            percentage:   float = 0.1,
            min_quantity: int   = 1,
    ) -> PositionSizeResult:
        """Allocate a fixed fraction of capital to the position."""
        self._validate_price(price)

        percentage    = min(percentage, self.max_position_pct)
        notional      = self.capital * percentage
        quantity      = max(notional / price, min_quantity)
        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"Fixed {percentage * 100:.1f}% allocation",
        )

    def risk_based(
            self,
            price:           float,
            stop_loss_price: float,
            risk_per_trade:  float = 0.01,
            min_quantity:    int   = 1,
    ) -> PositionSizeResult:
        """
        Size a position so that hitting the stop loss costs exactly
        `risk_per_trade * capital`.
        """
        self._validate_price(price)
        if stop_loss_price <= 0:
            raise ValueError("Stop loss price must be positive")
        if price == stop_loss_price:
            raise ValueError("Entry price cannot equal stop-loss price")

        risk_amount   = self.capital * risk_per_trade
        risk_per_unit = abs(price - stop_loss_price)
        quantity      = risk_amount / risk_per_unit
        quantity      = max(quantity, min_quantity)
        notional      = quantity * price

        max_notional = self.capital * self.max_position_pct
        if notional > max_notional:
            quantity = max_notional / price
            notional = quantity * price

        actual_risk_pct = (quantity * risk_per_unit) / self.capital

        return PositionSizeResult(
            quantity=quantity,
            notional_value=notional,
            percent_of_capital=notional / self.capital,
            reasoning=f"Risk {actual_risk_pct * 100:.2f}% if stop at ${stop_loss_price:.2f}",
        )

    def kelly_criterion(
            self,
            price:          float,
            win_rate:       float,
            avg_win:        float,
            avg_loss:       float,
            kelly_fraction: float = 0.25,
            min_quantity:   int   = 1,
    ) -> PositionSizeResult:
        """
        Kelly criterion sizing for optimal long-run growth.

        A fractional Kelly (default 0.25) is recommended in practice to
        reduce variance while preserving most of the growth benefit.
        """
        self._validate_price(price)
        if avg_loss == 0:
            raise ValueError("Average loss cannot be zero")
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")

        wl_ratio   = avg_win / avg_loss
        kelly_pct  = max(0.0, (win_rate * wl_ratio - (1 - win_rate)) / wl_ratio)
        adjusted   = min(kelly_pct * kelly_fraction, self.max_position_pct)

        notional  = self.capital * adjusted
        quantity  = max(notional / price, min_quantity)
        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"Kelly {kelly_pct * 100:.1f}% (using {kelly_fraction * 100:.0f}% of full Kelly)",
        )

    def equal_weight(
            self,
            price:         float,
            num_positions: int,
            min_quantity:  int = 1,
    ) -> PositionSizeResult:
        """Divide capital equally among `num_positions` positions."""
        self._validate_price(price)
        if num_positions <= 0:
            raise ValueError("Number of positions must be positive")

        weight   = min(1.0 / num_positions, self.max_position_pct)
        notional = self.capital * weight
        quantity = max(notional / price, min_quantity)
        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"Equal weight: 1/{num_positions} positions",
        )

    def volatility_adjusted(
            self,
            price:             float,
            volatility:        float,
            target_volatility: float = 0.15,
            base_allocation:   float = 0.1,
            min_quantity:      int   = 1,
    ) -> PositionSizeResult:
        """
        Scale the allocation inversely with realised volatility so that
        each position contributes approximately equal risk.
        """
        self._validate_price(price)
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        allocation = min(base_allocation * (target_volatility / volatility),
                         self.max_position_pct)
        notional   = self.capital * allocation
        quantity   = max(notional / price, min_quantity)
        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=(
                f"Vol-adjusted: {volatility * 100:.1f}% vol "
                f"→ {allocation * 100:.1f}% allocation"
            ),
        )

    def leveraged(
            self,
            price:           float,
            leverage:        float,
            base_percentage: float = 0.1,
            min_quantity:    int   = 1,
    ) -> PositionSizeResult:
        """Apply leverage to a base allocation."""
        self._validate_price(price)
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        target_notional = min(
            self.capital * base_percentage * leverage,
            self.capital * self.max_position_pct * leverage,
            )
        quantity = max(target_notional / price, min_quantity)
        actual_notional = quantity * price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=f"{leverage}x leverage on {base_percentage * 100:.1f}% allocation",
        )


class PortfolioPositionSizer:
    """
    Multi-asset position sizing with portfolio-level exposure constraints.

    Tracks notional exposure across all open positions and prevents any
    new order from breaching single-asset or total-portfolio limits.
    """

    def __init__(
            self,
            capital:              float,
            max_total_exposure:   float = 1.0,
            max_single_position:  float = 0.2,
    ):
        if capital <= 0:
            raise ValueError("Capital must be positive")
        self.capital             = capital
        self.max_total_exposure  = max_total_exposure
        self.max_single_position = max_single_position
        self.current_positions:  Dict[str, float] = {}

    def update_position(self, symbol: str, notional_value: float) -> None:
        """Record or clear the notional value of an open position."""
        if notional_value == 0:
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = notional_value

    def get_total_exposure(self) -> float:
        """Current gross exposure as a multiple of capital."""
        total = sum(abs(v) for v in self.current_positions.values())
        return total / self.capital if self.capital > 0 else 0.0

    def get_available_capital(self) -> float:
        """Remaining notional budget for new positions."""
        used        = sum(abs(v) for v in self.current_positions.values())
        max_allowed = self.capital * self.max_total_exposure
        return max(0.0, max_allowed - used)

    def can_add_position(self, symbol: str, notional_value: float) -> tuple[bool, str]:
        """Check whether a new position fits within all portfolio constraints."""
        current  = self.current_positions.get(symbol, 0.0)
        combined = abs(current) + abs(notional_value)

        if combined > self.capital * self.max_single_position:
            return False, f"Would exceed single-position limit ({self.max_single_position * 100:.0f}%)"

        additional = abs(notional_value) / self.capital
        if self.get_total_exposure() + additional > self.max_total_exposure:
            return False, f"Would exceed total exposure limit ({self.max_total_exposure * 100:.0f}%)"

        return True, "Position allowed"

    def size_new_position(
            self,
            symbol:             str,
            price:              float,
            desired_percentage: float,
    ) -> Optional[PositionSizeResult]:
        """
        Size a new position respecting portfolio constraints.

        Returns None if no additional exposure is available.
        """
        if price <= 0:
            raise ValueError("Price must be positive")

        available    = self.get_available_capital()
        if available <= 0:
            return None

        max_for_asset   = min(available, self.capital * self.max_single_position)
        desired_notional = self.capital * desired_percentage
        actual_notional  = min(desired_notional, max_for_asset)

        if actual_notional <= 0:
            return None

        quantity = actual_notional / price

        return PositionSizeResult(
            quantity=quantity,
            notional_value=actual_notional,
            percent_of_capital=actual_notional / self.capital,
            reasoning=(
                f"Portfolio-constrained: "
                f"{self.get_total_exposure() * 100:.1f}% exposure used"
            ),
        )