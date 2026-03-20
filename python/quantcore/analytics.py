"""
Performance analytics for backtesting results

Standard metrics for evaluating strategy performance:
- Returns (total, annualized, rolling)
- Risk metrics (Sharpe, Sortino, volatility, drawdown)
- Trade analysis (win rate, profit factor, avg trade)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


MAX_RATIO_VALUE = 99.99


def infer_periods_per_year(timestamps: np.ndarray) -> int:
    """
    Estimate the number of equity snapshots per year from a timestamp array.

    Measures the median inter-snapshot interval and divides one year (in
    nanoseconds) by it. The result is rounded to the nearest integer and
    clamped to [1, 525_960] (1 per year to 1 per minute).

    Common outputs:
      ~252        daily bars
      ~1_260      weekly bars (but these would read as ~252*5)
      ~362_880    1-minute snapshots  (252 * 1440)
      ~525_960    1-second snapshots  (252 * 24 * 60 * 60 / ~1.16)

    Falls back to 252 when fewer than 2 timestamps are provided or the
    median gap is zero.
    """
    if timestamps is None or len(timestamps) < 2:
        return 252

    gaps = np.diff(timestamps.astype(np.int64))
    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return 252

    median_gap_ns  = float(np.median(positive_gaps))
    if median_gap_ns <= 0:
        return 252

    ns_per_year    = 365.25 * 24 * 3600 * 1_000_000_000
    periods        = ns_per_year / median_gap_ns
    return int(np.clip(round(periods), 1, 525_960))


@dataclass
class PerformanceMetrics:
    """Container for backtest metrics"""

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # calendar days; -1 if no timestamps were provided

    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Other
    calmar_ratio: float

    def __str__(self) -> str:
        """Format for printing"""
        dd_dur = (
            f"{self.max_drawdown_duration} days"
            if self.max_drawdown_duration >= 0
            else "n/a (no timestamps)"
        )
        lines = [
            "=" * 60,
            "  Performance Metrics",
            "=" * 60,
            "",
            "Returns:",
            f"  Total Return:          {self.total_return:>12.2f}%",
            f"  Annualized Return:     {self.annualized_return:>12.2f}%",
            "",
            "Risk Metrics:",
            f"  Sharpe Ratio:          {self.sharpe_ratio:>12.2f}",
            f"  Sortino Ratio:         {self.sortino_ratio:>12.2f}",
            f"  Volatility:            {self.volatility:>12.2f}%",
            f"  Max Drawdown:          {self.max_drawdown:>12.2f}%",
            f"  Max DD Duration:       {dd_dur:>12}",
            f"  Calmar Ratio:          {self.calmar_ratio:>12.2f}",
            "",
            "Trading Metrics:",
            f"  Total Trades:          {self.total_trades:>12}",
            f"  Win Rate:              {self.win_rate:>12.2f}%",
            f"  Profit Factor:         {self.profit_factor:>12.2f}",
            f"  Average Win:           ${self.avg_win:>11,.2f}",
            f"  Average Loss:          ${self.avg_loss:>11,.2f}",
            f"  Largest Win:           ${self.largest_win:>11,.2f}",
            f"  Largest Loss:          ${self.largest_loss:>11,.2f}",
            "",
            "=" * 60,
            ]
        return "\n".join(lines)


def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate period returns from an equity curve.
    """
    if len(equity_curve) < 2:
        return np.array([])

    returns = np.diff(equity_curve) / equity_curve[:-1]
    return returns


def calculate_total_return(equity_curve: np.ndarray) -> float:
    """
    Total return as percentage.
    """
    if len(equity_curve) < 2:
        return 0.0

    return ((equity_curve[-1] / equity_curve[0]) - 1.0) * 100.0


def calculate_annualized_return(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized return.
    """
    if len(equity_curve) < 2:
        return 0.0

    total_return = equity_curve[-1] / equity_curve[0]
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    # total_return is the growth factor (e.g. 0.5 means portfolio halved).
    # Taking a fractional power of a non-positive number is undefined in real
    # arithmetic and produces a RuntimeWarning + NaN. Sign-preserve instead:
    # treat the magnitude as if it were positive, compute the annualised rate,
    # then restore the sign.
    if total_return <= 0:
        annualized = -(abs(total_return) ** (1.0 / years) - 1.0) * 100.0
    else:
        annualized = (total_return ** (1.0 / years) - 1.0) * 100.0

    return annualized


def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized volatility as percentage.
    """
    if len(returns) == 0:
        return 0.0

    return np.std(returns) * np.sqrt(periods_per_year) * 100.0


def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe ratio.

    risk_free_rate is the annual rate (e.g. 0.02 = 2%).
    It is divided by periods_per_year before subtracting from each
    per-period return so the units are consistent.

    Returns 0.0 when std is effectively zero (all returns identical), which
    guards against divide-by-zero on fully flat equity curves.
    """
    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess < 1e-10:
        if mean_excess > 0:
            return MAX_RATIO_VALUE
        elif mean_excess < 0:
            return -MAX_RATIO_VALUE
        return 0.0

    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return float(np.clip(sharpe, -MAX_RATIO_VALUE, MAX_RATIO_VALUE))


def calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
) -> float:
    """
    Sortino ratio — uses downside deviation instead of total volatility.

    Returns 0.0 when std is effectively zero (see calculate_sharpe_ratio).
    """
    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    # only negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        if np.mean(excess_returns) > 0:
            return MAX_RATIO_VALUE
        return 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std < 1e-10:
        if np.mean(excess_returns) > 0:
            return MAX_RATIO_VALUE
        elif np.mean(excess_returns) < 0:
            return -MAX_RATIO_VALUE
        return 0.0

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return float(np.clip(sortino, -MAX_RATIO_VALUE, MAX_RATIO_VALUE))


def calculate_max_drawdown(
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Maximum drawdown percentage and duration.

    Args:
        equity_curve: Portfolio value series.
        timestamps:   Corresponding nanosecond UNIX timestamps. When provided,
                      the duration is returned in calendar days. When omitted
                      the duration is returned as -1 (indeterminate), because
                      counting equity snapshots is misleading for tick/minute
                      data where most snapshots show no change.

    Returns:
        (max_drawdown_pct, duration_in_calendar_days_or_minus_one)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100.0

    max_dd = float(np.min(drawdown))
    trough_idx = int(np.argmin(drawdown))

    # peak is the last all-time high before the trough
    peak_idx = int(np.argmax(equity_curve[: trough_idx + 1]))

    # look for full recovery after the trough
    recovery_idx = len(equity_curve) - 1  # default: no recovery by end of data
    for i in range(trough_idx, len(equity_curve)):
        if equity_curve[i] >= equity_curve[peak_idx]:
            recovery_idx = i
            break

    if timestamps is not None and len(timestamps) == len(equity_curve):
        # Convert nanosecond timestamps to calendar days.
        ns_per_day = 86_400 * 1_000_000_000
        duration = int((timestamps[recovery_idx] - timestamps[peak_idx]) / ns_per_day)
        duration = max(0, duration)
    else:
        # No timestamps: return sentinel so callers know the value is not in days.
        duration = -1

    return max_dd, duration


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calmar ratio: annualized return / abs(max drawdown).
    """
    if abs(max_drawdown) < 0.01:
        if annualized_return > 0:
            return MAX_RATIO_VALUE
        elif annualized_return < 0:
            return -MAX_RATIO_VALUE
        return 0.0

    calmar = annualized_return / abs(max_drawdown)
    return float(np.clip(calmar, -MAX_RATIO_VALUE, MAX_RATIO_VALUE))


def analyze_trades(trade_pnls: List[float]) -> dict:
    """
    Analyze individual trade results.

    Args:
        trade_pnls: List of PnL per trade (positive = win, negative = loss).
    """
    if not trade_pnls:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
        }

    total_trades = len(trade_pnls)
    wins   = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]

    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    avg_win  = np.mean(wins)   if wins   else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    total_wins   = sum(wins)        if wins   else 0.0
    total_losses = abs(sum(losses)) if losses else 0.0

    if total_losses > 0:
        profit_factor = min(total_wins / total_losses, MAX_RATIO_VALUE)
    else:
        profit_factor = MAX_RATIO_VALUE if total_wins > 0 else 0.0

    largest_win  = max(wins)   if wins   else 0.0
    largest_loss = min(losses) if losses else 0.0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
    }


def calculate_all_metrics(
        equity_curve: np.ndarray,
        trade_pnls: Optional[List[float]] = None,
        timestamps: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: Optional[int] = None,
) -> PerformanceMetrics:
    """
    Calculate all performance metrics from an equity curve.

    Args:
        equity_curve:     Portfolio value series.
        trade_pnls:       Per-trade PnL list for trade-level metrics.
        timestamps:       Nanosecond UNIX timestamps aligned with equity_curve.
                          Used for max_drawdown_duration (calendar days) and,
                          when periods_per_year is None, to auto-infer the
                          correct annualisation factor.
        risk_free_rate:   Annual risk-free rate (e.g. 0.02 = 2%).
        periods_per_year: Snapshots per year for annualisation. When None
                          (default), inferred from timestamps if provided,
                          otherwise falls back to 252. Pass an explicit value
                          to override (e.g. 252 for daily, 252*1440 for
                          1-minute snapshots).
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(timestamps) \
            if timestamps is not None else 252

    returns = calculate_returns(equity_curve)

    # return metrics
    total_return      = calculate_total_return(equity_curve)
    annualized_return = calculate_annualized_return(equity_curve, periods_per_year)

    # risk metrics
    volatility         = calculate_volatility(returns, periods_per_year)
    sharpe             = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino            = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, max_dd_dur = calculate_max_drawdown(equity_curve, timestamps)
    calmar             = calculate_calmar_ratio(annualized_return, max_dd)

    # trading metrics
    if trade_pnls is None:
        trade_metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
        }
    else:
        trade_metrics = analyze_trades(trade_pnls)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        volatility=volatility,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        calmar_ratio=calmar,
        **trade_metrics
    )


def rolling_sharpe(
        returns: np.ndarray,
        window: int = 60,
        periods_per_year: int = 252,
) -> np.ndarray:
    """
    Rolling Sharpe ratio.

    Windows where std is effectively zero (all returns identical, typically
    all zero between trades) produce NaN, which renders as a gap in the
    chart rather than a misleading zero or a spike.
    """
    if len(returns) < window:
        return np.array([])

    n = len(returns)
    result = np.empty(n - window + 1)

    window_returns  = returns[:window]
    running_sum     = np.sum(window_returns)
    running_sq_sum  = np.sum(window_returns ** 2)

    annualization = np.sqrt(periods_per_year)

    for i in range(n - window + 1):
        if i > 0:
            old_val = returns[i - 1]
            new_val = returns[i + window - 1]
            running_sum    += new_val - old_val
            running_sq_sum += new_val ** 2 - old_val ** 2

        mean = running_sum / window
        # Variance using E[X²] - E[X]²
        variance = (running_sq_sum / window) - (mean ** 2)

        # Bessel's correction for sample std
        if window > 1 and variance > 0:
            std = np.sqrt(variance * window / (window - 1))
        else:
            std = 0.0

        if std < 1e-10:
            # Truly flat window (all returns identical, usually all zero).
            # Return NaN so the chart renders a gap rather than a misleading 0.
            result[i] = np.nan
        else:
            sharpe = (mean / std) * annualization
            result[i] = np.clip(sharpe, -MAX_RATIO_VALUE, MAX_RATIO_VALUE)

    return result


def rolling_volatility(
        returns: np.ndarray,
        window: int = 60,
        periods_per_year: int = 252,
) -> np.ndarray:
    """
    Rolling annualized volatility (%).

    Windows that are entirely flat (all returns zero) produce 0.0, which is
    technically correct — there is no volatility — but callers should be aware
    that this can look like artificially smooth periods on tick/minute data.
    """
    if len(returns) < window:
        return np.array([])

    n = len(returns)
    result = np.empty(n - window + 1)

    window_returns = returns[:window]
    running_sum    = np.sum(window_returns)
    running_sq_sum = np.sum(window_returns ** 2)

    annualization = np.sqrt(periods_per_year) * 100.0

    for i in range(n - window + 1):
        if i > 0:
            old_val = returns[i - 1]
            new_val = returns[i + window - 1]
            running_sum    += new_val - old_val
            running_sq_sum += new_val ** 2 - old_val ** 2

        mean = running_sum / window
        # Variance using E[X²] - E[X]²
        variance = (running_sq_sum / window) - (mean ** 2)

        # population std (consistent with calculate_volatility)
        std = np.sqrt(max(0.0, variance))
        result[i] = std * annualization

    return result


def _get_month_end_offset() -> str:
    """
    Get the correct month-end offset string for the installed pandas.

    'ME' (month-end) was introduced in pandas 2.2 as replacement for 'M'.
    The old 'M' alias is deprecated in 2.2+ but still works, while 'ME'
    doesn't exist in pandas < 2.2.
    """
    pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
    return 'ME' if pandas_version >= (2, 2) else 'M'


def monthly_returns(equity_curve: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
    """
    Calculate monthly returns table.

    Returns a DataFrame with monthly returns by year (rows = years, cols = months).
    """
    df = pd.DataFrame({
        'timestamp': timestamps,
        'equity': equity_curve
    })

    if df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

    df.set_index('timestamp', inplace=True)

    df['returns'] = df['equity'].pct_change()

    month_offset = _get_month_end_offset()
    monthly = df['returns'].resample(month_offset).apply(lambda x: (1 + x).prod() - 1)

    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values * 100
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return pivot


def underwater_plot_data(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate drawdown data for an underwater plot.

    Returns an array of the same length as equity_curve where each value is
    the drawdown percentage from the running peak (always <= 0).
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return drawdown