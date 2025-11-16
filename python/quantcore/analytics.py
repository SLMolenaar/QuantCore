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
    max_drawdown_duration: int

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
            f"  Max DD Duration:       {self.max_drawdown_duration:>12} days",
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
    Calculate returns from equity curve
    """
    if len(equity_curve) < 2:
        return np.array([])

    returns = np.diff(equity_curve) / equity_curve[:-1]
    return returns


def calculate_total_return(equity_curve: np.ndarray) -> float:
    """
    Total return as percentage
    """
    if len(equity_curve) < 2:
        return 0.0

    return ((equity_curve[-1] / equity_curve[0]) - 1.0) * 100.0


def calculate_annualized_return(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized return
    """
    if len(equity_curve) < 2:
        return 0.0

    total_return = equity_curve[-1] / equity_curve[0]
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    annualized = (total_return ** (1.0 / years) - 1.0) * 100.0
    return annualized


def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized volatility as percentage
    """
    if len(returns) == 0:
        return 0.0

    return np.std(returns) * np.sqrt(periods_per_year) * 100.0


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    # Handle zero or near-zero volatility
    if std_excess < 1e-10:
        return 0.0

    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
) -> float:
    """
    Sortino ratio - uses downside deviation instead of total volatility
    """
    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    # only negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return sortino


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int]:
    """
    Maximum drawdown and its duration

    Returns (max_drawdown_pct, duration_in_periods)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    max_dd = np.min(drawdown) * 100.0

    # find duration of maximum drawdown
    max_dd_idx = np.argmin(drawdown)

    # find peak before max drawdown
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if equity_curve[i] == running_max[max_dd_idx]:
            peak_idx = i
            break

    # find when equity recovered (if it did)
    recovery_idx = len(equity_curve)
    peak_value = equity_curve[peak_idx]
    for i in range(max_dd_idx + 1, len(equity_curve)):
        if equity_curve[i] >= peak_value:
            recovery_idx = i
            break

    duration = recovery_idx - peak_idx

    return max_dd, duration


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calmar ratio = annualized return / max drawdown
    """
    if max_drawdown >= 0:
        return 0.0

    if abs(max_drawdown) < 0.001:  # Avoid division by near-zero
        return 0.0

    return annualized_return / abs(max_drawdown)


def analyze_trades(trade_pnls: List[float]) -> Dict[str, float]:
    """
    Analyze individual trade performance
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

    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]

    total_trades = len(trade_pnls)
    win_count = len(wins)
    loss_count = len(losses)

    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    total_wins = sum(wins) if wins else 0.0
    total_losses = abs(sum(losses)) if losses else 0.0

    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

    largest_win = max(wins) if wins else 0.0
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
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate all performance metrics from equity curve
    """
    returns = calculate_returns(equity_curve)

    # return metrics
    total_return = calculate_total_return(equity_curve)
    annualized_return = calculate_annualized_return(equity_curve, periods_per_year)

    # risk metrics
    volatility = calculate_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(annualized_return, max_dd)

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
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        **trade_metrics
    )


def rolling_sharpe(
        returns: np.ndarray,
        window: int = 60,
        periods_per_year: int = 252
) -> np.ndarray:
    """
    Rolling Sharpe ratio
    """
    if len(returns) < window:
        return np.array([])

    rolling_sharpes = []

    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        sharpe = calculate_sharpe_ratio(window_returns, 0.0, periods_per_year)
        rolling_sharpes.append(sharpe)

    return np.array(rolling_sharpes)


def rolling_volatility(
        returns: np.ndarray,
        window: int = 60,
        periods_per_year: int = 252
) -> np.ndarray:
    """
    Rolling volatility
    """
    if len(returns) < window:
        return np.array([])

    rolling_vols = []

    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        vol = calculate_volatility(window_returns, periods_per_year)
        rolling_vols.append(vol)

    return np.array(rolling_vols)


def monthly_returns(equity_curve: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
    """
    Calculate monthly returns table

    Returns DataFrame with monthly returns by year
    """
    df = pd.DataFrame({
        'timestamp': timestamps,
        'equity': equity_curve
    })

    # convert timestamps if needed
    if df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

    df.set_index('timestamp', inplace=True)

    # calculate returns
    df['returns'] = df['equity'].pct_change()

    # resample to monthly
    monthly = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

    # pivot to year x month table
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
    Calculate drawdown data for underwater plot
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return drawdown