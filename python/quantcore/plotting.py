"""
Visualization tools for backtesting results

Provides publication-quality plots:
- Equity curves with drawdown shading
- Returns distribution
- Rolling metrics (Sharpe, volatility)
- Monthly returns heatmap
- Trade analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime

# Set default style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def plot_equity_curve(
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Equity Curve",
        figsize: Tuple[int, int] = (14, 7),
        show_drawdown: bool = True
) -> plt.Figure:
    """
    Plot equity curve with optional drawdown shading

    Args:
        equity_curve: Array of portfolio values
        timestamps: Optional array of timestamps
        title: Plot title
        figsize: Figure size
        show_drawdown: Whether to shade drawdown periods

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if timestamps is not None:
        # Convert timestamps to datetime if needed
        if timestamps.dtype == 'int64':
            dates = pd.to_datetime(timestamps, unit='ns')
        else:
            dates = timestamps
        x = dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    else:
        x = np.arange(len(equity_curve))
        ax.set_xlabel("Time Period")

    # Plot equity curve
    ax.plot(x, equity_curve, linewidth=2, color='#2E86AB', label='Portfolio Value')

    # Shade drawdown periods
    if show_drawdown:
        running_max = np.maximum.accumulate(equity_curve)
        drawdown_mask = equity_curve < running_max

        if np.any(drawdown_mask):
            ax.fill_between(
                x, equity_curve, running_max,
                where=drawdown_mask,
                alpha=0.3,
                color='red',
                label='Drawdown'
            )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    return fig


def plot_underwater(
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Underwater Plot",
        figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot drawdown over time (underwater plot)

    Args:
        equity_curve: Array of portfolio values
        timestamps: Optional array of timestamps
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100.0

    if timestamps is not None:
        if timestamps.dtype == 'int64':
            dates = pd.to_datetime(timestamps, unit='ns')
        else:
            dates = timestamps
        x = dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    else:
        x = np.arange(len(equity_curve))

    ax.fill_between(x, drawdown, 0, alpha=0.3, color='red')
    ax.plot(x, drawdown, linewidth=1.5, color='darkred')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_returns_distribution(
        returns: np.ndarray,
        title: str = "Returns Distribution",
        figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot returns distribution histogram with statistics

    Args:
        returns: Array of returns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(returns * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    ax1.axvline(x=np.mean(returns) * 100, color='green', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(returns) * 100:.2f}%')

    ax1.set_title("Returns Histogram", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Return (%)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normality Check)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_rolling_metrics(
        returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window: int = 60,
        title: str = "Rolling Performance Metrics",
        figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio and volatility

    Args:
        returns: Array of returns
        timestamps: Optional array of timestamps
        window: Rolling window size
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from .analytics import rolling_sharpe, rolling_volatility

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Calculate rolling metrics
    roll_sharpe = rolling_sharpe(returns, window)
    roll_vol = rolling_volatility(returns, window)

    if timestamps is not None:
        if timestamps.dtype == 'int64':
            dates = pd.to_datetime(timestamps, unit='ns')
        else:
            dates = timestamps
        # Align with rolling window
        x = dates[window:]
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    else:
        x = np.arange(window, len(returns) + 1)

    # Rolling Sharpe
    ax1.plot(x, roll_sharpe, linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
    ax1.set_title(f"Rolling Sharpe Ratio (window={window})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Sharpe Ratio", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rolling Volatility
    ax2.plot(x, roll_vol, linewidth=2, color='#A23B72')
    ax2.set_title(f"Rolling Volatility (window={window})", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Volatility (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
        monthly_returns: pd.DataFrame,
        title: str = "Monthly Returns (%)",
        figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot monthly returns as heatmap

    Args:
        monthly_returns: DataFrame with years as rows, months as columns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(monthly_returns.values, cmap='RdYlGn', aspect='auto',
                   vmin=-10, vmax=10, interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(monthly_returns.columns)))
    ax.set_yticks(np.arange(len(monthly_returns.index)))
    ax.set_xticklabels(monthly_returns.columns)
    ax.set_yticklabels(monthly_returns.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)', rotation=270, labelpad=20, fontsize=12)

    # Add text annotations
    for i in range(len(monthly_returns.index)):
        for j in range(len(monthly_returns.columns)):
            value = monthly_returns.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 5 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                        color=text_color, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_trade_analysis(
        entry_prices: List[float],
        exit_prices: List[float],
        entry_dates: Optional[List] = None,
        title: str = "Trade Analysis",
        figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot trade entry/exit analysis

    Args:
        entry_prices: List of entry prices
        exit_prices: List of exit prices
        entry_dates: Optional list of entry dates
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate PnL for each trade
    pnls = [(exit - entry) / entry * 100 for entry, exit in zip(entry_prices, exit_prices)]

    # Trade PnL scatter
    if entry_dates is not None:
        ax1.scatter(entry_dates, pnls, c=['green' if p > 0 else 'red' for p in pnls],
                    alpha=0.6, s=50)
        ax1.set_xlabel("Date", fontsize=12)
    else:
        ax1.scatter(range(len(pnls)), pnls, c=['green' if p > 0 else 'red' for p in pnls],
                    alpha=0.6, s=50)
        ax1.set_xlabel("Trade Number", fontsize=12)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel("Trade PnL (%)", fontsize=12)
    ax1.set_title("Individual Trade Returns", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Cumulative PnL
    cum_pnl = np.cumsum(pnls)
    ax2.plot(cum_pnl, linewidth=2, color='#2E86AB')
    ax2.fill_between(range(len(cum_pnl)), cum_pnl, 0, alpha=0.3, color='#2E86AB')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel("Trade Number", fontsize=12)
    ax2.set_ylabel("Cumulative PnL (%)", fontsize=12)
    ax2.set_title("Cumulative Trade Returns", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_full_tearsheet(
        equity_curve: np.ndarray,
        returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Strategy Performance Tearsheet",
        figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive tearsheet with multiple plots

    Args:
        equity_curve: Array of portfolio values
        returns: Array of returns
        timestamps: Optional array of timestamps
        title: Overall title
        figsize: Figure size

    Returns:
        Matplotlib figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Equity curve (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    if timestamps is not None:
        if timestamps.dtype == 'int64':
            dates = pd.to_datetime(timestamps, unit='ns')
        else:
            dates = timestamps
        x = dates
    else:
        x = np.arange(len(equity_curve))

    ax1.plot(x, equity_curve, linewidth=2, color='#2E86AB')
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_mask = equity_curve < running_max
    if np.any(drawdown_mask):
        ax1.fill_between(x, equity_curve, running_max, where=drawdown_mask,
                         alpha=0.3, color='red', label='Drawdown')
    ax1.set_title("Equity Curve", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Underwater plot (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    drawdown = (equity_curve - running_max) / running_max * 100.0
    ax2.fill_between(x, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(x, drawdown, linewidth=1.5, color='darkred')
    ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. Returns distribution (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(returns) * 100, color='green', linestyle='--', linewidth=2)
    ax3.set_title("Returns Distribution", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Return (%)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Rolling Sharpe (bottom left)
    from .analytics import rolling_sharpe
    ax4 = fig.add_subplot(gs[2, 0])
    roll_sharpe = rolling_sharpe(returns, window=60)
    if timestamps is not None:
        x_rolling = dates[60:]
    else:
        x_rolling = np.arange(60, len(returns) + 1)
    ax4.plot(x_rolling, roll_sharpe, linewidth=2, color='#2E86AB')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title("Rolling Sharpe (60-day)", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Sharpe Ratio", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Volatility (bottom right)
    from .analytics import rolling_volatility
    ax5 = fig.add_subplot(gs[2, 1])
    roll_vol = rolling_volatility(returns, window=60)
    ax5.plot(x_rolling, roll_vol, linewidth=2, color='#A23B72')
    ax5.set_title("Rolling Volatility (60-day)", fontsize=14, fontweight='bold')
    ax5.set_ylabel("Volatility (%)", fontsize=10)
    ax5.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    return fig


def save_all_plots(
        equity_curve: np.ndarray,
        returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        output_dir: str = "plots",
        strategy_name: str = "strategy"
):
    """
    Generate and save all plots to directory

    Args:
        equity_curve: Array of portfolio values
        returns: Array of returns
        timestamps: Optional array of timestamps
        output_dir: Directory to save plots
        strategy_name: Name for file prefixes
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Equity curve
    fig = plot_equity_curve(equity_curve, timestamps)
    fig.savefig(f"{output_dir}/{strategy_name}_equity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Underwater
    fig = plot_underwater(equity_curve, timestamps)
    fig.savefig(f"{output_dir}/{strategy_name}_underwater.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Returns distribution
    fig = plot_returns_distribution(returns)
    fig.savefig(f"{output_dir}/{strategy_name}_returns_dist.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Rolling metrics
    fig = plot_rolling_metrics(returns, timestamps)
    fig.savefig(f"{output_dir}/{strategy_name}_rolling.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Full tearsheet
    fig = plot_full_tearsheet(equity_curve, returns, timestamps)
    fig.savefig(f"{output_dir}/{strategy_name}_tearsheet.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"All plots saved to {output_dir}/")