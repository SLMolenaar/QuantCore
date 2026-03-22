"""
Visualization tools for backtesting results

Creates plots for equity curves, returns, drawdowns, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Optional, List, Tuple

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def _infer_rolling_window(
        returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        target_calendar_days: int = 14,
        fallback_window: int = 60,
) -> int:
    """
    Choose a rolling window that spans roughly `target_calendar_days` of
    real time.

    `timestamps` is the equity-length array (one element longer than
    `returns`). The median inter-snapshot interval is measured on the
    returns-aligned slice `timestamps[1:]`. The result is clamped to
    [10, len(returns) // 4].
    """
    n = len(returns)
    if n < 20:
        return max(2, n // 2)

    if timestamps is not None and len(timestamps) >= 3:
        ts   = timestamps[1:].astype(np.int64)
        gaps = np.diff(ts)
        positive_gaps = gaps[gaps > 0]
        if len(positive_gaps) > 0:
            median_gap_ns   = float(np.median(positive_gaps))
            ns_per_day      = 86_400 * 1_000_000_000
            periods_per_day = ns_per_day / median_gap_ns
            window          = int(round(periods_per_day * target_calendar_days))
            return int(np.clip(window, 10, n // 4))

    return int(np.clip(fallback_window, 10, n // 4))


def _to_dates(timestamps: np.ndarray) -> pd.DatetimeIndex:
    """Convert a nanosecond int64 timestamp array to a DatetimeIndex."""
    return pd.to_datetime(timestamps.astype(np.int64), unit='ns')


def _configure_date_axis(ax, x: pd.DatetimeIndex) -> None:
    """
    Apply date formatting and set xlim explicitly.

    AutoDateLocator can default to epoch 0 when it fires before the axis
    data range is known. Setting xlim immediately after plot() prevents this.
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Explicit xlim so the locator always has the correct range to work with.
    ax.set_xlim(x[0], x[-1])
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


def _equity_x(timestamps: Optional[np.ndarray], equity_len: int) -> np.ndarray:
    """Return x values for an equity-length chart (plain index if no timestamps)."""
    if timestamps is not None:
        return _to_dates(timestamps)
    return np.arange(equity_len)


def _rolling_x(
        timestamps: Optional[np.ndarray],
        returns_len: int,
        window: int,
) -> np.ndarray:
    """
    Return x values for a rolling metric chart.

    Rolling output has length `returns_len - window + 1`. Each value
    corresponds to the window ending at return index `window - 1 + i`,
    which maps to equity timestamp index `window + i`. We therefore
    slice `timestamps[window : window + n_rolling]`.
    """
    n_rolling = returns_len - window + 1

    if timestamps is not None and len(timestamps) > window:
        ts_slice = timestamps[window : window + n_rolling]
        if len(ts_slice) > 0:
            return _to_dates(ts_slice)

    return np.arange(window, returns_len + 1)


def _normalise_to_100(equity: np.ndarray) -> np.ndarray:
    """Rescale an equity curve so it starts at 100, for overlay comparison."""
    if len(equity) == 0 or equity[0] == 0:
        return equity
    return equity / equity[0] * 100.0


def plot_equity_curve(
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Equity Curve",
        figsize: Tuple[int, int] = (14, 7),
        show_drawdown: bool = True,
        benchmark_equity: Optional[np.ndarray] = None,
        benchmark_label: str = "Benchmark",
) -> plt.Figure:
    """
    Plot equity curve with optional drawdown shading and benchmark overlay.

    Args:
        equity_curve:     Strategy equity curve.
        timestamps:       Nanosecond timestamps aligned with equity_curve.
        title:            Chart title.
        figsize:          Figure dimensions.
        show_drawdown:    Shade drawdown periods in red.
        benchmark_equity: Optional benchmark equity curve of the same length.
                          Both curves are normalised to 100 at the start so they
                          can be compared regardless of initial capital differences.
        benchmark_label:  Legend label for the benchmark line.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = _equity_x(timestamps, len(equity_curve))

    if benchmark_equity is not None:
        # Normalise both to 100 so different initial capitals don't distort the overlay.
        strat_plot = _normalise_to_100(equity_curve)
        bm_plot    = _normalise_to_100(
            benchmark_equity[:len(equity_curve)]
        )
        ax.plot(x, strat_plot, linewidth=2, color='#2E86AB', label='Strategy')
        ax.plot(x, bm_plot,    linewidth=1.5, color='#E84855',
                linestyle='--', alpha=0.8, label=benchmark_label)
        ax.set_ylabel("Normalised Value (100 = start)", fontsize=12)
        if show_drawdown:
            running_max   = np.maximum.accumulate(strat_plot)
            drawdown_mask = strat_plot < running_max
            if np.any(drawdown_mask):
                ax.fill_between(x, strat_plot, running_max,
                                where=drawdown_mask,
                                alpha=0.2, color='red', label='Strategy Drawdown')
    else:
        ax.plot(x, equity_curve, linewidth=2, color='#2E86AB', label='Portfolio Value')
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
        if show_drawdown:
            running_max   = np.maximum.accumulate(equity_curve)
            drawdown_mask = equity_curve < running_max
            if np.any(drawdown_mask):
                ax.fill_between(x, equity_curve, running_max,
                                where=drawdown_mask,
                                alpha=0.3, color='red', label='Drawdown')

    if timestamps is not None:
        _configure_date_axis(ax, x)
    else:
        ax.set_xlabel("Time Period")

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_benchmark_comparison(
        equity_curve: np.ndarray,
        benchmark_equity: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        benchmark_label: str = "Benchmark",
        strategy_label: str = "Strategy",
        title: str = "Strategy vs Benchmark",
        figsize: Tuple[int, int] = (14, 10),
        periods_per_year: Optional[int] = None,
) -> plt.Figure:
    """
    Three-panel benchmark comparison chart.

    Top panel: normalised equity curves overlaid (both start at 100).
    Middle panel: rolling active return (strategy minus benchmark, annualised).
    Bottom panel: underwater drawdown for strategy and benchmark side by side.

    Args:
        equity_curve:     Strategy equity curve.
        benchmark_equity: Benchmark equity curve (will be trimmed to match length).
        timestamps:       Nanosecond timestamps aligned with equity_curve.
        benchmark_label:  Legend label for benchmark.
        strategy_label:   Legend label for strategy.
        title:            Figure title.
        figsize:          Figure dimensions.
        periods_per_year: Annualisation factor. When None, inferred from timestamps
                          or falls back to 252.
    """
    from .analytics import (
        calculate_returns, infer_periods_per_year, rolling_volatility,
    )

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(timestamps) \
            if timestamps is not None else 252

    n = min(len(equity_curve), len(benchmark_equity))
    eq   = equity_curve[:n]
    bm   = benchmark_equity[:n]
    ts   = timestamps[:n] if timestamps is not None else None

    strat_norm = _normalise_to_100(eq)
    bm_norm    = _normalise_to_100(bm)
    x          = _equity_x(ts, n)

    strat_returns = calculate_returns(eq)
    bm_returns    = calculate_returns(bm)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize,
                                        gridspec_kw={'height_ratios': [3, 2, 2]})

    # ---- top: normalised equity curves ----
    ax1.plot(x, strat_norm, linewidth=2,   color='#2E86AB', label=strategy_label)
    ax1.plot(x, bm_norm,    linewidth=1.5, color='#E84855',
             linestyle='--', alpha=0.8, label=benchmark_label)
    ax1.set_title("Normalised Performance (100 = start)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Value", fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    if ts is not None and isinstance(x, pd.DatetimeIndex):
        _configure_date_axis(ax1, x)

    # ---- middle: rolling active return (annualised %) ----
    # Compute with a window sized to ~63 calendar days (one quarter).
    # Rolling active return = cumulative product of (1 + s_i) / (1 + b_i) - 1
    # over the window, then annualise.
    roll_window = max(10, int(round(periods_per_year / 4)))
    roll_window = min(roll_window, len(strat_returns) // 4) if len(strat_returns) > 8 else 2
    active_per_period = strat_returns - bm_returns

    if len(active_per_period) >= roll_window:
        rolling_active = np.empty(len(active_per_period) - roll_window + 1)
        for i in range(len(rolling_active)):
            window_active = active_per_period[i : i + roll_window]
            cumulative    = float(np.prod(1.0 + window_active) - 1.0)
            rolling_active[i] = cumulative * (periods_per_year / roll_window) * 100.0

        x_roll = _rolling_x(ts, len(strat_returns), roll_window)
        ax2.plot(x_roll, rolling_active, linewidth=1.5, color='#2E86AB')
        ax2.fill_between(x_roll, rolling_active, 0,
                         where=rolling_active >= 0, alpha=0.2, color='green',
                         label='Outperforming')
        ax2.fill_between(x_roll, rolling_active, 0,
                         where=rolling_active < 0, alpha=0.2, color='red',
                         label='Underperforming')
        ax2.axhline(y=0, color='black', linewidth=0.8)
        if ts is not None and isinstance(x_roll, pd.DatetimeIndex) and len(x_roll) > 0:
            _configure_date_axis(ax2, x_roll)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for rolling window",
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.set_title(f"Rolling Active Return (ann., {roll_window}-period window)",
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel("Active Return (%)", fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- bottom: underwater plots for both ----
    strat_running_max = np.maximum.accumulate(strat_norm)
    strat_dd          = (strat_norm - strat_running_max) / strat_running_max * 100.0

    bm_running_max = np.maximum.accumulate(bm_norm)
    bm_dd          = (bm_norm - bm_running_max) / bm_running_max * 100.0

    ax3.fill_between(x, strat_dd, 0, alpha=0.4, color='#2E86AB', label=strategy_label)
    ax3.fill_between(x, bm_dd,    0, alpha=0.3, color='#E84855', label=benchmark_label)
    ax3.plot(x, strat_dd, linewidth=1,   color='#2E86AB', alpha=0.7)
    ax3.plot(x, bm_dd,    linewidth=1,   color='#E84855', linestyle='--', alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title("Drawdown Comparison", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if ts is not None and isinstance(x, pd.DatetimeIndex):
        _configure_date_axis(ax3, x)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_underwater(
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Underwater Plot",
        figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Shows drawdown over time."""
    fig, ax = plt.subplots(figsize=figsize)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown    = (equity_curve - running_max) / running_max * 100.0

    x = _equity_x(timestamps, len(equity_curve))
    ax.fill_between(x, drawdown, 0, alpha=0.3, color='red')
    ax.plot(x, drawdown, linewidth=1.5, color='darkred')

    if timestamps is not None:
        _configure_date_axis(ax, x)

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
    Histogram of returns with Q-Q plot.

    Zero-return periods (flat equity, no open position) are excluded from
    both the histogram and Q-Q plot. These dominate the distribution in
    tick/minute backtests and make the histogram useless if included.
    An annotation reports the excluded fraction when it exceeds 5%.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    nonzero_mask   = returns != 0.0
    active_returns = returns[nonzero_mask]
    zero_fraction  = 1.0 - nonzero_mask.mean()

    if len(active_returns) == 0:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No non-zero returns",
                    ha='center', va='center', transform=ax.transAxes)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    display_returns = active_returns * 100
    n_bins = min(50, max(10, len(active_returns) // 20))

    ax1.hist(display_returns, bins=n_bins,
             alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(x=0, color='red',   linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(x=float(np.mean(display_returns)), color='green', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(display_returns):.3f}%')

    ax1.set_title("Returns Histogram (non-zero periods)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Return (%)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if zero_fraction > 0.05:
        ax1.annotate(
            f"{zero_fraction * 100:.0f}% of periods excluded\n(flat equity — no open position)",
            xy=(0.97, 0.97), xycoords='axes fraction',
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
        )

    from scipy import stats
    stats.probplot(active_returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (non-zero periods)", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_rolling_metrics(
        returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window: int = 0,
        title: str = "Rolling Performance Metrics",
        figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Rolling Sharpe and volatility over time.

    When `window` is 0 (default) it is inferred from `timestamps` to span
    ~14 calendar days. NaN values (flat-equity windows) render as gaps.
    """
    from .analytics import rolling_sharpe, rolling_volatility, infer_periods_per_year

    effective_window = window if window > 0 else _infer_rolling_window(
        returns, timestamps, target_calendar_days=14
    )
    ppy = infer_periods_per_year(timestamps) if timestamps is not None else 252

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    roll_sharpe_vals = rolling_sharpe(returns, effective_window, ppy)
    roll_vol_vals    = rolling_volatility(returns, effective_window, ppy)
    x                = _rolling_x(timestamps, len(returns), effective_window)

    ax1.plot(x, roll_sharpe_vals, linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=0, color='red',   linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
    ax1.set_title(f"Rolling Sharpe ({effective_window}-period window)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Sharpe Ratio", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, roll_vol_vals, linewidth=1.5, color='#A23B72')
    ax2.set_title(f"Rolling Volatility ({effective_window}-period window)",
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel("Volatility (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # configure date axis on ax2 (sharex means ax1 inherits)
    if timestamps is not None and isinstance(x, pd.DatetimeIndex) and len(x) > 0:
        _configure_date_axis(ax2, x)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
        monthly_returns: pd.DataFrame,
        title: str = "Monthly Returns (%)",
        figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Heatmap showing monthly returns by year."""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(monthly_returns.values, cmap='RdYlGn', aspect='auto',
                   vmin=-10, vmax=10, interpolation='nearest')

    ax.set_xticks(np.arange(len(monthly_returns.columns)))
    ax.set_yticks(np.arange(len(monthly_returns.index)))
    ax.set_xticklabels(monthly_returns.columns)
    ax.set_yticklabels(monthly_returns.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)', rotation=270, labelpad=20, fontsize=12)

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
    """Visualize individual trade PnL and cumulative performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    pnls   = [(e - n) / n * 100 for n, e in zip(entry_prices, exit_prices)]
    colors = ['green' if p > 0 else 'red' for p in pnls]

    if entry_dates is not None:
        ax1.scatter(entry_dates, pnls, c=colors, alpha=0.6, s=50)
        ax1.set_xlabel("Date", fontsize=12)
    else:
        ax1.scatter(range(len(pnls)), pnls, c=colors, alpha=0.6, s=50)
        ax1.set_xlabel("Trade Number", fontsize=12)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel("Trade PnL (%)", fontsize=12)
    ax1.set_title("Individual Trade Returns", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

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
        figsize: Tuple[int, int] = (16, 12),
        benchmark_equity: Optional[np.ndarray] = None,
        benchmark_label: str = "Benchmark",
) -> plt.Figure:
    """
    Combined view of all key metrics in one figure.

    When `benchmark_equity` is supplied the equity curve panel shows both
    curves normalised to 100, and a benchmark metrics table is appended
    below the tearsheet.

    Args:
        equity_curve:     Strategy equity curve.
        returns:          Strategy period returns (output of calculate_returns).
        timestamps:       Nanosecond timestamps aligned with equity_curve.
        title:            Figure title.
        figsize:          Figure dimensions.
        benchmark_equity: Optional benchmark equity curve of the same length.
        benchmark_label:  Legend label for the benchmark line.
    """
    from .analytics import rolling_sharpe, rolling_volatility, infer_periods_per_year

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    roll_window = _infer_rolling_window(returns, timestamps, target_calendar_days=14)
    ppy         = infer_periods_per_year(timestamps) if timestamps is not None else 252

    x_eq      = _equity_x(timestamps, len(equity_curve))
    x_rolling = _rolling_x(timestamps, len(returns), roll_window)
    use_dates = timestamps is not None and isinstance(x_rolling, pd.DatetimeIndex) \
                and len(x_rolling) > 0

    # ---- equity curve (full width) ----
    ax1 = fig.add_subplot(gs[0, :])
    if benchmark_equity is not None:
        bm_n         = min(len(benchmark_equity), len(equity_curve))
        strat_norm   = _normalise_to_100(equity_curve)
        bm_norm      = _normalise_to_100(benchmark_equity[:bm_n])
        x_bm         = _equity_x(timestamps[:bm_n] if timestamps is not None else None, bm_n)

        ax1.plot(x_eq[:bm_n], strat_norm[:bm_n], linewidth=2,
                 color='#2E86AB', label='Strategy')
        ax1.plot(x_bm, bm_norm, linewidth=1.5, color='#E84855',
                 linestyle='--', alpha=0.8, label=benchmark_label)
        ax1.set_ylabel("Normalised Value (100 = start)", fontsize=10)

        running_max   = np.maximum.accumulate(strat_norm[:bm_n])
        drawdown_mask = strat_norm[:bm_n] < running_max
        if np.any(drawdown_mask):
            ax1.fill_between(x_eq[:bm_n], strat_norm[:bm_n], running_max,
                             where=drawdown_mask, alpha=0.2, color='red',
                             label='Strategy Drawdown')
    else:
        ax1.plot(x_eq, equity_curve, linewidth=2, color='#2E86AB')
        running_max   = np.maximum.accumulate(equity_curve)
        drawdown_mask = equity_curve < running_max
        if np.any(drawdown_mask):
            ax1.fill_between(x_eq, equity_curve, running_max,
                             where=drawdown_mask, alpha=0.3, color='red', label='Drawdown')
        ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    ax1.set_title("Equity Curve", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if timestamps is not None and isinstance(x_eq, pd.DatetimeIndex):
        _configure_date_axis(ax1, x_eq)

    # ---- drawdown ----
    ax2 = fig.add_subplot(gs[1, 0])
    running_max_full = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max_full) / running_max_full * 100.0
    ax2.fill_between(x_eq, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(x_eq, drawdown, linewidth=1.5, color='darkred')
    ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    if timestamps is not None and isinstance(x_eq, pd.DatetimeIndex):
        _configure_date_axis(ax2, x_eq)

    # ---- returns distribution (non-zero only) ----
    ax3 = fig.add_subplot(gs[1, 1])
    nonzero_mask   = returns != 0.0
    active_returns = returns[nonzero_mask]
    zero_fraction  = 1.0 - nonzero_mask.mean()

    if len(active_returns) > 0:
        display_r = active_returns * 100
        n_bins    = min(50, max(10, len(active_returns) // 20))
        ax3.hist(display_r, bins=n_bins, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax3.axvline(x=0, color='red',   linestyle='--', linewidth=2)
        ax3.axvline(x=float(np.mean(display_r)), color='green', linestyle='--', linewidth=2)
        if zero_fraction > 0.05:
            ax3.annotate(
                f"{zero_fraction * 100:.0f}% flat periods\nexcluded",
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', alpha=0.8),
            )
    else:
        ax3.text(0.5, 0.5, "No non-zero returns",
                 ha='center', va='center', transform=ax3.transAxes)

    ax3.set_title("Returns Distribution (active periods)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Return (%)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ---- rolling Sharpe ----
    ax4 = fig.add_subplot(gs[2, 0])
    roll_sharpe_vals = rolling_sharpe(returns, roll_window, ppy)
    ax4.plot(x_rolling, roll_sharpe_vals, linewidth=1.5, color='#2E86AB')
    ax4.axhline(y=0, color='red',   linestyle='--', linewidth=1, alpha=0.5)
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title(f"Rolling Sharpe ({roll_window}-period)", fontsize=13, fontweight='bold')
    ax4.set_ylabel("Sharpe Ratio", fontsize=10)
    ax4.grid(True, alpha=0.3)
    if use_dates:
        _configure_date_axis(ax4, x_rolling)

    # ---- rolling volatility ----
    ax5 = fig.add_subplot(gs[2, 1])
    roll_vol_vals = rolling_volatility(returns, roll_window, ppy)
    ax5.plot(x_rolling, roll_vol_vals, linewidth=1.5, color='#A23B72')
    ax5.set_title(f"Rolling Volatility ({roll_window}-period)", fontsize=13, fontweight='bold')
    ax5.set_ylabel("Volatility (%)", fontsize=10)
    ax5.grid(True, alpha=0.3)
    if use_dates:
        _configure_date_axis(ax5, x_rolling)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    return fig


def save_all_plots(
        equity_curve: np.ndarray,
        returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        output_dir: str = "plots",
        strategy_name: str = "strategy",
        benchmark_equity: Optional[np.ndarray] = None,
        benchmark_label: str = "Benchmark",
):
    """
    Save all plots to output directory.

    When `benchmark_equity` is provided an additional benchmark comparison
    chart is saved as `{strategy_name}_benchmark.png`.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_jobs = [
        (lambda: plot_equity_curve(equity_curve, timestamps,
                                   benchmark_equity=benchmark_equity,
                                   benchmark_label=benchmark_label),  "_equity"),
        (lambda: plot_underwater(equity_curve, timestamps),            "_underwater"),
        (lambda: plot_returns_distribution(returns),                   "_returns_dist"),
        (lambda: plot_rolling_metrics(returns, timestamps),            "_rolling"),
        (lambda: plot_full_tearsheet(equity_curve, returns, timestamps,
                                     benchmark_equity=benchmark_equity,
                                     benchmark_label=benchmark_label), "_tearsheet"),
    ]

    if benchmark_equity is not None:
        plot_jobs.append(
            (lambda: plot_benchmark_comparison(equity_curve, benchmark_equity,
                                               timestamps,
                                               benchmark_label=benchmark_label),
             "_benchmark")
        )

    for plot_fn, suffix in plot_jobs:
        fig = plot_fn()
        fig.savefig(f"{output_dir}/{strategy_name}{suffix}.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"All plots saved to {output_dir}/")