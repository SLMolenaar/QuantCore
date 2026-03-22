"""
Performance analytics for backtesting results

Standard metrics for evaluating strategy performance:
- Returns (total, annualized, rolling)
- Risk metrics (Sharpe, Sortino, volatility, drawdown)
- Trade analysis (win rate, profit factor, avg trade)
- Benchmark comparison (alpha, beta, information ratio, capture ratios)
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


@dataclass
class BenchmarkMetrics:
    """
    Benchmark-relative performance metrics.

    All return/alpha values are in percentage format (10.5 = 10.5%),
    consistent with PerformanceMetrics. Ratios (beta, correlation, R²,
    capture ratios, information ratio) are dimensionless.

    Fields
    ------
    benchmark_total_return      : total return of the benchmark over the period (%)
    benchmark_annualized_return : annualized benchmark return (%)
    active_return               : annualized strategy return minus annualized
                                  benchmark return (%)
    alpha                       : CAPM alpha — annualized excess return after
                                  removing the portion explained by beta (%)
    beta                        : slope of strategy returns regressed on benchmark
                                  returns; 1.0 means the strategy moves in lock-step
                                  with the benchmark
    correlation                 : Pearson correlation between strategy and benchmark
                                  period returns
    r_squared                   : fraction of strategy return variance explained by
                                  the benchmark (correlation²)
    tracking_error              : annualized standard deviation of active returns
                                  (strategy minus benchmark) (%)
    information_ratio           : active_return / tracking_error; measures
                                  risk-adjusted outperformance consistency
    up_capture                  : ratio of strategy return to benchmark return
                                  in periods where the benchmark rose (>100 = beats
                                  benchmark on up moves)
    down_capture                : ratio of strategy return to benchmark return
                                  in periods where the benchmark fell (<100 = loses
                                  less than the benchmark on down moves)
    """
    benchmark_total_return:      float
    benchmark_annualized_return: float
    active_return:               float
    alpha:                       float
    beta:                        float
    correlation:                 float
    r_squared:                   float
    tracking_error:              float
    information_ratio:           float
    up_capture:                  float
    down_capture:                float

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Benchmark Comparison",
            "=" * 60,
            "",
            f"  Benchmark Total Return:    {self.benchmark_total_return:>10.2f}%",
            f"  Benchmark Annual Return:   {self.benchmark_annualized_return:>10.2f}%",
            f"  Active Return (ann.):      {self.active_return:>10.2f}%",
            "",
            f"  Alpha (ann.):              {self.alpha:>10.2f}%",
            f"  Beta:                      {self.beta:>10.2f}",
            f"  Correlation:               {self.correlation:>10.2f}",
            f"  R-Squared:                 {self.r_squared:>10.2f}",
            "",
            f"  Tracking Error (ann.):     {self.tracking_error:>10.2f}%",
            f"  Information Ratio:         {self.information_ratio:>10.2f}",
            "",
            f"  Up Capture:                {self.up_capture:>10.2f}%",
            f"  Down Capture:              {self.down_capture:>10.2f}%",
            "",
            "=" * 60,
            ]
        return "\n".join(lines)


def calculate_benchmark_metrics(
        strategy_returns:   np.ndarray,
        benchmark_returns:  np.ndarray,
        periods_per_year:   Optional[int] = None,
        timestamps:         Optional[np.ndarray] = None,
) -> BenchmarkMetrics:
    """
    Calculate benchmark-relative performance metrics.

    Both arrays must be period returns (output of calculate_returns), not
    equity curves. They must have the same length — align them on a common
    timestamp index before calling this function if the two backtests
    produced different-length equity curves.

    Args:
        strategy_returns:  Period returns for the strategy.
        benchmark_returns: Period returns for the benchmark.
        periods_per_year:  Snapshots per year for annualisation. When None
                           (default), inferred from timestamps if provided,
                           otherwise falls back to 252. Pass an explicit value
                           to override (e.g. 252 for daily, 525_960 for
                           1-minute crypto bars).
        timestamps:        Nanosecond UNIX timestamps aligned with the
                           equity curve (one element longer than the returns
                           arrays). Used to auto-infer periods_per_year when
                           that argument is None.

    Returns:
        BenchmarkMetrics dataclass.
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(timestamps) \
            if timestamps is not None else 252
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return BenchmarkMetrics(
            benchmark_total_return=0.0,
            benchmark_annualized_return=0.0,
            active_return=0.0,
            alpha=0.0,
            beta=0.0,
            correlation=0.0,
            r_squared=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            up_capture=0.0,
            down_capture=0.0,
        )

    # Align lengths — use the shorter of the two to avoid index errors.
    n = min(len(strategy_returns), len(benchmark_returns))
    s = strategy_returns[:n]
    b = benchmark_returns[:n]

    # ---- benchmark return summary ----
    bm_growth = float(np.prod(1.0 + b))
    bm_total_return = (bm_growth - 1.0) * 100.0
    years = n / periods_per_year
    if years > 0 and bm_growth > 0:
        bm_ann_return = (bm_growth ** (1.0 / years) - 1.0) * 100.0
    else:
        bm_ann_return = 0.0

    # ---- active return (annualised) ----
    strat_growth = float(np.prod(1.0 + s))
    if years > 0 and strat_growth > 0:
        strat_ann_return = (strat_growth ** (1.0 / years) - 1.0) * 100.0
    else:
        strat_ann_return = 0.0
    active_return = strat_ann_return - bm_ann_return

    # ---- beta and CAPM alpha ----
    # OLS regression: s = alpha_period + beta * b + epsilon
    # beta  = Cov(s, b) / Var(b)
    # alpha_period = mean(s) - beta * mean(b)
    # Annualise alpha: alpha_ann = alpha_period * periods_per_year
    bm_var = float(np.var(b, ddof=1))
    if bm_var > 1e-12:
        cov = float(np.cov(s, b)[0, 1])
        beta = cov / bm_var
    else:
        beta = 0.0

    alpha_period = float(np.mean(s)) - beta * float(np.mean(b))
    alpha_ann    = alpha_period * periods_per_year * 100.0

    # ---- correlation and R² ----
    s_std = float(np.std(s, ddof=1))
    b_std = float(np.std(b, ddof=1))
    if s_std > 1e-12 and b_std > 1e-12:
        correlation = float(np.corrcoef(s, b)[0, 1])
    else:
        correlation = 0.0
    r_squared = correlation ** 2

    # ---- tracking error (annualised) ----
    active_returns = s - b
    tracking_error = float(np.std(active_returns, ddof=1)) * np.sqrt(periods_per_year) * 100.0

    # ---- information ratio ----
    # IR = mean(active_returns) / std(active_returns) — not annualised separately
    # because both numerator and denominator scale the same way with period length.
    if tracking_error > 1e-10:
        # tracking_error is already annualised (%), so annualise the numerator too.
        information_ratio = (active_return / tracking_error)
    else:
        if active_return > 0:
            information_ratio = MAX_RATIO_VALUE
        elif active_return < 0:
            information_ratio = -MAX_RATIO_VALUE
        else:
            information_ratio = 0.0

    # ---- up/down capture ratios ----
    up_mask   = b > 0
    down_mask = b < 0

    if np.any(up_mask):
        # geometric compounding of strategy and benchmark returns in up periods
        up_strat_return = float(np.prod(1.0 + s[up_mask]) - 1.0)
        up_bm_return    = float(np.prod(1.0 + b[up_mask]) - 1.0)
        up_capture = (up_strat_return / up_bm_return * 100.0) if abs(up_bm_return) > 1e-12 else 0.0
    else:
        up_capture = 0.0

    if np.any(down_mask):
        down_strat_return = float(np.prod(1.0 + s[down_mask]) - 1.0)
        down_bm_return    = float(np.prod(1.0 + b[down_mask]) - 1.0)
        down_capture = (down_strat_return / down_bm_return * 100.0) if abs(down_bm_return) > 1e-12 else 0.0
    else:
        down_capture = 0.0

    return BenchmarkMetrics(
        benchmark_total_return=bm_total_return,
        benchmark_annualized_return=bm_ann_return,
        active_return=active_return,
        alpha=alpha_ann,
        beta=float(np.clip(beta, -MAX_RATIO_VALUE, MAX_RATIO_VALUE)),
        correlation=float(np.clip(correlation, -1.0, 1.0)),
        r_squared=float(np.clip(r_squared, 0.0, 1.0)),
        tracking_error=tracking_error,
        information_ratio=float(np.clip(information_ratio, -MAX_RATIO_VALUE, MAX_RATIO_VALUE)),
        up_capture=up_capture,
        down_capture=down_capture,
    )


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
                      the duration is returned in calendar days. When omitted,
                      the duration is returned as a snapshot count
                      (recovery_idx - peak_idx), which equals the number of
                      bars for daily data but is misleading for tick/minute data
                      where most snapshots show no change.

    Returns:
        (max_drawdown_pct, duration_in_calendar_days_or_snapshot_count)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100.0

    max_dd = float(np.min(drawdown))
    trough_idx = int(np.argmin(drawdown))

    # peak is the last all-time high before the trough
    peak_idx = int(np.argmax(equity_curve[: trough_idx + 1]))

    # look for full recovery after the trough.
    # defaults to trough_idx (no recovery) so that snapshot-count duration
    # equals trough, peak when timestamps are not provided.
    recovery_idx = trough_idx
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
        # No timestamps: fall back to snapshot count (recovery_idx - peak_idx).
        duration = recovery_idx - peak_idx

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