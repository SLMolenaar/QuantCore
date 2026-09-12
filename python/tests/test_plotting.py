"""
Tests for the plotting module.

Assertions inspect the actual data plotted onto the matplotlib Axes
(line data, histogram bar heights, image arrays, drawdown values)
rather than just checking that a Figure object comes back.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantcore.analytics import (
    calculate_returns, rolling_sharpe, rolling_volatility, infer_periods_per_year,
)
from quantcore.plotting import (
    _infer_rolling_window,
    _normalise_to_100,
    _to_dates,
    _equity_x,
    _rolling_x,
    plot_equity_curve,
    plot_underwater,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns_heatmap,
    plot_trade_analysis,
    plot_full_tearsheet,
    save_all_plots,
)

SEC = 1_000_000_000
DAY = 86_400 * SEC


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so matplotlib doesn't warn about leaks."""
    yield
    plt.close("all")


def _daily_timestamps(n: int, start: int = 1_700_000_000 * SEC) -> np.ndarray:
    return np.array([start + i * DAY for i in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

class TestNormaliseTo100:
    def test_rescales_to_start_at_100(self):
        result = _normalise_to_100(np.array([50.0, 100.0, 150.0]))
        np.testing.assert_allclose(result, [100.0, 200.0, 300.0])

    def test_zero_first_value_returned_unchanged(self):
        equity = np.array([0.0, 10.0, 20.0])
        result = _normalise_to_100(equity)
        np.testing.assert_array_equal(result, equity)

    def test_empty_array_returned_unchanged(self):
        equity = np.array([])
        result = _normalise_to_100(equity)
        assert len(result) == 0


class TestToDates:
    def test_converts_ns_to_datetimeindex(self):
        ts = np.array([1_700_000_000 * SEC, 1_700_086_400 * SEC], dtype=np.int64)
        result = _to_dates(ts)
        expected = pd.to_datetime(ts, unit="ns")
        assert isinstance(result, pd.DatetimeIndex)
        pd.testing.assert_index_equal(result, expected)


class TestEquityX:
    def test_no_timestamps_returns_plain_index(self):
        result = _equity_x(None, 5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_with_timestamps_returns_dates(self):
        ts = _daily_timestamps(3)
        result = _equity_x(ts, 3)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3


class TestRollingX:
    def test_no_timestamps_returns_index_range(self):
        result = _rolling_x(None, returns_len=10, window=4)
        np.testing.assert_array_equal(result, np.arange(4, 11))

    def test_with_timestamps_slices_correctly(self):
        # equity has returns_len + 1 timestamps
        ts = _daily_timestamps(11)
        result = _rolling_x(ts, returns_len=10, window=4)
        n_rolling = 10 - 4 + 1
        assert len(result) == n_rolling
        expected = _to_dates(ts[4:4 + n_rolling])
        pd.testing.assert_index_equal(result, expected)


class TestInferRollingWindow:
    def test_short_series_uses_half_length(self):
        returns = np.zeros(10)
        assert _infer_rolling_window(returns, None) == max(2, 10 // 2)

    def test_falls_back_to_default_without_timestamps(self):
        returns = np.zeros(400)
        window = _infer_rolling_window(returns, None, fallback_window=60)
        assert window == 60

    def test_clamped_to_upper_bound(self):
        returns = np.zeros(40)  # n // 4 == 10
        window = _infer_rolling_window(returns, None, fallback_window=60)
        assert window == 10

    def test_daily_timestamps_target_14_days(self):
        n = 200
        returns = np.zeros(n)
        ts = _daily_timestamps(n + 1)  # equity-length: one longer than returns
        window = _infer_rolling_window(returns, ts, target_calendar_days=14)
        assert window == 14


# ---------------------------------------------------------------------------
# plot_equity_curve
# ---------------------------------------------------------------------------

class TestPlotEquityCurve:
    def test_plots_raw_equity_when_no_timestamps(self):
        equity = np.array([100.0, 105.0, 95.0, 110.0])
        fig = plot_equity_curve(equity, show_drawdown=False)
        ax = fig.axes[0]

        np.testing.assert_array_equal(ax.lines[0].get_ydata(), equity)
        assert ax.get_xlabel() == "Time Period"
        assert len(ax.collections) == 0  # drawdown disabled -> no fill_between

    def test_drawdown_shaded_only_when_present(self):
        # Strictly increasing equity: no drawdown, no shading even though requested.
        equity = np.array([100.0, 101.0, 102.0, 103.0])
        fig = plot_equity_curve(equity, show_drawdown=True)
        assert len(fig.axes[0].collections) == 0

        # Equity that dips below its running max: shading collection is added.
        equity_with_dip = np.array([100.0, 120.0, 90.0, 130.0])
        fig2 = plot_equity_curve(equity_with_dip, show_drawdown=True)
        assert len(fig2.axes[0].collections) == 1

    def test_benchmark_normalised_to_100(self):
        equity     = np.array([200.0, 220.0, 240.0])
        benchmark  = np.array([50.0, 55.0, 60.0])
        fig = plot_equity_curve(equity, benchmark_equity=benchmark, show_drawdown=False)
        ax = fig.axes[0]

        assert len(ax.lines) == 2
        np.testing.assert_allclose(ax.lines[0].get_ydata(), equity / 200.0 * 100.0)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), benchmark / 50.0 * 100.0)
        assert ax.get_ylabel() == "Normalised Value (100 = start)"

    def test_timestamps_produce_date_axis_with_correct_xlim(self):
        equity = np.array([100.0, 101.0, 102.0])
        ts     = _daily_timestamps(3)
        fig = plot_equity_curve(equity, timestamps=ts)
        ax = fig.axes[0]

        assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter)
        expected_x = _to_dates(ts)
        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(mdates.date2num(expected_x[0]))
        assert xlim[1] == pytest.approx(mdates.date2num(expected_x[-1]))


# ---------------------------------------------------------------------------
# plot_underwater
# ---------------------------------------------------------------------------

class TestPlotUnderwater:
    def test_drawdown_values_match_manual_calculation(self):
        equity = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
        running_max = np.maximum.accumulate(equity)
        expected = (equity - running_max) / running_max * 100.0

        fig = plot_underwater(equity)
        ax = fig.axes[0]

        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        assert ax.get_ylabel() == "Drawdown (%)"


# ---------------------------------------------------------------------------
# plot_returns_distribution
# ---------------------------------------------------------------------------

class TestPlotReturnsDistribution:
    def test_histogram_excludes_zero_periods(self):
        returns = np.array([0.01, -0.02, 0.0, 0.03, 0.0, 0.0, -0.01, 0.02])
        active  = returns[returns != 0.0]

        fig = plot_returns_distribution(returns)
        ax_hist = fig.axes[0]

        total_count = sum(patch.get_height() for patch in ax_hist.patches)
        assert total_count == pytest.approx(len(active))

    def test_all_zero_returns_shows_placeholder_text(self):
        returns = np.zeros(10)
        fig = plot_returns_distribution(returns)
        ax_hist, ax_qq = fig.axes[0], fig.axes[1]

        assert len(ax_hist.patches) == 0
        assert any("No non-zero returns" in t.get_text() for t in ax_hist.texts)
        assert any("No non-zero returns" in t.get_text() for t in ax_qq.texts)

    def test_annotation_appears_when_zero_fraction_exceeds_5_percent(self):
        # 20% zero -> annotation expected
        returns = np.array([0.01] * 8 + [0.0] * 2)
        fig = plot_returns_distribution(returns)
        ax_hist = fig.axes[0]
        assert any("excluded" in t.get_text() for t in ax_hist.texts)

    def test_no_annotation_when_zero_fraction_below_5_percent(self):
        # 1 zero out of 100 -> 1%, below the 5% threshold
        returns = np.concatenate([np.full(99, 0.01), np.zeros(1)])
        fig = plot_returns_distribution(returns)
        ax_hist = fig.axes[0]
        assert not any("excluded" in t.get_text() for t in ax_hist.texts)


# ---------------------------------------------------------------------------
# plot_rolling_metrics
# ---------------------------------------------------------------------------

class TestPlotRollingMetrics:
    def test_rolling_lines_match_analytics_functions(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=120)
        window = 20

        fig = plot_rolling_metrics(returns, window=window)
        ax_sharpe, ax_vol = fig.axes[0], fig.axes[1]

        expected_sharpe = rolling_sharpe(returns, window, 252)
        expected_vol    = rolling_volatility(returns, window, 252)

        np.testing.assert_allclose(ax_sharpe.lines[0].get_ydata(), expected_sharpe, equal_nan=True)
        np.testing.assert_allclose(ax_vol.lines[0].get_ydata(), expected_vol, equal_nan=True)

    def test_window_zero_infers_window_from_timestamps(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0, 0.01, size=200)
        ts = _daily_timestamps(201)

        fig = plot_rolling_metrics(returns, timestamps=ts, window=0)
        ax_sharpe = fig.axes[0]

        inferred_window = _infer_rolling_window(returns, ts, target_calendar_days=14)
        ppy = infer_periods_per_year(ts)
        expected_sharpe = rolling_sharpe(returns, inferred_window, ppy)
        np.testing.assert_allclose(ax_sharpe.lines[0].get_ydata(), expected_sharpe, equal_nan=True)


# ---------------------------------------------------------------------------
# plot_monthly_returns_heatmap
# ---------------------------------------------------------------------------

class TestPlotMonthlyReturnsHeatmap:
    def test_image_data_and_labels_match_input(self):
        df = pd.DataFrame(
            [[1.5, -2.0, np.nan], [3.0, 4.0, -1.0]],
            index=[2022, 2023],
            columns=["Jan", "Feb", "Mar"],
        )
        fig = plot_monthly_returns_heatmap(df)
        ax = fig.axes[0]

        im_data = ax.images[0].get_array()
        np.testing.assert_allclose(np.asarray(im_data), df.values, equal_nan=True)

        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        assert xticklabels == list(df.columns.astype(str))
        assert yticklabels == list(df.index.astype(str))


# ---------------------------------------------------------------------------
# plot_trade_analysis
# ---------------------------------------------------------------------------

class TestPlotTradeAnalysis:
    def test_pnl_and_cumulative_pnl_match_manual_calculation(self):
        entry = [100.0, 100.0, 100.0]
        exit_ = [110.0, 90.0, 100.0]
        expected_pnls = [(e - n) / n * 100 for n, e in zip(entry, exit_)]

        fig = plot_trade_analysis(entry, exit_)
        ax_scatter, ax_cum = fig.axes[0], fig.axes[1]

        scatter_y = ax_scatter.collections[0].get_offsets()[:, 1]
        np.testing.assert_allclose(scatter_y, expected_pnls)

        np.testing.assert_allclose(ax_cum.lines[0].get_ydata(), np.cumsum(expected_pnls))


# ---------------------------------------------------------------------------
# plot_full_tearsheet
# ---------------------------------------------------------------------------

class TestPlotFullTearsheet:
    def test_produces_five_panels(self):
        rng = np.random.default_rng(7)
        equity = 10_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=100))
        equity = np.concatenate([[10_000.0], equity])
        returns = calculate_returns(equity)

        fig = plot_full_tearsheet(equity, returns)
        assert len(fig.axes) == 5

    def test_with_benchmark_still_five_panels_and_legend_has_both_labels(self):
        rng = np.random.default_rng(7)
        equity    = 10_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=100))
        equity    = np.concatenate([[10_000.0], equity])
        benchmark = 10_000 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=100))
        benchmark = np.concatenate([[10_000.0], benchmark])
        returns   = calculate_returns(equity)

        fig = plot_full_tearsheet(equity, returns, benchmark_equity=benchmark,
                                   benchmark_label="SPX")
        assert len(fig.axes) == 5

        legend_labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "Strategy" in legend_labels
        assert "SPX" in legend_labels


# ---------------------------------------------------------------------------
# save_all_plots
# ---------------------------------------------------------------------------

class TestSaveAllPlots:
    def test_writes_expected_files(self, tmp_path):
        rng = np.random.default_rng(3)
        equity = 10_000 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=60))
        equity = np.concatenate([[10_000.0], equity])
        returns = calculate_returns(equity)

        out_dir = tmp_path / "plots"
        save_all_plots(equity, returns, output_dir=str(out_dir), strategy_name="strat")

        expected = {
            "strat_equity.png", "strat_underwater.png", "strat_returns_dist.png",
            "strat_rolling.png", "strat_tearsheet.png",
        }
        actual = {p.name for p in out_dir.iterdir()}
        assert actual == expected

    def test_benchmark_adds_extra_file(self, tmp_path):
        rng = np.random.default_rng(3)
        equity    = 10_000 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=60))
        equity    = np.concatenate([[10_000.0], equity])
        benchmark = 10_000 * np.cumprod(1 + rng.normal(0.0002, 0.008, size=60))
        benchmark = np.concatenate([[10_000.0], benchmark])
        returns   = calculate_returns(equity)

        out_dir = tmp_path / "plots"
        save_all_plots(equity, returns, output_dir=str(out_dir),
                        strategy_name="strat", benchmark_equity=benchmark)

        actual = {p.name for p in out_dir.iterdir()}
        assert "strat_benchmark.png" in actual
        assert len(actual) == 6
