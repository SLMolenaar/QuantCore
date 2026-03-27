"""
Tests for walk_forward module.

Covers:
  - _build_windows: timestamp-based slicing, half-open intervals, gap handling
  - WalkForwardAnalyzer: single-asset (regression), multi-asset, edge cases
  - monte_carlo_validation: single-asset, multi-asset synchronized resampling,
    misaligned bar count rejection, both methods
"""

import pytest
import numpy as np
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quantcore as qc
from quantcore.walk_forward import (
    _build_windows,
    _serialize_data,
    _deserialize_data,
    WalkForwardAnalyzer,
    GridSearchOptimizer,
    ParameterGrid,
    OptimizationResult,
    WalkForwardResult,
    pct_to_decimal,
    decimal_to_pct,
    monte_carlo_validation,
)


# ============================================================================
# HELPERS
# ============================================================================

SEC = 1_000_000_000  # 1 second in nanoseconds
DAY = 86_400 * SEC   # 1 day in nanoseconds


def make_bars(n: int, symbol: str = "A", start_price: float = 100.0,
              base_ts: int = 0, step_ts: int = DAY) -> list:
    """Create n bars with a gentle upward drift so strategies can generate signals."""
    bars = []
    for i in range(n):
        price = start_price + i * 0.1
        bars.append(qc.BarData(
            symbol,
            base_ts + i * step_ts,
            price, price + 1.0, price - 1.0, price,
            1_000_000.0,
            ))
    return bars


def make_multi(n: int, symbols=("A", "B"), base_ts: int = 0) -> dict:
    """Create aligned bar series for multiple symbols with the same timestamps."""
    return {
        sym: make_bars(n, symbol=sym, start_price=100.0 + i * 10.0, base_ts=base_ts)
        for i, sym in enumerate(symbols)
    }


# Picklable factory used by WalkForwardAnalyzer / GridSearchOptimizer in tests.
class _BuyAndHoldFactory:
    def __call__(self, **_):
        return qc.BuyAndHold()

    def __reduce__(self):
        return (_BuyAndHoldFactory, ())


# ============================================================================
# FORMAT CONVERTERS
# ============================================================================

class TestFormatConverters:
    def test_pct_to_decimal(self):
        assert pct_to_decimal(10.5)  == pytest.approx(0.105)
        assert pct_to_decimal(0.0)   == pytest.approx(0.0)
        assert pct_to_decimal(-5.0)  == pytest.approx(-0.05)

    def test_decimal_to_pct(self):
        assert decimal_to_pct(0.105) == pytest.approx(10.5)
        assert decimal_to_pct(0.0)   == pytest.approx(0.0)
        assert decimal_to_pct(-0.05) == pytest.approx(-5.0)

    def test_round_trip(self):
        original = 12.345
        assert decimal_to_pct(pct_to_decimal(original)) == pytest.approx(original)


# ============================================================================
# SERIALISATION HELPERS
# ============================================================================

class TestSerialisation:
    def test_serialize_deserialize_single_symbol(self):
        bars = make_bars(5, "AAPL")
        data = {"AAPL": bars}

        serialized   = _serialize_data(data)
        deserialized = _deserialize_data(serialized)

        assert set(deserialized.keys()) == {"AAPL"}
        assert len(deserialized["AAPL"]) == 5
        assert deserialized["AAPL"][0].close        == pytest.approx(bars[0].close)
        assert deserialized["AAPL"][0].timestamp_ns == bars[0].timestamp_ns

    def test_serialize_preserves_all_fields(self):
        bars = make_bars(3, "X")
        data = {"X": bars}

        s = _serialize_data(data)
        d = _deserialize_data(s)

        for orig, restored in zip(bars, d["X"]):
            assert restored.symbol       == orig.symbol
            assert restored.timestamp_ns == orig.timestamp_ns
            assert restored.open         == pytest.approx(orig.open)
            assert restored.high         == pytest.approx(orig.high)
            assert restored.low          == pytest.approx(orig.low)
            assert restored.close        == pytest.approx(orig.close)
            assert restored.volume       == pytest.approx(orig.volume)

    def test_serialize_multi_symbol(self):
        data = make_multi(4, symbols=("X", "Y", "Z"))

        s = _serialize_data(data)
        d = _deserialize_data(s)

        assert set(d.keys()) == {"X", "Y", "Z"}
        for sym in ("X", "Y", "Z"):
            assert len(d[sym]) == 4


# ============================================================================
# _build_windows
# ============================================================================

class TestBuildWindows:
    def test_single_symbol_basic_window_count(self):
        # 10 bars, train=6, test=2.
        # pos=0: train [0,6), test [6,8)
        # pos=2: train [2,8), test [8,10)
        data    = {"A": make_bars(10)}
        windows = _build_windows(data, train_size=6, test_size=2)
        assert len(windows) == 2

    def test_window_bar_counts_match_sizes(self):
        data    = {"A": make_bars(10)}
        windows = _build_windows(data, train_size=6, test_size=2)

        for train, test in windows:
            assert len(train["A"]) == 6
            assert len(test["A"])  == 2

    def test_insufficient_data_returns_empty(self):
        data    = {"A": make_bars(5)}
        windows = _build_windows(data, train_size=4, test_size=4)
        assert windows == []

    def test_exact_fit_produces_one_window(self):
        data    = {"A": make_bars(6)}
        windows = _build_windows(data, train_size=4, test_size=2)
        assert len(windows) == 1

    def test_no_bar_shared_between_adjacent_test_windows(self):
        """Half-open intervals must ensure each bar appears in at most one test window."""
        data    = {"A": make_bars(12)}
        windows = _build_windows(data, train_size=4, test_size=4)

        all_test_timestamps = []
        for _, test in windows:
            all_test_timestamps.extend(b.timestamp_ns for b in test["A"])

        assert len(all_test_timestamps) == len(set(all_test_timestamps))

    def test_train_and_test_do_not_overlap(self):
        data    = {"A": make_bars(10)}
        windows = _build_windows(data, train_size=6, test_size=2)

        for train, test in windows:
            train_ts = {b.timestamp_ns for b in train["A"]}
            test_ts  = {b.timestamp_ns for b in test["A"]}
            assert train_ts.isdisjoint(test_ts)

    def test_multi_symbol_all_symbols_in_every_window(self):
        data    = make_multi(10, symbols=("X", "Y", "Z"))
        windows = _build_windows(data, train_size=6, test_size=2)

        assert len(windows) > 0
        for train, test in windows:
            assert set(train.keys()) == {"X", "Y", "Z"}
            assert set(test.keys())  == {"X", "Y", "Z"}

    def test_multi_symbol_aligned_series_have_matching_counts(self):
        data    = make_multi(10, symbols=("A", "B"))
        windows = _build_windows(data, train_size=6, test_size=2)

        for train, test in windows:
            assert len(train["A"]) == len(train["B"])
            assert len(test["A"])  == len(test["B"])

    def test_window_with_gap_in_secondary_symbol_is_skipped(self):
        """A window where any symbol has no bars must be dropped."""
        ref_bars = make_bars(10, "A", base_ts=0)
        # Symbol B only covers the first 6 timestamps; the second window's test
        # period falls outside B's range and must be skipped.
        b_bars   = make_bars(6, "B", base_ts=0)
        data     = {"A": ref_bars, "B": b_bars}

        windows = _build_windows(data, train_size=4, test_size=2)

        for train, test in windows:
            assert len(train["B"]) > 0
            assert len(test["B"])  > 0

    def test_reference_symbol_is_first_key(self):
        """Window sizes are measured in bars of the first symbol."""
        data    = make_multi(10, symbols=("REF", "OTHER"))
        windows = _build_windows(data, train_size=6, test_size=2)

        assert len(windows) > 0
        for train, _ in windows:
            assert len(train["REF"]) == 6


# ============================================================================
# PARAMETER GRID
# ============================================================================

class TestParameterGrid:
    def test_single_param(self):
        grid = list(ParameterGrid({"a": [1, 2, 3]}))
        assert len(grid) == 3
        assert {"a": 1} in grid

    def test_multiple_params_cartesian_product(self):
        grid = list(ParameterGrid({"a": [1, 2], "b": [10, 20]}))
        assert len(grid) == 4
        assert {"a": 1, "b": 10} in grid
        assert {"a": 2, "b": 20} in grid

    def test_len(self):
        pg = ParameterGrid({"a": [1, 2, 3], "b": [10, 20]})
        assert len(pg) == 6

    def test_empty_value_list(self):
        grid = list(ParameterGrid({"a": []}))
        assert grid == []


# ============================================================================
# OPTIMIZATION RESULT
# ============================================================================

class TestOptimizationResult:
    def _make(self, total_return=0.105, max_drawdown=-0.05):
        return OptimizationResult(
            params={"fast_period": 10, "slow_period": 50},
            sharpe_ratio=1.5,
            total_return=total_return,
            max_drawdown=max_drawdown,
            num_trades=20,
            final_value=110500.0,
        )

    def test_total_return_pct_property(self):
        assert self._make(total_return=0.105).total_return_pct == pytest.approx(10.5)

    def test_max_drawdown_pct_property(self):
        assert self._make(max_drawdown=-0.05).max_drawdown_pct == pytest.approx(-5.0)

    def test_repr_contains_sharpe_and_return(self):
        r = repr(self._make())
        assert "1.5" in r or "sharpe" in r.lower()
        assert "%" in r or "return" in r.lower()


# ============================================================================
# WALK-FORWARD ANALYSER; SINGLE-ASSET REGRESSION
# ============================================================================

class TestWalkForwardAnalyzerSingleAsset:
    """Verify the single-asset path still works correctly after the refactor."""

    def _run(self, n_bars=60, train=30, test=10) -> WalkForwardResult:
        return WalkForwardAnalyzer(
            strategy_factory=_BuyAndHoldFactory(),
            param_grid={"": [None]},
            train_size=train,
            test_size=test,
            metric="sharpe_ratio",
            n_jobs=1,
        ).analyze({"A": make_bars(n_bars)}, initial_capital=100_000.0, verbose=False)

    def test_returns_walk_forward_result(self):
        assert isinstance(self._run(), WalkForwardResult)

    def test_correct_number_of_windows(self):
        # 60 bars, train=30, test=10 → windows at pos 0, 10, 20 → 3 windows.
        result = self._run(n_bars=60, train=30, test=10)
        assert len(result.best_params_per_window) == 3

    def test_combined_equity_curve_non_empty(self):
        assert len(self._run().combined_equity_curve) > 0

    def test_overall_metrics_keys_present(self):
        metrics = self._run().overall_metrics
        for key in ("sharpe_ratio", "total_return", "max_drawdown", "num_windows"):
            assert key in metrics

    def test_num_windows_matches_oos_result_count(self):
        result = self._run()
        assert result.overall_metrics["num_windows"] == len(result.out_of_sample_results)

    def test_insufficient_data_raises(self):
        wfa = WalkForwardAnalyzer(
            strategy_factory=_BuyAndHoldFactory(),
            param_grid={},
            train_size=10,
            test_size=10,
        )
        with pytest.raises(ValueError, match="Not enough data"):
            wfa.analyze({"A": make_bars(5)})

    def test_summary_string_contains_expected_sections(self):
        summary = self._run().summary()
        assert "Walk-Forward" in summary
        assert "Window"       in summary


# ============================================================================
# WALK-FORWARD ANALYSER; MULTI-ASSET
# ============================================================================

class TestWalkForwardAnalyzerMultiAsset:

    def _run(self, n_bars=60, symbols=("A", "B"), train=30, test=10) -> WalkForwardResult:
        return WalkForwardAnalyzer(
            strategy_factory=_BuyAndHoldFactory(),
            param_grid={"": [None]},
            train_size=train,
            test_size=test,
            metric="sharpe_ratio",
            n_jobs=1,
        ).analyze(make_multi(n_bars, symbols=symbols), initial_capital=100_000.0, verbose=False)

    def test_two_symbols_returns_result(self):
        assert isinstance(self._run(), WalkForwardResult)

    def test_three_symbols_returns_result(self):
        assert isinstance(self._run(symbols=("A", "B", "C")), WalkForwardResult)

    def test_window_count_matches_single_asset(self):
        """Multi-asset produces the same window count as single-asset when series are aligned."""
        kwargs = dict(
            strategy_factory=_BuyAndHoldFactory(),
            param_grid={"": [None]},
            train_size=30,
            test_size=10,
            metric="sharpe_ratio",
            n_jobs=1,
        )
        single = WalkForwardAnalyzer(**kwargs).analyze({"A": make_bars(60)}, verbose=False)
        multi  = WalkForwardAnalyzer(**kwargs).analyze(make_multi(60, ("A", "B")), verbose=False)

        assert (len(single.best_params_per_window) ==
                len(multi.best_params_per_window))

    def test_combined_equity_curve_non_empty(self):
        assert len(self._run().combined_equity_curve) > 0

    def test_oos_results_have_expected_keys(self):
        for oos in self._run().out_of_sample_results:
            for key in ("sharpe_ratio", "total_return", "max_drawdown", "num_trades", "final_value"):
                assert key in oos
            # equity_curve must have been popped before storing
            assert "equity_curve" not in oos

    def test_no_warning_raised_for_multi_asset(self):
        """The old code emitted a UserWarning for len(data) > 1. That must be gone."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._run()  # must not raise any warning

    def test_combined_equity_curve_is_all_finite(self):
        curve = self._run(n_bars=80, train=30, test=10).combined_equity_curve
        assert np.all(np.isfinite(curve))


# ============================================================================
# MONTE CARLO VALIDATION; SINGLE ASSET
# ============================================================================

class TestMonteCarloSingleAsset:

    def _run(self, method="bootstrap", n=20):
        return monte_carlo_validation(
            strategy_factory=qc.BuyAndHold,
            params={},
            data={"A": make_bars(50)},
            n_simulations=n,
            initial_capital=100_000.0,
            method=method,
            n_jobs=1,
        )

    def test_returns_expected_keys(self):
        result = self._run()
        for key in ("sharpe_ratios", "returns", "drawdowns"):
            assert key in result

    def test_output_lengths_match_n_simulations(self):
        n      = 15
        result = self._run(n=n)
        assert len(result["sharpe_ratios"]) == n
        assert len(result["returns"])       == n
        assert len(result["drawdowns"])     == n

    def test_all_outputs_are_finite(self):
        result = self._run(n=10)
        for key in ("sharpe_ratios", "returns", "drawdowns"):
            assert np.all(np.isfinite(result[key]))

    def test_bootstrap_method(self):
        assert len(self._run(method="bootstrap", n=10)["returns"]) == 10

    def test_shuffle_method(self):
        assert len(self._run(method="shuffle", n=10)["returns"]) == 10

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            monte_carlo_validation(
                strategy_factory=qc.BuyAndHold,
                params={},
                data={"A": make_bars(20)},
                n_simulations=5,
                method="invalid_method",
            )

    def test_returns_in_decimal_format(self):
        """Returns must be in decimal format, not percentage."""
        result = self._run(n=20)
        # BuyAndHold on a mild drift series stays well within ±2 in decimal terms.
        assert np.all(np.abs(result["returns"]) < 2.0)

    def test_drawdowns_are_non_positive(self):
        result = self._run(n=10)
        assert np.all(result["drawdowns"] <= 1e-9)

    def test_different_seeds_produce_varied_results(self):
        """Each simulation uses a distinct seed; results must not all be identical."""
        result = self._run(n=20)
        assert len(set(result["returns"].tolist())) > 1


# ============================================================================
# MONTE CARLO VALIDATION; MULTI-ASSET
# ============================================================================

class TestMonteCarloMultiAsset:

    def _aligned(self, n=50, symbols=("A", "B")):
        return make_multi(n, symbols=symbols)

    def test_two_symbols_runs(self):
        result = monte_carlo_validation(
            strategy_factory=qc.BuyAndHold,
            params={},
            data=self._aligned(50, ("A", "B")),
            n_simulations=10,
            n_jobs=1,
        )
        assert len(result["returns"]) == 10

    def test_three_symbols_runs(self):
        result = monte_carlo_validation(
            strategy_factory=qc.BuyAndHold,
            params={},
            data=self._aligned(50, ("A", "B", "C")),
            n_simulations=10,
            n_jobs=1,
        )
        assert len(result["returns"]) == 10

    def test_misaligned_bar_counts_raises(self):
        """Different bar counts must raise ValueError, not silently truncate."""
        with pytest.raises(ValueError, match="same number of bars"):
            monte_carlo_validation(
                strategy_factory=qc.BuyAndHold,
                params={},
                data={"A": make_bars(50, "A"), "B": make_bars(40, "B")},
                n_simulations=5,
            )

    def test_error_message_includes_symbol_names_and_counts(self):
        with pytest.raises(ValueError) as exc_info:
            monte_carlo_validation(
                strategy_factory=qc.BuyAndHold,
                params={},
                data={"A": make_bars(50, "A"), "B": make_bars(30, "B")},
                n_simulations=5,
            )
        msg = str(exc_info.value)
        assert "A"  in msg
        assert "B"  in msg
        assert "50" in msg
        assert "30" in msg

    def test_no_warning_for_multi_asset(self):
        """The old code warned when len(data) > 1. That must be gone."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            monte_carlo_validation(
                strategy_factory=qc.BuyAndHold,
                params={},
                data=self._aligned(40, ("A", "B")),
                n_simulations=5,
                n_jobs=1,
            )

    def test_result_structure_matches_single_asset(self):
        """Result dict shape is identical regardless of asset count."""
        single = monte_carlo_validation(
            strategy_factory=qc.BuyAndHold, params={},
            data={"A": make_bars(40)}, n_simulations=5, n_jobs=1,
        )
        multi = monte_carlo_validation(
            strategy_factory=qc.BuyAndHold, params={},
            data=self._aligned(40, ("A", "B")), n_simulations=5, n_jobs=1,
        )

        assert single.keys() == multi.keys()
        for key in single:
            assert len(single[key]) == len(multi[key])

    def test_all_outputs_finite_for_multi_asset(self):
        result = monte_carlo_validation(
            strategy_factory=qc.BuyAndHold,
            params={},
            data=self._aligned(40, ("A", "B")),
            n_simulations=10,
            n_jobs=1,
        )
        for key in ("sharpe_ratios", "returns", "drawdowns"):
            assert np.all(np.isfinite(result[key]))


# ============================================================================
# GRID SEARCH OPTIMIZER; MULTI-ASSET SMOKE TEST
# ============================================================================

class TestGridSearchOptimizerMultiAsset:
    def test_multi_asset_data_accepted_and_returns_results(self):
        data = make_multi(60, symbols=("A", "B"))
        opt  = GridSearchOptimizer(
            strategy_factory=_BuyAndHoldFactory(),
            param_grid={"": [None]},
            metric="sharpe_ratio",
            n_jobs=1,
        )
        results = opt.optimize(data, initial_capital=100_000.0, verbose=False)

        assert isinstance(results, list)
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])