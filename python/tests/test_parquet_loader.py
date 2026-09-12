"""
Tests for ParquetDataLoader.
pyarrow is required; tests are skipped if not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pyarrow", reason="pyarrow not installed")

from quantcore.parquet_loader import ParquetDataLoader, _resolve_columns, _to_timestamp_ns

SEC = 1_000_000_000


def _write_parquet(tmp_path, df: pd.DataFrame, name: str = "data.parquet") -> Path:
    path = tmp_path / name
    df.to_parquet(path)
    return path


def _sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [1_700_000_000 + i * 86_400 for i in range(n)],  # seconds
        "open":      np.linspace(100.0, 104.0, n),
        "high":      np.linspace(101.0, 105.0, n),
        "low":       np.linspace(99.0, 103.0, n),
        "close":     np.linspace(100.5, 104.5, n),
        "volume":    np.linspace(1000.0, 5000.0, n),
    })


class TestResolveColumns:
    def test_standard_names(self):
        resolved = _resolve_columns(["timestamp", "open", "high", "low", "close", "volume"])
        assert resolved["open"] == "open"

    def test_case_insensitive_and_aliases(self):
        resolved = _resolve_columns(["Date", "O", "H", "L", "Adj Close", "Vol"])
        assert resolved["timestamp"] == "Date"
        assert resolved["open"] == "O"
        assert resolved["close"] == "Adj Close"
        assert resolved["volume"] == "Vol"

    def test_missing_required_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            _resolve_columns(["timestamp", "open", "high", "low"])

    def test_optional_symbol_not_required(self):
        resolved = _resolve_columns(["timestamp", "open", "high", "low", "close", "volume"])
        assert "symbol" not in resolved


class TestToTimestampNs:
    def test_seconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000]))
        assert result[0] == 1_700_000_000 * SEC

    def test_milliseconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000_000]))
        assert result[0] == 1_700_000_000 * SEC

    def test_microseconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000_000_000]))
        assert result[0] == 1_700_000_000 * SEC

    def test_nanoseconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000_000_000_000]))
        assert result[0] == 1_700_000_000_000_000_000

    def test_datetime64(self):
        series = pd.Series(pd.to_datetime(["2023-01-01"]))
        result = _to_timestamp_ns(series)
        expected = int(pd.Timestamp("2023-01-01").value)
        assert result[0] == expected

    def test_empty_series_returns_empty_array(self):
        result = _to_timestamp_ns(pd.Series([], dtype=np.int64))
        assert len(result) == 0


class TestLoad:
    def test_happy_path_returns_sorted_bardata(self, tmp_path):
        df = _sample_df(5)
        path = _write_parquet(tmp_path, df)

        bars = ParquetDataLoader.load(path, symbol="AAPL")

        assert len(bars) == 5
        assert all(b.symbol == "AAPL" for b in bars)
        assert [b.timestamp_ns for b in bars] == sorted(b.timestamp_ns for b in bars)

        for i, b in enumerate(bars):
            assert b.timestamp_ns == int(df["timestamp"].iloc[i]) * SEC
            assert b.open   == pytest.approx(df["open"].iloc[i])
            assert b.high   == pytest.approx(df["high"].iloc[i])
            assert b.low    == pytest.approx(df["low"].iloc[i])
            assert b.close  == pytest.approx(df["close"].iloc[i])
            assert b.volume == pytest.approx(df["volume"].iloc[i])

    def test_descending_input_is_sorted_ascending(self, tmp_path):
        df = _sample_df(5).iloc[::-1].reset_index(drop=True)
        path = _write_parquet(tmp_path, df)

        bars = ParquetDataLoader.load(path, symbol="AAPL")

        timestamps = [b.timestamp_ns for b in bars]
        assert timestamps == sorted(timestamps)
        assert timestamps[0] < timestamps[-1]

    def test_column_aliases_are_accepted(self, tmp_path):
        df = _sample_df(3).rename(columns={
            "timestamp": "Date", "open": "O", "high": "H",
            "low": "L", "close": "Adj Close", "volume": "Vol",
        })
        path = _write_parquet(tmp_path, df)

        bars = ParquetDataLoader.load(path, symbol="AAPL")

        assert len(bars) == 3

    def test_empty_file_raises_runtime_error(self, tmp_path):
        df = _sample_df(0)
        path = _write_parquet(tmp_path, df)

        with pytest.raises(RuntimeError, match="No valid data loaded"):
            ParquetDataLoader.load(path)

    def test_symbol_column_overrides_argument(self, tmp_path):
        df = _sample_df(3)
        df["symbol"] = ["MSFT"] * 3
        path = _write_parquet(tmp_path, df)

        bars = ParquetDataLoader.load(path, symbol="AAPL")

        assert all(b.symbol == "MSFT" for b in bars)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ParquetDataLoader.load(tmp_path / "nope.parquet")

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"timestamp": [1_700_000_000], "open": [1.0]})
        path = _write_parquet(tmp_path, df)

        with pytest.raises(ValueError, match="missing required columns"):
            ParquetDataLoader.load(path)

    def test_bad_rows_beyond_threshold_raises(self, tmp_path):
        # NaN symbol values fail BarData's str-typed constructor arg (TypeError),
        # which is the per-row failure path exercised by the skip-counting logic.
        df = _sample_df(5)
        df["symbol"] = ["AAA", np.nan, np.nan, "AAA", "AAA"]
        path = _write_parquet(tmp_path, df)

        with pytest.raises(ValueError, match="Data quality error"):
            ParquetDataLoader.load(path, max_skip_pct=0.1)

    def test_bad_rows_within_threshold_are_skipped(self, tmp_path):
        df = _sample_df(5)
        df["symbol"] = ["AAA", np.nan, "AAA", "AAA", "AAA"]
        path = _write_parquet(tmp_path, df)

        with pytest.warns(UserWarning, match="Skipping row 1"):
            bars = ParquetDataLoader.load(path, max_skip_pct=0.5)

        assert len(bars) == 4
        assert all(b.symbol == "AAA" for b in bars)

    def test_zero_valid_rows_raises_runtime_error(self, tmp_path):
        df = _sample_df(2)
        df["symbol"] = [np.nan, np.nan]
        path = _write_parquet(tmp_path, df)

        with pytest.raises(RuntimeError, match="No valid data loaded"):
            ParquetDataLoader.load(path, max_skip_pct=1.0)


class TestLoadNumpy:
    def test_shape_and_all_columns_match_source(self, tmp_path):
        df = _sample_df(4)
        path = _write_parquet(tmp_path, df)

        arr = ParquetDataLoader.load_numpy(path)

        assert arr.shape == (4, 6)
        assert arr.dtype == np.float64
        np.testing.assert_array_almost_equal(arr[:, 0], df["timestamp"].to_numpy() * SEC)
        np.testing.assert_array_almost_equal(arr[:, 1], df["open"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 2], df["high"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 3], df["low"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 4], df["close"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 5], df["volume"].to_numpy())

    def test_sorted_by_timestamp(self, tmp_path):
        df = _sample_df(4).iloc[::-1].reset_index(drop=True)  # descending order
        path = _write_parquet(tmp_path, df)

        arr = ParquetDataLoader.load_numpy(path)

        assert list(arr[:, 0]) == sorted(arr[:, 0])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ParquetDataLoader.load_numpy(tmp_path / "nope.parquet")

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"timestamp": [1_700_000_000], "open": [1.0]})
        path = _write_parquet(tmp_path, df)

        with pytest.raises(ValueError, match="missing required columns"):
            ParquetDataLoader.load_numpy(path)

    def test_symbol_column_ignored_since_return_is_numeric(self, tmp_path):
        df = _sample_df(3)
        df["symbol"] = ["A", "B", "C"]
        path = _write_parquet(tmp_path, df)

        arr = ParquetDataLoader.load_numpy(path)

        assert arr.shape == (3, 6)
