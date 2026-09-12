"""
Tests for TickParquetLoader.
pyarrow is required; tests are skipped if not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pyarrow", reason="pyarrow not installed")

import quantcore as qc
from quantcore.tick_parquet_loader import (
    TickParquetLoader,
    _resolve_columns,
    _to_timestamp_ns,
    _parse_side_array,
)

SEC = 1_000_000_000


def _write_parquet(tmp_path, df: pd.DataFrame, name: str = "ticks.parquet") -> Path:
    path = tmp_path / name
    df.to_parquet(path)
    return path


def _sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [1_700_000_000 + i for i in range(n)],  # seconds
        "price":     np.linspace(100.0, 104.0, n),
        "quantity":  np.linspace(10.0, 50.0, n),
    })


class TestResolveColumns:
    def test_standard_names(self):
        resolved = _resolve_columns(["timestamp", "price", "quantity", "side"])
        assert resolved["price"] == "price"

    def test_case_insensitive_and_aliases(self):
        resolved = _resolve_columns(["Time", "Px", "Qty", "Aggressor"])
        assert resolved["timestamp"] == "Time"
        assert resolved["price"] == "Px"
        assert resolved["quantity"] == "Qty"
        assert resolved["side"] == "Aggressor"

    def test_side_is_optional(self):
        resolved = _resolve_columns(["timestamp", "price", "quantity"])
        assert "side" not in resolved

    def test_missing_required_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            _resolve_columns(["timestamp", "price"])


class TestToTimestampNs:
    def test_seconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000]))
        assert result[0] == 1_700_000_000 * SEC

    def test_milliseconds(self):
        result = _to_timestamp_ns(pd.Series([1_700_000_000_000]))
        assert result[0] == 1_700_000_000 * SEC

    def test_datetime64(self):
        series = pd.Series(pd.to_datetime(["2023-01-01"]))
        result = _to_timestamp_ns(series)
        assert result[0] == int(pd.Timestamp("2023-01-01").value)


class TestParseSideArray:
    def test_buy_aliases(self):
        result = _parse_side_array(pd.Series(["b", "BUY", "Bid", "1", "long"]))
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_sell_aliases(self):
        result = _parse_side_array(pd.Series(["s", "SELL", "-1", "short"]))
        np.testing.assert_array_equal(result, [1, 1, 1, 1])

    def test_unknown_value_defaults_to_buy(self):
        result = _parse_side_array(pd.Series(["unknown"]))
        assert result[0] == 0

    def test_whitespace_and_case_insensitive(self):
        result = _parse_side_array(pd.Series([" Sell ", " BUY "]))
        np.testing.assert_array_equal(result, [1, 0])


class TestLoad:
    def test_happy_path_returns_sorted_tickdata(self, tmp_path):
        df = _sample_df(5)
        path = _write_parquet(tmp_path, df)

        ticks = TickParquetLoader.load(path, symbol="AAPL")

        assert len(ticks) == 5
        assert all(t.symbol == "AAPL" for t in ticks)
        timestamps = [t.timestamp_ns for t in ticks]
        assert timestamps == sorted(timestamps)

        for i, t in enumerate(ticks):
            assert t.timestamp_ns == int(df["timestamp"].iloc[i]) * SEC
            assert t.price == pytest.approx(df["price"].iloc[i])
            assert t.quantity == pytest.approx(df["quantity"].iloc[i])

    def test_default_side_is_buy_when_column_absent(self, tmp_path):
        df = _sample_df(3)
        path = _write_parquet(tmp_path, df)

        ticks = TickParquetLoader.load(path, symbol="AAPL")

        assert all(t.aggressor_side == qc.Side.BUY for t in ticks)

    def test_side_column_is_respected(self, tmp_path):
        df = _sample_df(3)
        df["side"] = ["buy", "sell", "buy"]
        path = _write_parquet(tmp_path, df)

        ticks = TickParquetLoader.load(path, symbol="AAPL")

        assert [t.aggressor_side for t in ticks] == [qc.Side.BUY, qc.Side.SELL, qc.Side.BUY]

    def test_descending_input_is_sorted_ascending(self, tmp_path):
        df = _sample_df(5).iloc[::-1].reset_index(drop=True)
        path = _write_parquet(tmp_path, df)

        ticks = TickParquetLoader.load(path, symbol="AAPL")

        timestamps = [t.timestamp_ns for t in ticks]
        assert timestamps == sorted(timestamps)
        assert timestamps[0] < timestamps[-1]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TickParquetLoader.load(tmp_path / "nope.parquet")

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"timestamp": [1_700_000_000]})
        path = _write_parquet(tmp_path, df)

        with pytest.raises(ValueError, match="missing required columns"):
            TickParquetLoader.load(path)

    def test_empty_file_raises_runtime_error(self, tmp_path):
        df = _sample_df(0)
        path = _write_parquet(tmp_path, df)

        with pytest.raises(RuntimeError, match="No valid tick data loaded"):
            TickParquetLoader.load(path)


class TestLoadNumpy:
    def test_shape_and_all_columns_match_source(self, tmp_path):
        df = _sample_df(4)
        df["side"] = ["buy", "sell", "buy", "sell"]
        path = _write_parquet(tmp_path, df)

        arr = TickParquetLoader.load_numpy(path)

        assert arr.shape == (4, 4)
        assert arr.dtype == np.float64
        np.testing.assert_array_almost_equal(arr[:, 0], df["timestamp"].to_numpy() * SEC)
        np.testing.assert_array_almost_equal(arr[:, 1], df["price"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 2], df["quantity"].to_numpy())
        np.testing.assert_array_almost_equal(arr[:, 3], [0.0, 1.0, 0.0, 1.0])

    def test_side_defaults_to_zero_when_absent(self, tmp_path):
        df = _sample_df(3)
        path = _write_parquet(tmp_path, df)

        arr = TickParquetLoader.load_numpy(path)

        np.testing.assert_array_equal(arr[:, 3], [0.0, 0.0, 0.0])

    def test_sorted_by_timestamp(self, tmp_path):
        df = _sample_df(4).iloc[::-1].reset_index(drop=True)
        path = _write_parquet(tmp_path, df)

        arr = TickParquetLoader.load_numpy(path)

        assert list(arr[:, 0]) == sorted(arr[:, 0])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TickParquetLoader.load_numpy(tmp_path / "nope.parquet")

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"timestamp": [1_700_000_000]})
        path = _write_parquet(tmp_path, df)

        with pytest.raises(ValueError, match="missing required columns"):
            TickParquetLoader.load_numpy(path)
