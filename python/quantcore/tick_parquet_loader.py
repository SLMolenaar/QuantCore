"""
python/quantcore/tick_parquet_loader.py
========================================
Parquet loader for tick data.

Expected schema (column names are case-insensitive):
    timestamp   -- UNIX timestamp (int64 ns, us, ms, or s) or datetime64
    price       -- trade price
    quantity    -- traded quantity
    side        -- optional, 'buy'/'sell' or 'B'/'S'

Requires: pyarrow (pip install pyarrow)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


_ALIASES = {
    "timestamp": {"timestamp", "time", "date", "datetime", "ts", "index"},
    "price":     {"price", "px", "trade_price", "last", "last_price"},
    "quantity":  {"quantity", "qty", "size", "volume", "vol", "v"},
    "side":      {"side", "aggressor", "aggressor_side", "direction"},
}

_SIDE_BUY  = {"b", "buy",  "bid", "1",  "long"}
_SIDE_SELL = {"s", "sell", "-1", "short"}


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    lower_map = {c.lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                resolved[field] = lower_map[alias]
                break
    required = {"timestamp", "price", "quantity"}
    missing = required - resolved.keys()
    if missing:
        raise ValueError(
            f"Parquet file is missing required columns: {missing}. "
            f"Found columns: {columns}"
        )
    return resolved


def _to_timestamp_ns(series) -> np.ndarray:
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ns]").astype(np.int64)
    values = series.to_numpy(dtype=np.int64)
    max_val = int(values.max()) if len(values) > 0 else 0
    if   max_val < 4_000_000_000:               return values * 1_000_000_000
    elif max_val < 4_000_000_000_000:           return values * 1_000_000
    elif max_val < 4_000_000_000_000_000:       return values * 1_000
    else:                                        return values


def _parse_side_array(series) -> np.ndarray:
    """Convert a side column to int8 array: 0 = Buy, 1 = Sell."""
    result = np.zeros(len(series), dtype=np.int8)
    for i, val in enumerate(series):
        s = str(val).strip().lower()
        if s in _SIDE_SELL:
            result[i] = 1
    return result


class TickParquetLoader:
    """Load tick data from Parquet files."""

    @staticmethod
    def load(
            filepath: str | Path,
            symbol:   str = "",
    ) -> "List":
        """
        Load a Parquet file and return a list of TickData objects.

        Returns:
            List of TickData objects sorted by timestamp ascending.

        Raises:
            ImportError:  If pyarrow is not installed.
            ValueError:   If required columns are missing.
            RuntimeError: If the file contains no valid data.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet support: pip install pyarrow"
            ) from exc

        from quantcore import TickData, Side

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Could not open file: {filepath}")

        table   = pq.read_table(filepath)
        df      = table.to_pandas()
        col_map = _resolve_columns(list(df.columns))

        timestamps = _to_timestamp_ns(df[col_map["timestamp"]])
        prices     = df[col_map["price"]].to_numpy(dtype=np.float64)
        quantities = df[col_map["quantity"]].to_numpy(dtype=np.float64)

        has_side   = "side" in col_map
        sides      = _parse_side_array(df[col_map["side"]]) if has_side else None

        ticks = []
        for i in range(len(df)):
            side = Side.SELL if (has_side and sides[i] == 1) else Side.BUY
            try:
                ticks.append(TickData(symbol, int(timestamps[i]),
                                      float(prices[i]), float(quantities[i]), side))
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping row {i}: {exc}", stacklevel=2)

        if not ticks:
            raise RuntimeError(f"No valid tick data loaded from: {filepath}")

        ticks.sort(key=lambda t: t.timestamp_ns)
        return ticks

    @staticmethod
    def _read_sorted(filepath: str | Path):
        """Read a tick file; return (timestamps_ns int64, values float64 (N, 3)) sorted by time."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet support: pip install pyarrow"
            ) from exc

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Could not open file: {filepath}")

        df      = pq.read_table(filepath).to_pandas()
        col_map = _resolve_columns(list(df.columns))

        timestamps = _to_timestamp_ns(df[col_map["timestamp"]])
        vals = np.empty((len(df), 3), dtype=np.float64)
        vals[:, 0] = df[col_map["price"]].to_numpy(dtype=np.float64)
        vals[:, 1] = df[col_map["quantity"]].to_numpy(dtype=np.float64)
        vals[:, 2] = _parse_side_array(df[col_map["side"]]).astype(np.float64) \
            if "side" in col_map else 0.0

        order = np.argsort(timestamps, kind="stable")
        return timestamps[order], vals[order]

    @staticmethod
    def load_numpy(filepath: str | Path) -> np.ndarray:
        """
        Load a Parquet tick file and return a (N, 4) float64 numpy array.

        Column order: [timestamp_ns, price, quantity, side]
        side encoding: 0.0 = Buy, 1.0 = Sell

        Sorted by timestamp ascending. Timestamps are float64 here, which cannot
        represent epoch-nanosecond values exactly (~256 ns steps); for ns-resolution
        ticks use load_numpy_exact().
        """
        ts, vals = TickParquetLoader._read_sorted(filepath)
        arr = np.empty((len(ts), 4), dtype=np.float64)
        arr[:, 0]  = ts.astype(np.float64)
        arr[:, 1:] = vals
        return arr

    @staticmethod
    def load_numpy_exact(filepath: str | Path):
        """
        Load a Parquet tick file with exact int64 nanosecond timestamps.

        Returns (timestamps_ns, data): int64 (N,) and float64 (N, 3) with columns
        [price, quantity, side]. Unpack straight into the engine:

            engine.add_tick_data("AAPL", *TickParquetLoader.load_numpy_exact(path))
        """
        return TickParquetLoader._read_sorted(filepath)
