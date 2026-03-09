"""
python/quantcore/parquet_loader.py
===================================
Parquet data loader for QuantCore.

Reads OHLCV data from Parquet files and converts it to BarData objects
compatible with BacktestEngine.add_data(). Accepts either the numpy
array path (fastest) or the BarData list path, matching the same
column conventions as CSVDataLoader.

Expected schema (column names are case-insensitive):
    timestamp   — UNIX timestamp (int64 ns, us, ms, or s) or datetime64
    open        — open price
    high        — high price
    low         — low price
    close       — close price
    volume      — traded volume

Optionally a 'symbol' column may be present; if so, it takes priority
over the symbol argument passed to load().

Requires: pyarrow (pip install pyarrow)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from quantcore._core import BarData


# Column name aliases accepted for each field
_ALIASES = {
    "timestamp": {"timestamp", "time", "date", "datetime", "ts", "index"},
    "open":      {"open", "o"},
    "high":      {"high", "h"},
    "low":       {"low", "l"},
    "close":     {"close", "c", "adj close", "adj_close"},
    "volume":    {"volume", "vol", "v"},
    "symbol":    {"symbol", "ticker", "asset", "sym"},
}


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    """
    Map required field names to the actual column names found in the file.

    Returns a dict of {field: actual_column_name}.
    Raises ValueError if any required field cannot be resolved.
    """
    lower_map = {c.lower(): c for c in columns}
    resolved: dict[str, str] = {}

    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                resolved[field] = lower_map[alias]
                break

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - resolved.keys()
    if missing:
        raise ValueError(
            f"Parquet file is missing required columns: {missing}. "
            f"Found columns: {columns}"
        )

    return resolved


def _to_timestamp_ns(series) -> np.ndarray:
    """
    Convert a timestamp column to int64 nanoseconds.

    Handles:
      - datetime64 (any resolution) or pandas Timestamp
      - integer seconds / milliseconds / microseconds / nanoseconds
    """
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ns]").astype(np.int64)

    values = series.to_numpy(dtype=np.int64)

    max_val = int(values.max()) if len(values) > 0 else 0

    if max_val < 4_000_000_000:                  # seconds
        return values * 1_000_000_000
    elif max_val < 4_000_000_000_000:            # milliseconds
        return values * 1_000_000
    elif max_val < 4_000_000_000_000_000:        # microseconds
        return values * 1_000
    else:                                        # nanoseconds
        return values


class ParquetDataLoader:
    """Load OHLCV bar data from Parquet files."""

    @staticmethod
    def load(
            filepath: str | Path,
            symbol: str = "",
            max_skip_pct: float = 0.20,
    ) -> "List[BarData]":
        """
        Load a Parquet file and return a list of BarData objects.

        Args:
            filepath:     Path to the .parquet file.
            symbol:       Symbol name to assign when no 'symbol' column is present.
            max_skip_pct: Fraction of rows allowed to fail validation before
                          raising an error (default 0.20).

        Returns:
            List of BarData objects sorted by timestamp ascending.

        Raises:
            ImportError:  If pyarrow is not installed.
            ValueError:   If required columns are missing or too many rows fail.
            RuntimeError: If the file contains no valid data.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet support: pip install pyarrow"
            ) from exc

        import pandas as pd
        from quantcore._core import BarData

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Could not open file: {filepath}")

        table = pq.read_table(filepath)
        df    = table.to_pandas()

        col_map  = _resolve_columns(list(df.columns))
        has_sym  = "symbol" in col_map

        timestamps = _to_timestamp_ns(df[col_map["timestamp"]])
        opens      = df[col_map["open"]].to_numpy(dtype=np.float64)
        highs      = df[col_map["high"]].to_numpy(dtype=np.float64)
        lows       = df[col_map["low"]].to_numpy(dtype=np.float64)
        closes     = df[col_map["close"]].to_numpy(dtype=np.float64)
        volumes    = df[col_map["volume"]].to_numpy(dtype=np.float64)
        symbols    = df[col_map["symbol"]].tolist() if has_sym else None

        n          = len(df)
        bars: list = []
        bad_lines  = 0

        for i in range(n):
            sym = symbols[i] if symbols else symbol
            try:
                bar = BarData(
                    sym,
                    int(timestamps[i]),
                    float(opens[i]),
                    float(highs[i]),
                    float(lows[i]),
                    float(closes[i]),
                    float(volumes[i]),
                )
                bars.append(bar)
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping row {i}: {exc}", stacklevel=2)
                bad_lines += 1

        skip_pct = bad_lines / n if n > 0 else 0.0
        if skip_pct > max_skip_pct:
            raise ValueError(
                f"Data quality error: {bad_lines} out of {n} rows skipped "
                f"({skip_pct * 100:.1f}%), exceeds threshold of "
                f"{max_skip_pct * 100:.1f}%"
            )

        if not bars:
            raise RuntimeError(f"No valid data loaded from file: {filepath}")

        bars.sort(key=lambda b: b.timestamp_ns)
        return bars

    @staticmethod
    def load_numpy(
            filepath: str | Path,
            symbol: str = "",
    ) -> np.ndarray:
        """
        Load a Parquet file and return a (N, 6) float64 numpy array.

        Column order: [timestamp_ns, open, high, low, close, volume]

        This array can be passed directly to BacktestEngine.add_data(symbol, array),
        which uses a single boundary crossing instead of N BarData objects.

        Args:
            filepath: Path to the .parquet file.
            symbol:   Unused — present for API symmetry with load(). Symbol is
                      supplied separately to add_data().

        Returns:
            numpy array of shape (N, 6), sorted by timestamp ascending.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet support: pip install pyarrow"
            ) from exc

        import pandas as pd

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Could not open file: {filepath}")

        table = pq.read_table(filepath)
        df    = table.to_pandas()

        col_map    = _resolve_columns(list(df.columns))
        timestamps = _to_timestamp_ns(df[col_map["timestamp"]])

        arr = np.empty((len(df), 6), dtype=np.float64)
        arr[:, 0] = timestamps.astype(np.float64)
        arr[:, 1] = df[col_map["open"]].to_numpy(dtype=np.float64)
        arr[:, 2] = df[col_map["high"]].to_numpy(dtype=np.float64)
        arr[:, 3] = df[col_map["low"]].to_numpy(dtype=np.float64)
        arr[:, 4] = df[col_map["close"]].to_numpy(dtype=np.float64)
        arr[:, 5] = df[col_map["volume"]].to_numpy(dtype=np.float64)

        # sort by timestamp
        order = np.argsort(arr[:, 0], kind="stable")
        return arr[order]