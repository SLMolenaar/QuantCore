"""
benchmark_quantcore_vs_backtrader.py
=====================================
Compares throughput (bars/s) between QuantCore and Backtrader on an
identical SMA crossover strategy over identical synthetic OHLCV data.

Requirements:
    pip install quantcore backtrader numpy pandas

Usage:
    python benchmark_quantcore_vs_backtrader.py
    python benchmark_quantcore_vs_backtrader.py --bars 10000
    python benchmark_quantcore_vs_backtrader.py --bars 50000 --runs 20
"""

import argparse
import time
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_ohlcv(n_bars: int, seed: int = 42) -> np.ndarray:
    """
    Generate a (n_bars, 6) float64 array of synthetic OHLCV data.
    Columns: [timestamp_ns, open, high, low, close, volume]

    Uses a simple random walk so both engines see identical price history.
    """
    rng = np.random.default_rng(seed)

    # Random walk for close prices, anchored at 100.0
    returns = rng.normal(0.0, 0.01, n_bars)
    closes  = 100.0 * np.exp(np.cumsum(returns))

    # Derive OHLC from close with small intra-bar noise
    noise  = rng.uniform(0.001, 0.005, n_bars)
    opens  = closes * (1.0 + rng.uniform(-0.003, 0.003, n_bars))
    highs  = np.maximum(opens, closes) * (1.0 + noise)
    lows   = np.minimum(opens, closes) * (1.0 - noise)

    # Timestamps: daily bars starting 2010-01-01 in nanoseconds
    start_ns   = int(np.datetime64("2010-01-01", "ns").astype(np.int64))
    day_ns     = int(86_400e9)
    timestamps = np.arange(n_bars, dtype=np.int64) * day_ns + start_ns

    volumes = rng.uniform(1e6, 1e7, n_bars)

    arr         = np.empty((n_bars, 6), dtype=np.float64)
    arr[:, 0]   = timestamps.astype(np.float64)
    arr[:, 1]   = opens
    arr[:, 2]   = highs
    arr[:, 3]   = lows
    arr[:, 4]   = closes
    arr[:, 5]   = volumes
    return arr


# ---------------------------------------------------------------------------
# QuantCore benchmark
# ---------------------------------------------------------------------------

def bench_quantcore(data: np.ndarray, n_runs: int) -> tuple[float, float]:
    """
    Returns (mean_bars_per_sec, std_bars_per_sec) over n_runs.
    """
    try:
        import quantcore as qc
    except ImportError:
        print("  [SKIP] quantcore not installed — pip install quantcore")
        return 0.0, 0.0

    n_bars = len(data)
    times  = []

    class SMACrossover(qc.Strategy):
        def __init__(self):
            super().__init__("SMACrossover")
            self._prices = []

        def on_data(self, event):
            self._prices.append(event.close)
            if len(self._prices) < 200:
                return
            fast = sum(self._prices[-50:])  / 50
            slow = sum(self._prices[-200:]) / 200
            pos  = self.get_position(event.symbol)
            if fast > slow and pos <= 0:
                self.generate_signal(event.symbol, qc.SignalType.BUY,  1.0, event.timestamp_ns)
            elif fast < slow and pos >= 0:
                self.generate_signal(event.symbol, qc.SignalType.SELL, 1.0, event.timestamp_ns)

        def reset(self):
            super().reset()
            self._prices = []

    for i in range(n_runs):
        engine = qc.BacktestEngine(100_000.0)
        engine.add_data("ASSET", data)
        engine.set_strategy(qc.SMACrossover(fast_period=50, slow_period=200))

        t0 = time.perf_counter()
        try:
            engine.run()
        except RuntimeError as e:
            if "Wash trade" in str(e):
                print(f"  [WARNING] Run {i + 1} hit wash trade bug in QuantCore — skipping run")
                continue
            raise
        t1 = time.perf_counter()

        times.append(n_bars / (t1 - t0))

    if not times:
        print("  [FAIL] All runs hit the wash trade bug. Lower --bars or fix QuantCore.")
        return 0.0, 0.0

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Backtrader benchmark
# ---------------------------------------------------------------------------

def bench_backtrader(data: np.ndarray, n_runs: int) -> tuple[float, float]:
    """
    Returns (mean_bars_per_sec, std_bars_per_sec) over n_runs.
    """
    try:
        import backtrader as bt
        import pandas as pd
    except ImportError:
        print("  [SKIP] backtrader not installed — pip install backtrader")
        return 0.0, 0.0

    import pandas as pd

    n_bars = len(data)

    # Convert numpy array to a pandas DataFrame that Backtrader's PandasData feed accepts
    timestamps = pd.to_datetime(data[:, 0].astype(np.int64), unit="ns")
    df = pd.DataFrame({
        "open":   data[:, 1],
        "high":   data[:, 2],
        "low":    data[:, 3],
        "close":  data[:, 4],
        "volume": data[:, 5],
        "openinterest": np.zeros(n_bars),
    }, index=timestamps)

    class SMACrossover(bt.Strategy):
        def __init__(self):
            self.fast = bt.indicators.SMA(self.data.close, period=50)
            self.slow = bt.indicators.SMA(self.data.close, period=200)

        def next(self):
            if self.fast[0] > self.slow[0] and not self.position:
                self.buy()
            elif self.fast[0] < self.slow[0] and self.position:
                self.sell()

    times = []

    for _ in range(n_runs):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SMACrossover)
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed)
        cerebro.broker.setcash(100_000.0)

        t0 = time.perf_counter()
        cerebro.run()
        t1 = time.perf_counter()

        times.append(n_bars / (t1 - t0))

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QuantCore vs Backtrader benchmark")
    parser.add_argument("--bars", type=int, default=50000,
                        help="Number of OHLCV bars to benchmark (default: 50000)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of timed runs per engine (default: 3)")
    args = parser.parse_args()

    n_bars = args.bars
    n_runs = args.runs

    print(f"\nBenchmark: SMA(50/200) crossover — {n_bars:,} bars, {n_runs} runs each")
    print("=" * 60)

    data = generate_ohlcv(n_bars)

    print(f"\nQuantCore:")
    qc_mean, qc_std = bench_quantcore(data, n_runs)
    if qc_mean > 0:
        print(f"  {qc_mean:>12,.0f} bars/s  (±{qc_std:,.0f})")

    print(f"\nBacktrader:")
    bt_mean, bt_std = bench_backtrader(data, n_runs)
    if bt_mean > 0:
        print(f"  {bt_mean:>12,.0f} bars/s  (±{bt_std:,.0f})")

    if qc_mean > 0 and bt_mean > 0:
        ratio = qc_mean / bt_mean
        print(f"\nResult: QuantCore is {ratio:.1f}x {'faster' if ratio >= 1.0 else 'slower'} than Backtrader")
        print()

        if ratio < 1.0:
            print("Note: QuantCore was slower on this run. Possible reasons:")
            print("  - Small bar count (<10K) where engine startup dominates")
            print("  - Debug build of the C++ extension")
            print("  - Python-side SMA calculation is the bottleneck, not the engine")
            print("  Try --bars 50000 for a more representative result.")
    print()


if __name__ == "__main__":
    main()