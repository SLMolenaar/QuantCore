"""
benchmarks/bench_tick_python.py
================================
Python-layer benchmarks for QuantCore tick data support.

Run from the project root after building the extension:
    python python/build_module.py
    python benchmarks/bench_tick_python.py

Requirements: quantcore extension built, numpy
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

try:
    import quantcore as qc
except ImportError as e:
    print(f"ERROR: Could not import quantcore: {e}")
    print("Build the extension first:  python python/build_module.py")
    sys.exit(1)


# ============================================================================
# Synthetic data helpers
# ============================================================================

def make_ticks(symbol: str, n: int,
               interval_ns: int = 1_000_000_000,
               start_price: float = 100.0) -> list:
    ticks = []
    start_ns = 1_577_836_800_000_000_000  # 2020-01-01
    price    = start_price
    for i in range(n):
        price *= 1.0 + 0.00005 * math.sin(i * 0.01)
        side   = qc.Side.BUY if i % 2 == 0 else qc.Side.SELL
        ticks.append(qc.TickData(symbol, start_ns + i * interval_ns,
                                 price, 100.0, side))
    return ticks


def make_ticks_numpy(n: int,
                     interval_ns: int = 1_000_000_000,
                     start_price: float = 100.0) -> np.ndarray:
    """
    Build tick data as a (N, 4) float64 array:
    [timestamp_ns, price, quantity, side]
    side encoding: 0.0 = Buy, 1.0 = Sell
    """
    arr   = np.empty((n, 4), dtype=np.float64)
    price = start_price
    for i in range(n):
        price       *= 1.0 + 0.00005 * math.sin(i * 0.01)
        arr[i, 0]    = float(i * interval_ns)
        arr[i, 1]    = price
        arr[i, 2]    = 100.0
        arr[i, 3]    = 0.0 if i % 2 == 0 else 1.0
    return arr


def make_bars(symbol: str, n: int, start_price: float = 100.0) -> list:
    bars     = []
    start_ns = 1_577_836_800_000_000_000
    day_ns   = 86_400_000_000_000
    price    = start_price
    for i in range(n):
        price *= 1.0 + 0.0002 + 0.01 * math.sin(i * 0.05)
        bars.append(qc.BarData(symbol, start_ns + i * day_ns,
                               price, price * 1.005, price * 0.995,
                               price, 1_000_000.0))
    return bars


# ============================================================================
# Timing helpers
# ============================================================================

def timeit(fn, warmup: int = 2, runs: int = 10) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return times


def median(times: list[float]) -> float:
    s = sorted(times)
    return s[len(s) // 2]


def pct(times: list[float], p: int) -> float:
    s = sorted(times)
    return s[len(s) * p // 100]


# ============================================================================
# Formatting
# ============================================================================

def sep(c: str = "-", w: int = 72) -> None:
    print(c * w)


def section(title: str) -> None:
    sep("=")
    print(f"  {title}")
    sep("=")


# ============================================================================
# 1. Tick throughput
# ============================================================================

def bench_tick_throughput():
    section("1. Tick Throughput, single asset (BuyAndHold)")

    fmt = f"  {{:<34}} {{:>10}}  {{:>14}}  {{:>10}}"
    print(fmt.format("Scenario", "Ticks", "Ticks/s", "p50 (ms)"))
    sep()

    for n, interval_ns in [
        (1_000,     1_000_000_000),
        (10_000,    1_000_000_000),
        (60_000,    1_000_000_000),
        (360_000,   1_000_000_000),
    ]:
        ticks = make_ticks("ASSET", n, interval_ns)

        def run(t=ticks):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("ASSET", t)
            eng.set_strategy(qc.BuyAndHold())
            eng.run()

        times = timeit(run, warmup=2, runs=10)
        p50   = median(times)
        tps   = n / p50

        label = f"{n // 1000}K ticks (1s interval)"
        print(fmt.format(label, f"{n:,}", f"{tps:,.0f}", f"{p50 * 1000:.2f}"))

    print()


# ============================================================================
# 2. MM throttle comparison
# ============================================================================

def bench_mm_throttle():
    section("2. Market Maker Throttle; 10K ticks, varying refresh interval")

    n     = 10_000
    ticks = make_ticks("ASSET", n, 100_000_000)  # 100ms ticks

    fmt = f"  {{:<34}} {{:>14}}  {{:>10}}"
    print(fmt.format("Config", "Ticks/s", "p50 (ms)"))
    sep()

    configs = [
        ("No throttle (every tick)",   0),
        ("100ms refresh interval",     100_000_000),
        ("1s refresh interval",        1_000_000_000),
        ("10s refresh interval",       10_000_000_000),
    ]

    for label, interval in configs:
        def run(t=ticks, iv=interval):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("ASSET", t)
            eng.set_mm_refresh_interval(iv)
            eng.set_strategy(qc.BuyAndHold())
            eng.run()

        times = timeit(run, warmup=2, runs=10)
        p50   = median(times)
        tps   = n / p50

        print(fmt.format(label, f"{tps:,.0f}", f"{p50 * 1000:.2f}"))

    print()


# ============================================================================
# 3. Aggregation throughput
# ============================================================================

def bench_aggregation():
    section("3. Aggregation Throughput; aggregate_to_bars")

    n     = 1_000_000
    ticks = make_ticks("ASSET", n, 1_000_000_000)

    fmt = f"  {{:<26}} {{:>10}}  {{:>16}}  {{:>10}}"
    print(fmt.format("Bar duration", "Bars out", "Ticks/s", "p50 (ms)"))
    sep()

    durations = [
        ("1-second bars",    1_000_000_000),
        ("1-minute bars",   60_000_000_000),
        ("1-hour bars",  3_600_000_000_000),
        ("1-day bars",  86_400_000_000_000),
    ]

    for label, dur_ns in durations:
        def run(t=ticks, d=dur_ns):
            return qc.aggregate_ticks_to_bars(t, d)

        times    = timeit(run, warmup=2, runs=10)
        p50      = median(times)
        tps      = n / p50
        bars_out = len(run())

        print(fmt.format(label, f"{bars_out:,}", f"{tps:,.0f}", f"{p50 * 1000:.2f}"))

    print()


# ============================================================================
# 4. Tick vs bar; equivalent dataset
# ============================================================================

def bench_tick_vs_bar():
    section("4. Tick vs Bar, equivalent dataset, BuyAndHold")
    print("  252 daily bars  vs  252 * N ticks over the same period")
    print()

    fmt = f"  {{:<40}} {{:>10}}  {{:>14}}  {{:>10}}"
    print(fmt.format("Mode", "Events", "Events/s", "p50 (ms)"))
    sep()

    # bar baseline
    bars = make_bars("ASSET", 252)

    def run_bars(b=bars):
        eng = qc.BacktestEngine(100_000.0)
        eng.add_data("ASSET", b)
        eng.set_strategy(qc.BuyAndHold())
        eng.run()

    times    = timeit(run_bars, warmup=2, runs=20)
    p50      = median(times)
    print(fmt.format("Bar mode (252 daily bars)", "252",
                     f"{252 / p50:,.0f}", f"{p50 * 1000:.2f}"))

    # tick equivalents
    day_ns = 86_400_000_000_000
    for tpb in [1, 10, 390]:
        n_ticks      = 252 * tpb
        interval_ns  = day_ns // tpb
        ticks        = make_ticks("ASSET", n_ticks, interval_ns)

        def run_ticks(t=ticks):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("ASSET", t)
            eng.set_mm_refresh_interval(day_ns)  # refresh once per day
            eng.set_strategy(qc.BuyAndHold())
            eng.run()

        times = timeit(run_ticks, warmup=2, runs=10)
        p50   = median(times)
        label = f"Tick mode ({tpb} ticks/day, {n_ticks} total)"
        print(fmt.format(label, f"{n_ticks:,}",
                         f"{n_ticks / p50:,.0f}", f"{p50 * 1000:.2f}"))

    print()


# ============================================================================
# 5. Equity snapshot cost
# ============================================================================

def bench_snapshot_cost():
    section("5. Equity Snapshot Cost; 10K ticks")

    n     = 10_000
    ticks = make_ticks("ASSET", n, 1_000_000_000)  # 1s ticks

    fmt = f"  {{:<34}} {{:>10}}  {{:>14}}  {{:>10}}"
    print(fmt.format("Snapshot interval", "Equity pts", "Ticks/s", "p50 (ms)"))
    sep()

    configs = [
        ("Every tick (interval = 0)",  0),
        ("Every 1 minute",             60_000_000_000),
        ("Every 1 hour",            3_600_000_000_000),
    ]

    for label, interval in configs:
        def run(t=ticks, iv=interval):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("ASSET", t)
            eng.set_mm_refresh_interval(1_000_000_000)
            eng.set_equity_snapshot_interval(iv)
            eng.set_strategy(qc.BuyAndHold())
            eng.run()
            return len(eng.get_equity_curve())

        times      = timeit(run, warmup=2, runs=10)
        p50        = median(times)
        tps        = n / p50
        equity_pts = run()

        print(fmt.format(label, f"{equity_pts:,}",
                         f"{tps:,.0f}", f"{p50 * 1000:.2f}"))

    print()


# ============================================================================
# 6. numpy add_tick_data vs List[TickData]
# ============================================================================

def bench_numpy_add_tick_data():
    section("6. add_tick_data; numpy array vs List[TickData]")

    print("  Measures the cost of loading tick data into the engine before run().")
    print("  List[TickData] path: N individual pybind11 object crossings.")
    print("  numpy path         : 1 boundary crossing + C loop to construct TickSeries.")
    print()

    # verify numpy overload is available
    try:
        eng_test  = qc.BacktestEngine(100_000.0)
        arr_test  = make_ticks_numpy(10)
        eng_test.add_tick_data("TEST", arr_test)
    except TypeError:
        print("  SKIP: numpy add_tick_data overload not available.")
        print()
        return

    fmt = f"  {{:<34}} {{:>8}}  {{:>14}}  {{:>10}}  {{:>10}}"
    print(fmt.format("Path", "Ticks", "Ticks/s (add)", "p50 (ms)", "p99 (ms)"))
    sep()

    for n, label in [(1_000, "1K"), (10_000, "10K"), (60_000, "60K")]:
        ticks_list = make_ticks("SYM", n)
        ticks_np   = make_ticks_numpy(n)

        def add_list(tl=ticks_list):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("SYM", tl)

        def add_numpy(tn=ticks_np):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_tick_data("SYM", tn)

        t_list  = timeit(add_list,  warmup=3, runs=50)
        t_numpy = timeit(add_numpy, warmup=3, runs=50)

        p50_list  = pct(t_list,  50)
        p50_numpy = pct(t_numpy, 50)
        p99_list  = pct(t_list,  99)
        p99_numpy = pct(t_numpy, 99)

        speedup = p50_list / p50_numpy if p50_numpy > 0 else 0.0

        print(fmt.format(f"List[TickData]; {label} ticks", n,
                         f"{n / p50_list:>14,.0f}",
                         f"{p50_list * 1000:>10.2f}",
                         f"{p99_list * 1000:>10.2f}"))
        print(fmt.format(f"numpy (N,4); {label} ticks", n,
                         f"{n / p50_numpy:>14,.0f}",
                         f"{p50_numpy * 1000:>10.2f}",
                         f"{p99_numpy * 1000:>10.2f}"))
        print(f"  {'speedup':>34}  {speedup:>8.2f}x")
        print()


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    print()
    sep("=")
    print("  QuantCore; Tick Data Python Benchmarks")
    sep("=")
    print()

    bench_tick_throughput()
    bench_mm_throttle()
    bench_aggregation()
    bench_tick_vs_bar()
    bench_snapshot_cost()
    bench_numpy_add_tick_data()

    print("Done.")
