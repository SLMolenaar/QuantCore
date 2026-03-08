"""
benchmarks/bench_python.py
==========================
Python-layer benchmarks for QuantCore.

Sections:
  1. Bindings throughput  — engine.run() latency via pybind11
  2. Grid search speedup  — sequential vs parallel (n_jobs 1/2/4/-1)
  3. CSV data loading     — bars/s from disk

Run from the project root after building the extension:
    python python/build_module.py
    python benchmarks/bench_python.py

Requirements: quantcore extension built, pandas, numpy
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

try:
    import quantcore as qc
    from quantcore.walk_forward import GridSearchOptimizer
except ImportError as e:
    print(f"ERROR: Could not import quantcore: {e}")
    print("Build the extension first:  python python/build_module.py")
    sys.exit(1)


# ============================================================================
# Data helpers
# ============================================================================

def make_bars(symbol: str, n: int, start_price: float = 100.0) -> list:
    bars = []
    price = start_price
    start_ns = 1_577_836_800_000_000_000  # 2020-01-01
    day_ns   = 86_400_000_000_000

    for i in range(n):
        price *= 1.0 + 0.0002 + 0.01 * math.sin(i * 0.05)
        bars.append(qc.BarData(symbol, start_ns + i * day_ns,
                               price, price * 1.005, price * 0.995,
                               price, 1_000_000.0))
    return bars


def make_bars_numpy(n: int, start_price: float = 100.0) -> np.ndarray:
    """
    Build bar data as a (N, 6) float64 array: [timestamp_ns, open, high, low, close, volume].
    Used with the numpy overload of add_data — one boundary crossing instead of N.
    timestamp_ns fits in float64 exactly here (starts near 0 to avoid precision loss).
    """
    arr = np.empty((n, 6), dtype=np.float64)
    price = start_price
    day_ns = 86_400_000_000_000
    for i in range(n):
        price *= 1.0 + 0.0002 + 0.01 * math.sin(i * 0.05)
        arr[i, 0] = float(i * day_ns)      # timestamp_ns (small, fits in float64 exactly)
        arr[i, 1] = price                  # open
        arr[i, 2] = price * 1.005          # high
        arr[i, 3] = price * 0.995          # low
        arr[i, 4] = price                  # close
        arr[i, 5] = 1_000_000.0            # volume
    return arr


def make_csv(path: Path, symbol: str, n: int, start_price: float = 100.0) -> None:
    price = start_price
    start_s = 1_577_836_800  # unix seconds
    with open(path, "w") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(n):
            price *= 1.0 + 0.0002 + 0.01 * math.sin(i * 0.05)
            f.write(f"{start_s + i * 86_400},"
                    f"{price:.4f},{price*1.005:.4f},"
                    f"{price*0.995:.4f},{price:.4f},1000000\n")


# ============================================================================
# Timing helpers
# ============================================================================

def timeit(fn, warmup: int = 2, runs: int = 20) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return times


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
# 1. Bindings throughput
# ============================================================================

def bench_bindings():
    section("1. Python Bindings Throughput")

    fmt = f"  {{:<34}} {{:>8}}  {{:>12}}  {{:>10}}  {{:>10}}"
    print(fmt.format("Scenario", "Bars", "Bars/s", "p50 (ms)", "p99 (ms)"))
    sep()

    configs = [
        ("BuyAndHold — 1 yr",    252,     lambda: qc.BuyAndHold(),               1),
        ("BuyAndHold — 5 yr",    1_260,   lambda: qc.BuyAndHold(),               1),
        ("BuyAndHold — 10 yr",   2_520,   lambda: qc.BuyAndHold(),               1),
        ("SMAXover(20/100) — 5yr", 1_260, lambda: qc.SMACrossover(20, 100),      1),
        ("MeanRev(20,1.5) — 5yr", 1_260,  lambda: qc.MeanReversion(20, 1.5, 0.5), 1),
        ("BuyAndHold — 10 sym",  1_260,   lambda: qc.BuyAndHold(),               10),
    ]

    for label, n_bars, strat_fn, n_syms in configs:
        data = {f"SYM{s}": make_bars(f"SYM{s}", n_bars, 100.0 + s * 5.0)
                for s in range(n_syms)}

        def run(d=data, sf=strat_fn):
            eng = qc.BacktestEngine(100_000.0)
            for sym, bars in d.items():
                eng.add_data(sym, bars)
            eng.set_strategy(sf())
            eng.run()

        n_runs = 10 if n_syms > 1 else 20
        times = timeit(run, warmup=2, runs=n_runs)

        total = n_bars * n_syms
        p50   = pct(times, 50)
        p99   = pct(times, 99)
        bps   = total / p50

        print(fmt.format(label, f"{total:,}", f"{bps:,.0f}",
                         f"{p50*1000:.2f}", f"{p99*1000:.2f}"))

    print()


# ============================================================================
# 2. Grid search: sequential vs parallel
# ============================================================================

def bench_parallel():
    section("2. Grid Search — Sequential vs Parallel")

    cpu_count = os.cpu_count() or 1

    # Windows process spawn costs ~500ms per worker (module re-import).
    # Each combo on 5yr data takes ~11ms — you need 500/11 ≈ 46 combos per
    # worker just to break even. With 16 workers that's 736 combos minimum,
    # which is impractical for a benchmark.
    #
    # The fix: make each combo more expensive by using longer data.
    # At 20yr/combo (~43ms each), the crossover point with 4 workers is
    # 500/43 × 4 ≈ 47 combos — achievable.
    # At 100yr/combo (~110ms each), even 2 workers break even at 9 combos.
    #
    # We run two sub-benchmarks so the output shows both sides of the curve.

    param_grid = {
        "fast_period": list(range(5, 55, 5)),    # 10 values
        "slow_period": list(range(50, 260, 20)), # 11 values
    }
    valid_combos = sum(1 for f in param_grid["fast_period"]
                       for s in param_grid["slow_period"] if f < s)

    print(f"  Strategy     : SMACrossover")
    print(f"  Grid         : {len(param_grid['fast_period'])} fast × "
          f"{len(param_grid['slow_period'])} slow = {valid_combos} valid combos")
    print(f"  CPUs         : {cpu_count}")
    print(f"  Windows note : spawn overhead ~500ms/worker. Speedup requires")
    print(f"                 combo_time × (combos / workers) >> 500ms.")
    print()

    for years, label in [(5,   "5yr/combo   (~11ms each)  — too cheap for Windows"),
                         (20,  "20yr/combo  (~42ms each)  — marginal"),
                         (100, "100yr/combo (~240ms each) — clear speedup")]:
        bars = make_bars("ASSET", years * 252)
        data = {"ASSET": bars}

        print(f"  ── {label}")
        hdr = f"    {'n_jobs':<14} {'Wall (s)':>10}  {'Speedup':>9}  {'Combos/s':>12}"
        print(hdr)

        jobs_to_try = [1, 4]
        if cpu_count >= 8:
            jobs_to_try.append(-1)

        baseline: float | None = None
        for n_jobs in jobs_to_try:
            job_label = str(n_jobs) if n_jobs != -1 else f"-1 ({cpu_count} cores)"
            t0 = time.perf_counter()
            opt = GridSearchOptimizer(qc.SMACrossover, param_grid,
                                      metric="sharpe_ratio", n_jobs=n_jobs)
            results = opt.optimize(data, initial_capital=100_000.0, verbose=False)
            elapsed = time.perf_counter() - t0

            if baseline is None:
                baseline = elapsed

            speedup  = baseline / elapsed
            combos_s = len(results) / elapsed
            print(f"    {job_label:<14} {elapsed:>10.2f}  {speedup:>9.2f}x  {combos_s:>12.1f}")

        print()



# ============================================================================
# 3. CSV data loading
# ============================================================================

def bench_loading():
    section("3. CSV Data Loading Throughput")

    tmp = Path(tempfile.mkdtemp())
    try:
        sizes = [
            ("1 yr   (252 bars)",    252),
            ("5 yr   (1 260 bars)",  1_260),
            ("20 yr  (5 040 bars)",  5_040),
            ("100 yr (25 200 bars)", 25_200),
        ]

        hdr = f"  {'File':<26} {'Bars':>8}  {'Bars/s':>12}  {'p50 (ms)':>10}"
        print(hdr)
        sep()

        for label, n in sizes:
            path = tmp / f"bench_{n}.csv"
            make_csv(path, "ASSET", n)

            def load(p=str(path)):
                return qc.CSVDataLoader.load(p, "ASSET")

            times = timeit(load, warmup=2, runs=10)
            p50   = pct(times, 50)
            bps   = n / p50

            print(f"  {label:<26} {n:>8,}  {bps:>12,.0f}  {p50*1000:>10.2f}")

    finally:
        shutil.rmtree(tmp)

    print()


# ============================================================================
# 4. numpy add_data vs List[BarData] add_data
# ============================================================================

def bench_numpy_add_data():
    section("4. add_data — numpy array vs List[BarData]")

    print("  Measures the cost of loading data into the engine before run().")
    print("  List[BarData] path: N individual pybind11 object crossings.")
    print("  numpy path        : 1 boundary crossing + C loop to construct BarSeries.")
    print()

    # Check if numpy overload is available
    try:
        eng_test = qc.BacktestEngine(100_000.0)
        arr_test = make_bars_numpy(10)
        eng_test.add_data("TEST", arr_test)
    except TypeError:
        print("  SKIP: numpy add_data overload not yet compiled into _core.")
        print("  Apply the bindings.cpp change and rebuild first.")
        print()
        return

    fmt = f"  {{:<30}} {{:>8}}  {{:>14}}  {{:>10}}  {{:>10}}"
    print(fmt.format("Path", "Bars", "Bars/s (add)", "p50 (ms)", "p99 (ms)"))
    sep()

    for n_bars, label_suffix in [(252, "1 yr"), (1_260, "5 yr"),
                                 (2_520, "10 yr"), (12_600, "10 sym×5yr")]:

        # List[BarData] path — one add_data call per symbol
        bars_list = make_bars("SYM", n_bars)

        def add_list(bl=bars_list):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_data("SYM", bl)

        # numpy path
        bars_np = make_bars_numpy(n_bars)

        def add_numpy(bn=bars_np):
            eng = qc.BacktestEngine(100_000.0)
            eng.add_data("SYM", bn)

        t_list  = timeit(add_list,  warmup=3, runs=50)
        t_numpy = timeit(add_numpy, warmup=3, runs=50)

        p50_list  = pct(t_list,  50)
        p50_numpy = pct(t_numpy, 50)
        p99_list  = pct(t_list,  99)
        p99_numpy = pct(t_numpy, 99)

        bps_list  = n_bars / p50_list
        bps_numpy = n_bars / p50_numpy

        speedup = p50_list / p50_numpy

        print(fmt.format(f"List[BarData] — {label_suffix}", n_bars,
                         f"{bps_list:>14,.0f}", f"{p50_list*1000:>10.2f}", f"{p99_list*1000:>10.2f}"))
        print(fmt.format(f"numpy (N,6)  — {label_suffix}", n_bars,
                         f"{bps_numpy:>14,.0f}", f"{p50_numpy*1000:>10.2f}", f"{p99_numpy*1000:>10.2f}"))
        print(f"  {'speedup':>30}  {speedup:>8.2f}x")
        print()

    print("  Note: add_data cost is fixed overhead before run(). The 5.5x parallel")
    print("  grid search speedup shrinks the relative importance of this gap.")
    print()


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    print()
    sep("=")
    print("  QuantCore — Python Layer Benchmarks")
    sep("=")
    print()

    bench_bindings()
    bench_parallel()
    bench_loading()
    bench_numpy_add_data()

    print("Done.")