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

    bars = make_bars("ASSET", 1_260)
    data = {"ASSET": bars}

    param_grid = {
        "fast_period": [5, 10, 20, 50],
        "slow_period": [50, 100, 200],
    }
    raw_combos = len(param_grid["fast_period"]) * len(param_grid["slow_period"])
    cpu_count  = os.cpu_count() or 1

    print(f"  Strategy : SMACrossover")
    print(f"  Grid     : {len(param_grid['fast_period'])} fast × "
          f"{len(param_grid['slow_period'])} slow = {raw_combos} combos "
          f"(valid after fast<slow filter: ~9)")
    print(f"  Data     : 5 years (1 260 bars)")
    print(f"  CPUs     : {cpu_count}")
    print()

    hdr = f"  {'n_jobs':<12} {'Wall (s)':>10}  {'Speedup':>9}  {'Combos/s':>12}"
    print(hdr)
    sep()

    jobs_to_try = [1, 2, 4]
    if cpu_count > 4:
        jobs_to_try.append(-1)

    baseline: float | None = None

    for n_jobs in jobs_to_try:
        label = str(n_jobs) if n_jobs != -1 else f"-1 (all {cpu_count})"
        t0 = time.perf_counter()
        opt = GridSearchOptimizer(qc.SMACrossover, param_grid,
                                  metric="sharpe_ratio", n_jobs=n_jobs)
        results = opt.optimize(data, initial_capital=100_000.0, verbose=False)
        elapsed = time.perf_counter() - t0

        if baseline is None:
            baseline = elapsed

        speedup  = baseline / elapsed
        combos_s = len(results) / elapsed

        print(f"  {label:<12} {elapsed:>10.2f}  {speedup:>9.2f}x  {combos_s:>12.1f}")

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

    print("Done.")
