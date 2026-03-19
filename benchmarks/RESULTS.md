# Benchmark Results

Measured on Windows (Release build, MSVC `/O2`). C++ numbers from `bench_backtest_engine.exe`
and `bench_tick_data.exe`. Python numbers from `bench_python.py` and `bench_tick_python.py`.

---

## C++ Engine -- Throughput (single asset, BuyAndHold)

| Scenario       | Bars    | Events/s  | Bars/s    | Wall time  |
|----------------|---------|-----------|-----------|------------|
| 1-year         | 252     | 0.4 M/s   | 207 K/s   | 1.22 ms    |
| 5-year         | 1,260   | 0.5 M/s   | 263 K/s   | 4.78 ms    |
| 10-year        | 2,520   | 0.5 M/s   | 273 K/s   | 9.25 ms    |
| 50-year        | 12,600  | 0.6 M/s   | 277 K/s   | 45.5 ms    |
| 250-year       | 63,000  | 0.5 M/s   | 268 K/s   | 234.9 ms   |
| 1,000-year     | 252,000 | 0.6 M/s   | 290 K/s   | 869.7 ms   |

---

## C++ Engine -- Strategy Comparison (5 years, 1 asset)

| Strategy              | Bars  | Events/s  | Bars/s    | Wall time |
|-----------------------|-------|-----------|-----------|-----------|
| BuyAndHold            | 1,260 | 0.6 M/s   | 286 K/s   | 4.40 ms   |
| SMACrossover(20/100)  | 1,260 | 0.5 M/s   | 270 K/s   | 4.67 ms   |
| MeanReversion(20,1.5) | 1,260 | 0.5 M/s   | 266 K/s   | 4.74 ms   |

---

## C++ Engine -- Multi-Asset Scaling (5 years, MeanReversion)

| Symbols | Bars   | Events/s  | Bars/s   | Wall time  |
|---------|--------|-----------|----------|------------|
| 1       | 1,260  | 0.5 M/s   | 268 K/s  | 4.69 ms    |
| 2       | 2,520  | 0.4 M/s   | 206 K/s  | 12.2 ms    |
| 5       | 6,300  | 0.3 M/s   | 139 K/s  | 45.4 ms    |
| 10      | 12,600 | 0.1 M/s   | 60 K/s   | 209.6 ms   |

Multi-asset throughput drops as symbol count increases because each symbol has its own execution
engine and triggers a portfolio update per bar. The bottleneck shifts from the matching engine to
the per-symbol portfolio loop.

---

## Order Book -- Raw Operations

| Pattern                      | Ops/s      | Detail                      |
|------------------------------|------------|-----------------------------|
| Add + cancel (MM refresh)    | 13.0 M/s   | 500,000 pairs, 76.9 ms      |
| Add + match  (taker sweep)   | 4.9 M/s    | 300,000 pairs, 122.2 ms     |

---

## C++ Engine -- Latency Distribution (1-year, 100 samples)

| Strategy              | p50      | p95      | p99      | max      |
|-----------------------|----------|----------|----------|----------|
| BuyAndHold            | 0.851 ms | 0.933 ms | 1.089 ms | 1.089 ms |
| SMACrossover(20/100)  | 0.830 ms | 0.867 ms | 0.928 ms | 0.928 ms |
| MeanReversion(20,1.5) | 0.884 ms | 0.922 ms | 0.981 ms | 0.981 ms |

All strategies pass the `< 5 ms` target for a 1-year backtest.

---

## C++ Engine -- Tick Data Throughput

### Engine throughput (BuyAndHold, 1-second ticks, no MM throttle)

| Ticks     | Ticks/s   | Wall time  |
|-----------|-----------|------------|
| 1,000     | 334 K/s   | 3.0 ms     |
| 10,000    | 315 K/s   | 31.7 ms    |
| 60,000    | 325 K/s   | 184.7 ms   |
| 360,000   | 314 K/s   | 1,147 ms   |
| 1,000,000 | 310 K/s   | 3,225 ms   |

Without throttling the market maker refreshes its quotes on every tick, which dominates cost.

### MM throttle -- key lever for tick performance

| MM refresh interval      | Ticks/s  | Wall time (10K ticks) | vs no throttle |
|--------------------------|---------|-----------------------|----------------|
| No throttle (every tick) | 340 K/s  | 29.4 ms               | 1.0x           |
| 100ms                    | 338 K/s  | 29.6 ms               | 1.0x           |
| 1s                       | 1.66 M/s | 6.0 ms                | 4.9x           |
| 10s                      | 2.78 M/s | 3.6 ms                | 8.2x           |

The 100ms interval has no effect here because the test ticks are 1 second apart. At sub-second
tick data the gap between 100ms and no-throttle would be significant.

### Tick vs bar -- equivalent dataset (252-day period, BuyAndHold)

MM refresh set to once per day so the comparison is fair.

| Mode                              | Events  | Events/s   | Wall time |
|-----------------------------------|---------|------------|-----------|
| Bar mode (252 daily bars)         | 252     | 355 K/s    | 0.7 ms    |
| Tick mode (1 tick/day, 252 total) | 252     | 348 K/s    | 0.7 ms    |
| Tick mode (10 ticks/day, 2,520)   | 2,520   | 1,749 K/s  | 1.4 ms    |
| Tick mode (390 ticks/day, 98,280) | 98,280  | 2,077 K/s  | 47.3 ms   |

Per-event cost in tick mode is lower than bar mode once the MM is throttled, because ticks are
lighter events with no OHLCV unpacking.

### Aggregation -- aggregate_to_bars (1M ticks)

| Bar duration    | Bars out  | Ticks/s   | Wall time |
|-----------------|-----------|-----------|-----------|
| 1-second bars   | 1,000,000 | 23 M/s    | 43.3 ms   |
| 1-minute bars   | 16,667    | 247 M/s   | 4.1 ms    |
| 1-hour bars     | 278       | 368 M/s   | 2.7 ms    |
| 1-day bars      | 12        | 366 M/s   | 2.7 ms    |

Throughput is dominated by output volume. 1-second bars produce 1M `BarData` objects and are
~10x slower than 1-minute bars. Aggregation is not the bottleneck in practice.

### Equity snapshot cost (10K ticks, 1s MM throttle)

Snapshot frequency has negligible impact on throughput.

| Snapshot interval | Equity pts | Ticks/s |
|-------------------|------------|---------|
| Every tick        | 10,001     | 334 K/s |
| Every 1 minute    | 168        | 338 K/s |
| Every 1 hour      | 4          | 342 K/s |

---

## Target Verification

| Target                                    | Result                            |
|-------------------------------------------|-----------------------------------|
| 1-year backtest p99 < 5 ms                | PASS -- 0.94 ms                   |
| Order book > 500 K ops/s (MM pattern)     | PASS -- 13.0 M ops/s              |
| Memory growth < 10 MB (1000-year run)     | SKIP -- not measurable on Windows |
| 10K ticks (1s MM throttle) p50 < 50ms    | PASS -- 29.5 ms                   |
| 1M ticks aggregated to 1-min bars < 100ms | PASS -- 3.8 ms                   |

---

## Python Layer -- Bindings Throughput

| Scenario                     | Bars   | Bars/s    | p50      | p99      |
|------------------------------|--------|-----------|----------|----------|
| BuyAndHold -- 1 yr           | 252    | 183,313   | 1.37 ms  | 1.47 ms  |
| BuyAndHold -- 5 yr           | 1,260  | 255,438   | 4.93 ms  | 5.19 ms  |
| BuyAndHold -- 10 yr          | 2,520  | 255,454   | 9.86 ms  | 10.90 ms |
| SMACrossover(20/100) -- 5yr  | 1,260  | 244,917   | 5.14 ms  | 5.32 ms  |
| MeanReversion(20,1.5) -- 5yr | 1,260  | 236,580   | 5.33 ms  | 5.78 ms  |
| BuyAndHold -- 10 symbols     | 12,600 | 81,706    | 154.2 ms | 157.0 ms |

Python overhead is minimal for single-asset workloads. The 10-symbol case is slower due to
pybind11 boundary crossings per symbol; use the numpy `add_data` path to reduce this.

---

## Python Layer -- Tick Data Throughput

| Scenario                               | Ticks  | Ticks/s   | p50      |
|----------------------------------------|--------|-----------|----------|
| BuyAndHold -- 1K ticks, no throttle    | 1,000  | 331 K/s   | 3.0 ms   |
| BuyAndHold -- 10K ticks, no throttle   | 10,000 | 308 K/s   | 32.5 ms  |
| BuyAndHold -- 10K ticks, 1s throttle   | 10,000 | 1.35 M/s  | 7.4 ms   |
| BuyAndHold -- 10K ticks, 10s throttle  | 10,000 | 2.05 M/s  | 4.9 ms   |

---

## Python Layer -- numpy vs List[BarData] for `add_data`

| Path                       | Bars   | Bars/s (add_data) | p50      | Speedup |
|----------------------------|--------|-------------------|----------|---------|
| List[BarData] -- 1yr       | 252    | 13.8 M/s          | 0.02 ms  | -       |
| numpy (N,6) -- 1yr         | 252    | 32.3 M/s          | 0.01 ms  | 2.33x   |
| List[BarData] -- 5yr       | 1,260  | 15.6 M/s          | 0.08 ms  | -       |
| numpy (N,6) -- 5yr         | 1,260  | 56.8 M/s          | 0.02 ms  | 3.64x   |
| List[BarData] -- 10yr      | 2,520  | 15.7 M/s          | 0.16 ms  | -       |
| numpy (N,6) -- 10yr        | 2,520  | 62.8 M/s          | 0.04 ms  | 4.01x   |
| List[BarData] -- 10sym×5yr | 12,600 | 15.8 M/s          | 0.80 ms  | -       |
| numpy (N,6) -- 10sym×5yr   | 12,600 | 70.9 M/s          | 0.18 ms  | 4.48x   |

---

## Python Layer -- numpy vs List[TickData] for `add_tick_data`

| Path                        | Ticks  | Ticks/s (add_tick_data) | p50      | Speedup |
|-----------------------------|--------|-------------------------|----------|---------|
| List[TickData] -- 1K ticks  | 1,000  | 12 M/s                  | 0.08 ms  | -       |
| numpy (N,4) -- 1K ticks     | 1,000  | 33 M/s                  | 0.03 ms  | 2.71x   |
| List[TickData] -- 10K ticks | 10,000 | 16 M/s                  | 0.64 ms  | -       |
| numpy (N,4) -- 10K ticks    | 10,000 | 74 M/s                  | 0.13 ms  | 4.75x   |
| List[TickData] -- 60K ticks | 60,000 | 9 M/s                   | 6.42 ms  | -       |
| numpy (N,4) -- 60K ticks    | 60,000 | 32 M/s                  | 1.85 ms  | 3.47x   |

For large tick datasets always prefer the numpy path to `add_tick_data`. The `(N, 4)` array
uses a single boundary crossing plus a C-side construction loop, vs. N individual pybind11
object constructions.

---

## Python Layer -- Grid Search (SMACrossover, 109 valid combos)

### 5-year per combo (~11 ms each) -- below Windows spawn break-even

| n_jobs         | Wall time | Speedup | Combos/s |
|----------------|-----------|---------|----------|
| 1 (sequential) | 0.75 s    | 1.00x   | 145.6    |
| 4              | 1.05 s    | 0.72x   | 104.3    |
| -1 (16 cores)  | 1.58 s    | 0.47x   | 69.0     |

### 20-year per combo (~42 ms each) -- marginal

| n_jobs         | Wall time | Speedup | Combos/s |
|----------------|-----------|---------|----------|
| 1 (sequential) | 3.09 s    | 1.00x   | 35.2     |
| 4              | 1.69 s    | 1.83x   | 64.4     |
| -1 (16 cores)  | 1.95 s    | 1.58x   | 55.8     |

### 100-year per combo (~240 ms each) -- clear speedup

| n_jobs         | Wall time | Speedup | Combos/s |
|----------------|-----------|---------|----------|
| 1 (sequential) | 16.12 s   | 1.00x   | 6.8      |
| 4              | 5.56 s    | 2.90x   | 19.6     |
| -1 (16 cores)  | 4.00 s    | 4.03x   | 27.2     |

On Windows, `multiprocessing` uses `spawn`, which costs roughly 500 ms per worker for module
re-import. Parallelism only pays off when individual combo runtime × (combos / workers)
substantially exceeds that overhead. On Linux (`fork`-based) the break-even point is much lower.

---

## CSV Loading Throughput

| Dataset           | Bars   | Bars/s    | p50 (ms) |
|-------------------|--------|-----------|----------|
| 1-year (252)      | 252    | 458,933   | 0.55     |
| 5-year (1,260)    | 1,260  | 503,135   | 2.50     |
| 20-year (5,040)   | 5,040  | 471,072   | 10.70    |
| 100-year (25,200) | 25,200 | 472,113   | 53.38    |

Loading throughput is stable across dataset sizes.

---

## Reproducing

```bash
# C++ benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_backtest_engine bench_tick_data
./build/bench_backtest_engine
./build/bench_tick_data

# Python benchmarks (requires built bindings)
python python/build_module.py
python benchmarks/bench_python.py
python benchmarks/bench_tick_python.py
```