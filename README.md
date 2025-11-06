# QuantCore

High-performance backtesting engine for trading strategies in C++20.

## Overview

QuantCore is an event-driven backtester that simulates realistic order execution. It's built on top of my [orderbook simulator]([https://github.com/yourusername/orderbook](https://github.com/SLMolenaar/orderbook-simulator-cpp)) and processes events chronologically to avoid look-ahead bias.

The goal is a system where you can develop strategies in Python, backtest them against historical data with realistic execution simulation, and get comprehensive performance analysis with proper risk metrics. The C++ core handles the heavy lifting (event processing, order matching, position tracking), while Python provides an intuitive interface for research and visualization.

## Current Status

In progress ... 

## Example Strategies

- Buy and Hold (baseline)
- SMA Crossover (trend following)
- Mean Reversion (statistical arbitrage)

---

Building this as part of my quant finance portfolio.
