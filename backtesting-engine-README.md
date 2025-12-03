# Backtesting Engine (Moving Average Crossover)
A simple single-asset backtesting engine written in Python.
It uses a moving-average crossover strategy (fast vs slow) on a price series and reports basic performance metrics.

## Features
- Uses synthetic prices by default (no external data required)
- Optionally loads market data from CSV (date, close)
- Implements a long-only moving average crossover strategy
- Reports:
  - Total return
  - Number of trades
  - Win rate
  - Max drawdown

## Requirements
- Python 3.9+

## Usage
### Synthetic data
python backtest.py

### CSV data
python backtest.py --csv data/prices.csv --fast 20 --slow 50 --capital 10000

### Extensions
- Add RSI or breakout strategies
- Add multi-asset support
- Add slippage and execution models
