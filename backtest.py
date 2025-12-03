#!/usr/bin/env python3
"""
Simple Backtesting Engine
-------------------------

A tiny backtester for a single asset using a moving average crossover strategy.

Features:
- Generates synthetic price data OR loads from CSV (date,close).
- Simple moving-average crossover strategy (fast vs slow).
- Tracks PnL, number of trades, win rate, max drawdown.

Usage:
    python backtest.py
    python backtest.py --csv data/prices.csv --fast 20 --slow 50 --capital 10000
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Trade:
    entry_price: float
    exit_price: float
    profit: float


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    max_drawdown_pct: float


def generate_synthetic_prices(n: int = 500, start_price: float = 100.0) -> List[float]:
    """
    Generate a synthetic price series using a simple random walk.
    This avoids needing real market data for a quick demo.
    """
    prices = [start_price]
    for _ in range(1, n):
        # Simulate a small daily return between -2% and +2%
        daily_return = random.uniform(-0.02, 0.02)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    return prices


def load_prices_from_csv(path: str) -> List[float]:
    """
    Load closing prices from a CSV file with columns: date, close.
    Extra columns are ignored.
    """
    prices: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "close" not in reader.fieldnames:
            raise ValueError("CSV must have a 'close' column")
        for row in reader:
            prices.append(float(row["close"]))
    if not prices:
        raise ValueError("No prices found in CSV")
    return prices


def sma(values: List[float], window: int) -> List[Optional[float]]:
    """
    Compute simple moving average over a list.
    Returns list of same length; first (window-1) values are None.
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    out: List[Optional[float]] = [None] * len(values)
    if len(values) < window:
        return out

    window_sum = sum(values[:window])
    out[window - 1] = window_sum / window

    for i in range(window, len(values)):
        window_sum += values[i] - values[i - window]
        out[i] = window_sum / window

    return out


def run_backtest(
    prices: List[float],
    fast_window: int,
    slow_window: int,
    initial_capital: float,
    fee_bps: float = 1.0,  # basis points (1 bps = 0.01%)
) -> BacktestResult:
    """
    Simple long-only MA crossover backtest.

    Rules:
    - If fast MA > slow MA and we are flat → buy as much as possible.
    - If fast MA < slow MA and we are long → sell everything.
    - No short selling, no position sizing sophistication.
    """

    if slow_window <= fast_window:
        raise ValueError("slow_window should be > fast_window")

    fast = sma(prices, fast_window)
    slow = sma(prices, slow_window)

    cash = initial_capital
    position = 0.0  # number of shares
    equity_curve: List[float] = []
    open_trade_price: Optional[float] = None
    trades: List[Trade] = []

    for i, price in enumerate(prices):
        # Compute current equity
        equity = cash + position * price
        equity_curve.append(equity)

        f = fast[i]
        s = slow[i]

        # We need both MAs to be available (not None)
        if f is None or s is None:
            continue

        # Entry condition: fast > slow and currently flat
        if f > s and position == 0.0:
            # Buy with all cash
            # Apply fee for buying
            fee = cash * (fee_bps / 10000.0)
            cash_after_fee = cash - fee
            position = cash_after_fee / price
            cash = 0.0
            open_trade_price = price

        # Exit condition: fast < slow and currently long
        elif f < s and position > 0.0:
            # Sell everything
            gross_proceeds = position * price
            fee = gross_proceeds * (fee_bps / 10000.0)
            net_proceeds = gross_proceeds - fee
            cash = net_proceeds
            if open_trade_price is not None:
                profit = (price - open_trade_price) * position - fee
                trades.append(Trade(open_trade_price, price, profit))
            position = 0.0
            open_trade_price = None

    # Liquidate at end if still in a position
    final_price = prices[-1]
    if position > 0.0:
        gross_proceeds = position * final_price
        fee = gross_proceeds * (fee_bps / 10000.0)
        net_proceeds = gross_proceeds - fee
        cash = net_proceeds
        if open_trade_price is not None:
            profit = (final_price - open_trade_price) * position - fee
            trades.append(Trade(open_trade_price, final_price, profit))
        position = 0.0

    final_equity = cash
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0

    # Win rate
    num_trades = len(trades)
    wins = sum(1 for t in trades if t.profit > 0)
    win_rate_pct = (wins / num_trades * 100.0) if num_trades > 0 else 0.0

    # Max drawdown
    max_equity = -math.inf
    max_drawdown_pct = 0.0
    for eq in equity_curve:
        max_equity = max(max_equity, eq)
        if max_equity > 0:
            drawdown = (eq / max_equity - 1.0) * 100.0
            max_drawdown_pct = min(max_drawdown_pct, drawdown)

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MA crossover backtesting engine")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to CSV file with columns date,close")
    parser.add_argument("--fast", type=int, default=20, help="Fast MA window")
    parser.add_argument("--slow", type=int, default=50, help="Slow MA window")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Initial capital for the backtest")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic prices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.csv and os.path.exists(args.csv):
        prices = load_prices_from_csv(args.csv)
        print(f"Loaded {len(prices)} prices from {args.csv}")
    else:
        prices = generate_synthetic_prices()
        print(f"No CSV provided or file not found. Generated {len(prices)} synthetic prices.")

    result = run_backtest(
        prices=prices,
        fast_window=args.fast,
        slow_window=args.slow,
        initial_capital=args.capital,
    )

    print("\n=== Backtest Summary ===")
    print(f"Initial capital : {result.initial_capital:,.2f}")
    print(f"Final equity    : {result.final_equity:,.2f}")
    print(f"Total return    : {result.total_return_pct:.2f}%")
    print(f"Number of trades: {result.num_trades}")
    print(f"Win rate        : {result.win_rate_pct:.2f}%")
    print(f"Max drawdown    : {result.max_drawdown_pct:.2f}%")

    print("\nTip: plug in real CSV market data to test real strategies later.")


if __name__ == "__main__":
    main()
