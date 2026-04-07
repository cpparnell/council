"""
Backtest engine — portfolio simulation and CLI runner.

Uses the backtesting.py library to simulate trading based on pre-computed
LLM council signals.

Usage:
    python -m src.backtest.engine \\
        --start 2024-01-01 \\
        --end   2025-01-01 \\
        [--capital 10000] \\
        [--no-save-csv]
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

from src.backtest.data import fetch_full_ohlcv, fetch_historical_sentiment
from src.backtest.signals import generate_signals

logger = logging.getLogger(__name__)


# ----------------------------------------------------------- Strategy


class LLMCouncilStrategy(Strategy):
    """backtesting.py Strategy that replays pre-computed LLM council signals.

    Reads signal/size_pct/sl_price/tp_price columns from the OHLCV+signals
    DataFrame on each bar and executes accordingly:
      - BUY  (no open position): buy with stop-loss (and take-profit if > 0)
      - SELL (open position):    close position
      - HOLD: do nothing
    """

    def init(self):
        pass

    def next(self):
        signal = self.data.signal[-1]
        size_pct = self.data.size_pct[-1]
        sl_pct = self.data.sl_pct[-1]
        tp_pct = self.data.tp_pct[-1]

        if signal == "BUY" and not self.position:
            if sl_pct > 0 and size_pct > 0:
                current_close = self.data.Close[-1]
                self.buy(
                    size=size_pct / 100,
                    sl=current_close * sl_pct,
                    tp=current_close * tp_pct if tp_pct > 0 else None,
                )
        elif signal == "SELL" and self.position:
            self.position.close()


# ----------------------------------------------------------- Runner


async def run_backtest(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10_000.0,
    client=None,
    save_signals_csv: bool = True,
) -> dict:
    """Run a full LLM council backtest over the specified date range.

    Steps:
    1. Fetch full OHLCV history from Binance mainnet (read-only, no orders).
    2. Fetch historical Fear & Greed sentiment.
    3. Run the LLM council for each trading day (generate_signals).
    4. Simulate portfolio execution with backtesting.py.
    5. Optionally save the signals DataFrame to a CSV.

    Args:
        start_date: First day of the backtest (inclusive).
        end_date:   Last day of the backtest (exclusive).
        initial_capital: Starting portfolio value in USD.
        client: Optional AsyncAnthropic client; created from env if None.
        save_signals_csv: When True, writes backtest_signals_YYYYMMDD.csv.

    Returns:
        A dict of formatted backtest statistics.
    """
    logger.info("Fetching OHLCV history %s → %s", start_date.date(), end_date.date())
    full_ohlcv = fetch_full_ohlcv(start_date, end_date)

    logger.info("Fetching historical Fear & Greed sentiment")
    sentiment_history = fetch_historical_sentiment()

    logger.info("Generating signals (LLM council)…")
    signals_df = await generate_signals(
        full_ohlcv=full_ohlcv,
        sentiment_history=sentiment_history,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        client=client,
    )

    if signals_df.empty:
        raise ValueError("No signals generated — cannot run backtest.")

    # backtesting.py requires OHLCV columns to be title-cased
    bt_df = signals_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    bt = FractionalBacktest(
        bt_df,
        LLMCouncilStrategy,
        cash=initial_capital,
        commission=0.001,  # 0.1% per side (Kraken spot approximation)
        exclusive_orders=True,
    )
    stats = bt.run()

    if save_signals_csv:
        now = datetime.now(timezone.utc)
        csv_path = f"tmp/backtest_signals_{now}.csv"
        signals_df.to_csv(csv_path)
        logger.info("Signals saved to %s", csv_path)

    n_cycles = len(signals_df)
    n_trades = int(stats.get("# Trades", 0))

    formatted = {
        "start": start_date.date().isoformat(),
        "end": end_date.date().isoformat(),
        "cycles": n_cycles,
        "trades": n_trades,
        "return_pct": float(stats.get("Return [%]", 0.0)),
        "sharpe_ratio": float(stats.get("Sharpe Ratio", float("nan"))),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        "avg_trade_pct": float(stats.get("Avg. Trade [%]", 0.0)),
        "expectancy_usd": float(stats.get("Expectancy [%]", 0.0)) * initial_capital / 100,
        "csv_path": f"backtest_signals_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv" if save_signals_csv else None,
        "_raw_stats": stats,
    }
    return formatted


def _print_results(result: dict) -> None:
    n_cycles = result["cycles"]
    n_trades = result["trades"]
    print(f"\nBacktest: {result['start']} → {result['end']}  ({n_cycles} cycles, {n_trades} trades)")
    print("─" * 38)
    print(f"Return:          {result['return_pct']:+.1f}%")
    print(f"Sharpe Ratio:     {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {result['max_drawdown_pct']:.1f}%")
    print(f"Win Rate:         {result['win_rate_pct']:.1f}%")
    print(f"Avg Trade:        {result['avg_trade_pct']:+.1f}%")
    print(f"Expectancy:      ${result['expectancy_usd']:+.0f} / trade")
    print("─" * 38)
    if result.get("csv_path"):
        print(f"Signals saved to {result['csv_path']}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run LLM council backtest over a historical date range."
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date (inclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", required=True,
        help="End date (exclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Initial capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--no-save-csv", dest="save_csv", action="store_false", default=True,
        help="Skip saving the signals CSV",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    result = asyncio.run(
        run_backtest(
            start_date=start,
            end_date=end,
            initial_capital=args.capital,
            save_signals_csv=args.save_csv,
        )
    )
    _print_results(result)
