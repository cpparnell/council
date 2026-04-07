"""
Async LLM signal generation loop for backtesting.

Runs the real LLM council against historical data for each trading day in the
requested window, producing a signals DataFrame ready for portfolio simulation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.agents.base import AgentError
from src.agents.runner import run_council
from src.backtest.data import (
    NEUTRAL_NEWS_STUB,
    NEUTRAL_ONCHAIN_STUB,
    get_sentiment_for_date,
    slice_ohlcv,
)
from src.data.assembler import assemble_context
from src.data.indicators import compute_indicators
from src.data.price import build_price_data

logger = logging.getLogger(__name__)

# Required columns in the output DataFrame
SIGNAL_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "signal", "size_pct", "sl_price", "tp_price", "conviction", "vetoed",
]


# ---------------------------------------------------------- Portfolio tracker


@dataclass
class _PortfolioTracker:
    """Lightweight in-memory portfolio state for signal generation.

    Tracks cash and an open position so the council can make position-aware
    decisions without a database.
    """
    cash: float
    position_usd: float = 0.0
    peak: float = 0.0

    def __post_init__(self) -> None:
        self.peak = self.cash

    @property
    def total_value(self) -> float:
        return self.cash + self.position_usd

    @property
    def drawdown(self) -> float:
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - self.total_value) / self.peak)

    def open_position(self, size_usd: float) -> None:
        self.cash -= size_usd
        self.position_usd = size_usd
        self.peak = max(self.peak, self.total_value)

    def close_position(self, pnl_usd: float) -> None:
        self.cash += self.position_usd + pnl_usd
        self.position_usd = 0.0
        self.peak = max(self.peak, self.total_value)

    def to_dict(self) -> dict:
        """Return a PortfolioData-compatible dict."""
        return {
            "btc_position_usd": self.position_usd,
            "cash_usd": self.cash,
            "current_drawdown_pct": self.drawdown,
            "peak_portfolio_value": self.peak,
        }


# ---------------------------------------------------------- Signal generation


async def generate_signals(
    full_ohlcv: pd.DataFrame,
    sentiment_history: dict,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10_000.0,
    client=None,
) -> pd.DataFrame:
    """Run the LLM council for each trading day and return a signals DataFrame.

    For each day T in [start_date, end_date):
    1. Slice OHLCV to rows strictly before T (no lookahead).
    2. Retrieve the closest historical sentiment snapshot.
    3. Assemble a MarketContext using real OHLCV/indicators and stub news/onchain.
    4. Call run_council to get a signal.
    5. Record the day's row in the output DataFrame.
    6. Update the in-memory portfolio tracker.

    On AgentError for any cycle, logs a warning, records HOLD, and continues.

    Args:
        full_ohlcv: Full OHLCV DataFrame from fetch_full_ohlcv.
        sentiment_history: Output of fetch_historical_sentiment.
        start_date: First date to simulate (inclusive).
        end_date: Last date to simulate (exclusive).
        initial_capital: Starting portfolio value in USD.
        client: Optional AsyncAnthropic client; created from env if None.

    Returns:
        DataFrame indexed by date with columns: open, high, low, close, volume,
        signal, size_pct, sl_price, tp_price, conviction, vetoed.
    """
    tracker = _PortfolioTracker(cash=initial_capital)
    rows: list[dict] = []

    # Enumerate trading days in the range
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    while current < end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            # 1. Slice OHLCV — strict no-lookahead
            ohlcv_slice = slice_ohlcv(full_ohlcv, as_of=current)

            # 2. Get sentiment for this date
            sentiment = get_sentiment_for_date(sentiment_history, current)

            # 3. Build price and indicator data from the slice
            price_data = build_price_data(ohlcv_slice)
            indicator_data = compute_indicators(ohlcv_slice)

            # The last row of the slice is the "today" candle the agents see
            today_row = ohlcv_slice.iloc[-1]

            # 4. Assemble context — inject all historical data, skip live fetches
            ctx = await assemble_context(
                exchange=None,
                portfolio=tracker.to_dict(),
                timestamp=current,
                news_override=NEUTRAL_NEWS_STUB,
                sentiment_override=sentiment,
                onchain_override=NEUTRAL_ONCHAIN_STUB,
                skip_freshness=True,
            )

            # 5. Run council
            result = await run_council(ctx, client=client)

            signal = result.signal
            conviction = result.deliberation.conviction
            vetoed = result.vetoed
            size_pct = result.outputs.risk.approved_position_size_pct
            sl_price = result.outputs.risk.recommended_stop_loss
            tp_price = result.outputs.risk.recommended_take_profit

            logger.info(
                "Cycle %s: signal=%s conviction=%s vetoed=%s",
                date_str, signal, conviction, vetoed,
            )

        except (AgentError, ValueError) as exc:
            logger.warning("Cycle %s failed (%s: %s); recording HOLD.", date_str, type(exc).__name__, exc)
            signal = "HOLD"
            conviction = "low"
            vetoed = False
            size_pct = 0.0
            sl_price = 0.0
            tp_price = 0.0
            today_row = _get_today_row(full_ohlcv, current)

        # 6. Record row
        rows.append({
            "date": current,
            "open": float(today_row["open"]) if today_row is not None else 0.0,
            "high": float(today_row["high"]) if today_row is not None else 0.0,
            "low": float(today_row["low"]) if today_row is not None else 0.0,
            "close": float(today_row["close"]) if today_row is not None else 0.0,
            "volume": float(today_row["volume"]) if today_row is not None else 0.0,
            "signal": signal,
            "size_pct": size_pct,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "conviction": conviction,
            "vetoed": vetoed,
        })

        # 7. Update tracker (simplified: mark position open/close by signal)
        close_price = float(today_row["close"]) if today_row is not None else 0.0
        if signal == "BUY" and tracker.position_usd == 0.0 and size_pct > 0:
            size_usd = tracker.cash * (size_pct / 100)
            tracker.open_position(size_usd)
        elif signal == "SELL" and tracker.position_usd > 0:
            # Approximate PnL: we don't track entry price here — that's
            # backtesting.py's job. Just close the tracked position at par.
            tracker.close_position(pnl_usd=0.0)

        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["date"] + SIGNAL_COLUMNS).set_index("date")

    df = pd.DataFrame(rows).set_index("date")
    return df


def _get_today_row(full_df: pd.DataFrame, as_of: datetime):
    """Return the OHLCV row for as_of, or None if not found."""
    cutoff = pd.Timestamp(as_of, tz="UTC") if as_of.tzinfo is None else pd.Timestamp(as_of)
    # Look for rows strictly before the next day
    visible = full_df[full_df.index < cutoff]
    if visible.empty:
        return None
    return visible.iloc[-1]
