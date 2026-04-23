"""
Async LLM signal generation loop for backtesting.

Runs the real LLM council against historical data for each trading day in the
requested window, producing a signals DataFrame ready for portfolio simulation.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.agents.base import AgentError
from src.agents.runner import run_council
from src.agents.scoring import score_to_position_size_pct
from src.backtest.data import (
    NEUTRAL_NEWS_STUB,
    get_sentiment_for_date,
    slice_ohlcv,
)
from src.data.assembler import assemble_context
from src.data.indicators import compute_indicators
from src.data.macro import fetch_macro_for_date
from src.data.news_historical import fetch_news_for_date
from src.data.onchain_historical import fetch_onchain_for_date
from src.data.price import build_price_data

# GDELT coverage can be thin on some dates; fall back to the stub below this count.
_NEWS_MIN_ITEMS = 5

logger = logging.getLogger(__name__)

# Required columns in the output DataFrame
SIGNAL_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "signal", "size_pct", "sl_pct", "tp_pct", "conviction", "vetoed", "score",
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


def _append_agent_log(path: Path, date: str, data: dict) -> None:
    """Append one JSON-lines entry to an agent log file, flushing immediately."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date, "output": data}) + "\n")


async def generate_signals(
    full_ohlcv: pd.DataFrame,
    sentiment_history: dict,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10_000.0,
    client=None,
    csv_path: Path | None = None,
    agent_log_dir: Path | None = None,
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

    _csv_file = None
    _csv_writer = None
    if csv_path is not None:
        _csv_file = open(csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["date"] + SIGNAL_COLUMNS)
        _csv_file.flush()

    try:
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                # 1. Slice OHLCV — strict no-lookahead
                ohlcv_slice = slice_ohlcv(full_ohlcv, as_of=current)

                # 2. Get sentiment for this date
                sentiment = get_sentiment_for_date(sentiment_history, current)

                # 2b. Fetch historical news (GDELT), fall back to stub if empty
                news_items = fetch_news_for_date(current)
                if len(news_items) < _NEWS_MIN_ITEMS:
                    news_items = NEUTRAL_NEWS_STUB

                # 2c. Fetch historical on-chain (CoinMetrics Community, proxies)
                onchain_data = fetch_onchain_for_date(current)

                # 2d. Fetch macro snapshot (yfinance DXY/VIX/SPX/TNX)
                macro_data = fetch_macro_for_date(current)

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
                    news_override=news_items,
                    sentiment_override=sentiment,
                    onchain_override=onchain_data,
                    macro_override=macro_data,
                    skip_freshness=True,
                    min_news_items=_NEWS_MIN_ITEMS,
                )

                # 5. Run council
                result = await run_council(ctx, client=client)

                signal = result.signal
                conviction = result.deliberation.conviction
                vetoed = result.vetoed
                # v2 position size = score-derived sizing, capped by risk manager's
                # portfolio-risk approval. Risk veto is already reflected in signal=HOLD.
                score_size = score_to_position_size_pct(result.deliberation.score)
                size_pct = min(score_size, result.outputs.risk.approved_position_size_pct)
                close_price = float(ohlcv_slice.iloc[-1]["close"])
                sl_price = result.outputs.risk.recommended_stop_loss
                tp_price = result.outputs.risk.recommended_take_profit
                # Store as ratios relative to close so they work at any price scale
                sl_pct = (sl_price / close_price) if close_price > 0 and sl_price > 0 else 0.0
                tp_pct = (tp_price / close_price) if close_price > 0 and tp_price > 0 else 0.0

                logger.info(
                    "Cycle %s: signal=%s conviction=%s vetoed=%s",
                    date_str, signal, conviction, vetoed,
                )

                if agent_log_dir is not None:
                    agent_log_dir.mkdir(parents=True, exist_ok=True)
                    _append_agent_log(agent_log_dir / "technical_analyst.json", date_str, result.outputs.technical.model_dump())
                    _append_agent_log(agent_log_dir / "sentiment_analyst.json", date_str, result.outputs.sentiment.model_dump())
                    _append_agent_log(agent_log_dir / "fundamental_analyst.json", date_str, result.outputs.fundamental.model_dump())
                    _append_agent_log(agent_log_dir / "risk_manager.json", date_str, result.outputs.risk.model_dump())
                    _append_agent_log(agent_log_dir / "deliberation.json", date_str, result.deliberation.model_dump())

                score = result.deliberation.score

            except (AgentError, ValueError) as exc:
                logger.warning("Cycle %s failed (%s: %s); recording HOLD.", date_str, type(exc).__name__, exc)
                signal = "HOLD"
                conviction = "low"
                vetoed = False
                size_pct = 0.0
                sl_pct = 0.0
                tp_pct = 0.0
                score = 0.0
                today_row = _get_today_row(full_ohlcv, current)

            # 6. Record row
            row = {
                "date": current,
                "open": float(today_row["open"]) if today_row is not None else 0.0,
                "high": float(today_row["high"]) if today_row is not None else 0.0,
                "low": float(today_row["low"]) if today_row is not None else 0.0,
                "close": float(today_row["close"]) if today_row is not None else 0.0,
                "volume": float(today_row["volume"]) if today_row is not None else 0.0,
                "signal": signal,
                "size_pct": size_pct,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "conviction": conviction,
                "vetoed": vetoed,
                "score": score,
            }
            rows.append(row)

            if _csv_writer is not None:
                _csv_writer.writerow([
                    current.strftime("%Y-%m-%d"),
                    row["open"], row["high"], row["low"], row["close"], row["volume"],
                    row["signal"], row["size_pct"], row["sl_pct"], row["tp_pct"],
                    row["conviction"], row["vetoed"], row["score"],
                ])
                _csv_file.flush()

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
    finally:
        if _csv_file is not None:
            _csv_file.close()

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
