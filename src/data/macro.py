"""
Historical macro data via yfinance — DXY, VIX, S&P 500, 10Y Treasury yield.

Gives the fundamental agent a structured macro backdrop so it can distinguish
"accumulation in a risk-on tape" from "accumulation during a risk-off panic."

Tickers:
  ^DXY   — US Dollar Index. yfinance coverage is thin pre-2020; we accept that
           and fall back to neutral macro_bias when DXY data is missing.
  ^VIX   — CBOE Volatility Index.
  ^GSPC  — S&P 500.
  ^TNX   — 10-Year Treasury yield. Quoted ×10 by yfinance (39.7 = 3.97%) —
           divided internally before return.

Lookahead rule: we slice `date < as_of.date()`. Daily bars close at the end
of the trading day, so the as_of day's bar has not yet printed.
"""

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from src.models import MacroBias, MacroData

logger = logging.getLogger(__name__)

_TICKERS = {
    "dxy": "DX-Y.NYB",  # ICE US Dollar Index on yfinance (preferred over ^DXY)
    "vix": "^VIX",
    "spx": "^GSPC",
    "tnx": "^TNX",
}

_LOOKBACK_DAYS = 45   # need ≥30 bars for the 20-session change, with slack for holidays


def fetch_macro_for_date(as_of: datetime) -> MacroData:
    """Fetch macro context visible as of *as_of*.

    Returns a MacroData with VIX level, DXY spot, SPX 20-session change,
    10Y yield (corrected for the ×10 quote), and a derived macro_bias.

    Falls back to a neutral MacroData on fetch error or if any ticker returns
    empty data — the fundamental agent's prompt documents the fallback so it
    doesn't reason about absent values as signal.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    end_date = as_of.date()
    start_date = end_date - timedelta(days=_LOOKBACK_DAYS)

    frames: dict[str, pd.DataFrame] = {}
    for name, ticker in _TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                frames[name] = pd.DataFrame()
            else:
                frames[name] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            frames[name] = pd.DataFrame()

    vix = _latest_close(frames.get("vix"))
    dxy = _latest_close(frames.get("dxy"))
    spx_change = _pct_change_over(frames.get("spx"), window=20)
    tnx_raw = _latest_close(frames.get("tnx"))
    tnx_yield = tnx_raw / 10.0 if tnx_raw is not None else None

    if vix is None or dxy is None or spx_change is None:
        # Missing a required input — fall back to neutral
        return MacroData(
            vix=vix or 0.0,
            dxy=dxy or 0.0,
            spx_20d_change_pct=spx_change or 0.0,
            tnx_yield_pct=tnx_yield or 0.0,
            macro_bias="neutral",
        )

    bias = _compute_macro_bias(
        vix=vix,
        dxy_20d_change_pct=_pct_change_over(frames.get("dxy"), window=20) or 0.0,
        spx_20d_change_pct=spx_change,
    )

    return MacroData(
        vix=vix,
        dxy=dxy,
        spx_20d_change_pct=spx_change,
        tnx_yield_pct=tnx_yield or 0.0,
        macro_bias=bias,
    )


def _latest_close(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    try:
        return float(df["Close"].iloc[-1])
    except (IndexError, ValueError):
        return None


def _pct_change_over(df: pd.DataFrame | None, window: int) -> float | None:
    """Percent change from (row[-window-1] close) to (row[-1] close)."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"].dropna()
    if len(closes) < window + 1:
        return None
    try:
        start = float(closes.iloc[-(window + 1)])
        end = float(closes.iloc[-1])
        if start <= 0:
            return None
        return (end - start) / start * 100.0
    except (IndexError, ValueError):
        return None


def _compute_macro_bias(
    *,
    vix: float,
    dxy_20d_change_pct: float,
    spx_20d_change_pct: float,
) -> MacroBias:
    """Classify the macro regime from VIX / DXY / SPX trend.

    Rules:
      risk-on   : VIX < 20 AND DXY 20d change ≤ 0 AND SPX 20d change > 0
      risk-off  : VIX > 25 OR DXY 20d change > +1.5%
      neutral   : otherwise
    """
    if vix > 25 or dxy_20d_change_pct > 1.5:
        return "risk-off"
    if vix < 20 and dxy_20d_change_pct <= 0.0 and spx_20d_change_pct > 0:
        return "risk-on"
    return "neutral"
