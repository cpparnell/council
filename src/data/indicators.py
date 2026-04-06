"""
Technical indicator computation using pandas-ta.

All functions operate purely on historical OHLCV DataFrames — no network
calls, no side effects.  The only requirement is ≥ 200 rows (enforced by
the price fetcher).

Regime classification (spec §Key Challenges §3):
  - high_volatility : current ATR-14 > 2× rolling 30-candle average of ATR-14
  - trending        : ADX-14 > 25
  - ranging         : everything else
"""

import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute all technical indicators required by the context object schema.

    Args:
        df: OHLCV DataFrame with columns open/high/low/close/volume and a
            DatetimeIndex.  Must have ≥ 200 rows.

    Returns:
        Dict matching the IndicatorData schema, including *atr_30_avg* for
        Tier-0 volatility checks.

    Raises:
        ValueError: if the DataFrame has fewer than 200 rows.
    """
    if len(df) < 200:
        raise ValueError(
            f"indicators require ≥ 200 candles, got {len(df)}"
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ------------------------------------------------------------------ RSI-14
    # Column name: RSI_14
    rsi_series = ta.rsi(close, length=14)
    rsi_14 = float(rsi_series.iloc[-1])

    # --------------------------------------------------------------- MACD cross
    # Column names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    macd_line = macd_df["MACD_12_26_9"]
    signal_line = macd_df["MACDs_12_26_9"]

    curr_macd, prev_macd = float(macd_line.iloc[-1]), float(macd_line.iloc[-2])
    curr_sig, prev_sig = float(signal_line.iloc[-1]), float(signal_line.iloc[-2])

    if prev_macd <= prev_sig and curr_macd > curr_sig:
        macd_signal = "bullish_cross"
    elif prev_macd >= prev_sig and curr_macd < curr_sig:
        macd_signal = "bearish_cross"
    elif curr_macd > curr_sig:
        macd_signal = "bullish"
    elif curr_macd < curr_sig:
        macd_signal = "bearish"
    else:
        macd_signal = "neutral"

    # --------------------------------------------------- Bollinger Band position
    # Column names vary by pandas-ta version; pick BBP column dynamically.
    # BBP (percent-B) = (close - lower) / (upper - lower)
    bb_df = ta.bbands(close, length=20, std=2)
    bbp_col = next(c for c in bb_df.columns if c.startswith("BBP_"))
    bb_pct = float(bb_df[bbp_col].iloc[-1])

    if bb_pct >= 0.8:
        bb_position = "upper"
    elif bb_pct <= 0.2:
        bb_position = "lower"
    else:
        bb_position = "mid"

    # -------------------------------------------------------------------- EMAs
    ema_20 = float(ta.ema(close, length=20).iloc[-1])
    ema_50 = float(ta.ema(close, length=50).iloc[-1])
    ema_200 = float(ta.ema(close, length=200).iloc[-1])

    # ------------------------------------------------------------------- ATR-14
    # Column name: ATRr_14  (some versions use ATR_14 — pick dynamically)
    atr_series = ta.atr(high, low, close, length=14)
    atr_14 = float(atr_series.iloc[-1])

    # Rolling 30-candle average of ATR-14 (excluding the latest candle)
    # Used by Tier-0 to detect extreme volatility events.
    atr_30_avg = float(atr_series.iloc[-31:-1].mean())

    # ------------------------------------------------- Regime classification
    # Column name: ADX_14
    adx_df = ta.adx(high, low, close, length=14)
    adx = float(adx_df["ADX_14"].iloc[-1])

    if atr_14 > 2.0 * atr_30_avg:
        regime = "high_volatility"
    elif adx > 25:
        regime = "trending"
    else:
        regime = "ranging"

    return {
        "rsi_14": round(rsi_14, 2),
        "macd_signal": macd_signal,
        "bb_position": bb_position,
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "ema_200": round(ema_200, 2),
        "atr_14": round(atr_14, 2),
        "atr_30_avg": round(atr_30_avg, 2),
        "regime": regime,
    }
