"""
Standard market context renderer.

Produces the text block sent to every agent in a generic strategy run.
All agents receive the same input; prompts focus the agent on relevant sections.
"""

from src.models import MarketContext


def render_context_block(ctx: MarketContext) -> str:
    """Render the full MarketContext as a structured text block."""
    p = ctx.price
    ind = ctx.indicators
    s = ctx.sentiment
    o = ctx.onchain
    port = ctx.portfolio

    lines = [
        "=== MARKET CONTEXT ===",
        f"Asset: {ctx.asset}",
        f"Timestamp: {ctx.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "--- PRICE ---",
        f"Current: ${p.current:,.2f}",
        f"24h change: {p.change_pct_24h:+.2f}%",
        f"24h high/low: ${p.high_24h:,.2f} / ${p.low_24h:,.2f}",
        f"Volume vs 7d avg: {p.volume_vs_7d_avg:.2f}x",
        "",
        "--- TECHNICAL INDICATORS ---",
        f"RSI-14: {ind.rsi_14:.1f}",
        f"MACD: {ind.macd_signal}",
        f"Bollinger Band position: {ind.bb_position}",
        f"EMA-20: ${ind.ema_20:,.2f}  EMA-50: ${ind.ema_50:,.2f}  EMA-200: ${ind.ema_200:,.2f}",
        f"ATR-14: ${ind.atr_14:,.2f}",
        f"Regime: {ind.regime}",
        "",
        f"--- NEWS (last 24h, {len(ctx.news)} items) ---",
    ]

    for i, item in enumerate(ctx.news[:20], 1):
        lines.append(f"{i}. [{item.source}] {item.headline}")

    lines += [
        "",
        "--- SENTIMENT ---",
        f"Fear/Greed Index: {s.fear_greed_index} ({_fg_label(s.fear_greed_index)})",
        f"Reddit sentiment: {s.reddit_sentiment}",
        f"Social volume vs avg: {s.social_volume_vs_avg:.2f}x",
        "",
        "--- ON-CHAIN ---",
    ]

    flow_label = "outflow / accumulation" if o.exchange_net_flow_btc < 0 else "inflow / distribution"
    lines += [
        f"Exchange net flow: {o.exchange_net_flow_btc:+,.0f} BTC ({flow_label})",
        f"Whale transactions 24h: {o.whale_transactions_24h}",
        f"MVRV: {o.sopr:.3f}",
    ]

    if ctx.macro:
        m = ctx.macro
        lines += [
            "",
            "--- MACRO ---",
            (
                f"VIX: {m.vix:.1f}  |  DXY: {m.dxy:.1f}  |  "
                f"SPX 20d: {m.spx_20d_change_pct:+.1f}%  |  10Y yield: {m.tnx_yield_pct:.2f}%"
            ),
            f"Macro bias: {m.macro_bias}",
        ]

    lines += [
        "",
        "--- PORTFOLIO ---",
        f"BTC position: ${port.btc_position_usd:,.2f} USD",
        f"Cash: ${port.cash_usd:,.2f} USD",
        f"Current drawdown: {port.current_drawdown_pct:.1%}",
        f"Peak portfolio value: ${port.peak_portfolio_value:,.2f}",
    ]

    return "\n".join(lines)


def _fg_label(index: int) -> str:
    if index <= 20:
        return "extreme fear"
    if index <= 40:
        return "fear"
    if index <= 60:
        return "neutral"
    if index <= 80:
        return "greed"
    return "extreme greed"
