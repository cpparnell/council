"""Technical analyst agent — claude-haiku-4-5."""

import anthropic

from src.agents.base import AgentError, call_agent, load_prompt
from src.models import MarketContext, TechnicalAnalystOutput

MODEL = "claude-haiku-4-5"
PROMPT_FILE = "technical_analyst_v2.txt"


def _build_user_message(ctx: MarketContext) -> str:
    ind = ctx.indicators
    p = ctx.price
    return (
        f"Asset: {ctx.asset}\n"
        f"Current price: {p.current:,.2f} USD\n"
        f"24h change: {p.change_pct_24h:+.2f}%\n"
        f"Volume vs 7d avg: {p.volume_vs_7d_avg:.2f}x\n\n"
        f"Indicators:\n"
        f"  RSI-14: {ind.rsi_14:.1f}\n"
        f"  MACD signal: {ind.macd_signal}\n"
        f"  Bollinger Band position: {ind.bb_position}\n"
        f"  EMA-20: {ind.ema_20:,.2f}  EMA-50: {ind.ema_50:,.2f}  EMA-200: {ind.ema_200:,.2f}\n"
        f"  ATR-14: {ind.atr_14:,.2f}  (30-candle avg: {ind.atr_30_avg:,.2f})\n"
        f"  Regime: {ind.regime}\n\n"
        "Provide your technical analysis signal as JSON."
    )


async def run_technical_analyst(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic,
) -> TechnicalAnalystOutput:
    """Run the technical analyst agent and return its structured output."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(ctx)
    return await call_agent(client, MODEL, system, user, TechnicalAnalystOutput)
