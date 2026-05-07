"""Risk manager agent — claude-sonnet-4-6."""

import anthropic

from src.agents.base import call_agent, load_prompt
from src.models import MarketContext, RiskManagerOutput

MODEL = "claude-sonnet-4-6"
PROMPT_FILE = "risk_manager_v2.txt"


def _build_user_message(ctx: MarketContext) -> str:
    p = ctx.price
    ind = ctx.indicators
    port = ctx.portfolio
    portfolio_value = port.btc_position_usd + port.cash_usd
    return (
        f"Asset: {ctx.asset}\n"
        f"Current price: {p.current:,.2f} USD\n"
        f"ATR-14: {ind.atr_14:,.2f}  (30-candle avg: {ind.atr_30_avg:,.2f})\n"
        f"Regime: {ind.regime}\n\n"
        f"Portfolio:\n"
        f"  Total value: {portfolio_value:,.2f} USD\n"
        f"  BTC position: {port.btc_position_usd:,.2f} USD\n"
        f"  Cash: {port.cash_usd:,.2f} USD\n"
        f"  Current drawdown from peak: {port.current_drawdown_pct:.1%}\n"
        f"  Peak portfolio value: {port.peak_portfolio_value:,.2f} USD\n\n"
        "Evaluate risk and provide your risk management assessment as JSON."
    )


async def run_risk_manager(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic,
) -> RiskManagerOutput:
    """Run the risk manager agent and return its structured output."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(ctx)
    return await call_agent(client, MODEL, system, user, RiskManagerOutput)
