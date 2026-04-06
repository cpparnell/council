"""Fundamental / on-chain analyst agent — claude-haiku-4-5 (gemini placeholder)."""

import anthropic

from src.agents.base import call_agent, load_prompt
from src.models import FundamentalAnalystOutput, MarketContext

# Spec assigns gemini-2.0-flash here; using claude-haiku-4-5 as a drop-in until
# Gemini is wired up.
MODEL = "claude-haiku-4-5"
PROMPT_FILE = "fundamental_analyst_v1.txt"


def _build_user_message(ctx: MarketContext) -> str:
    p = ctx.price
    oc = ctx.onchain
    return (
        f"Asset: {ctx.asset}\n"
        f"Current price: {p.current:,.2f} USD  (24h change: {p.change_pct_24h:+.2f}%)\n\n"
        f"On-chain data:\n"
        f"  Exchange net flow (BTC): {oc.exchange_net_flow_btc:+,.0f} "
        f"({'outflows — accumulation signal' if oc.exchange_net_flow_btc < 0 else 'inflows — distribution signal'})\n"
        f"  Whale transactions (24h): {oc.whale_transactions_24h}\n"
        f"  SOPR: {oc.sopr:.4f} "
        f"({'holders selling at profit' if oc.sopr > 1 else 'holders selling at loss'})\n\n"
        "Provide your fundamental analysis signal as JSON."
    )


async def run_fundamental_analyst(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic,
) -> FundamentalAnalystOutput:
    """Run the fundamental analyst agent and return its structured output."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(ctx)
    return await call_agent(client, MODEL, system, user, FundamentalAnalystOutput)
