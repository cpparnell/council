"""Fundamental / on-chain analyst agent — claude-haiku-4-5 (gemini placeholder)."""

import anthropic

from src.agents.base import call_agent, load_prompt
from src.models import FundamentalAnalystOutput, MarketContext

# Spec assigns gemini-2.0-flash here; using claude-haiku-4-5 as a drop-in until
# Gemini is wired up.
MODEL = "claude-haiku-4-5"
PROMPT_FILE = "fundamental_analyst_v2.txt"


def _build_user_message(ctx: MarketContext) -> str:
    p = ctx.price
    oc = ctx.onchain
    parts = [
        f"Asset: {ctx.asset}",
        f"Current price: {p.current:,.2f} USD  (24h change: {p.change_pct_24h:+.2f}%)",
        "",
        "On-chain data (v2 proxies — see system prompt for semantics):",
        f"  MVRV proxy (in `sopr` field): {oc.sopr:.4f} "
        f"({'unrealised profit / potential sell pressure' if oc.sopr > 1.1 else 'underwater / potential floor' if oc.sopr < 0.9 else 'neutral'})",
        f"  Transfer-volume trend (in `exchange_net_flow_btc` field): {oc.exchange_net_flow_btc:+,.1f} "
        f"({'rising activity — distribution-leaning' if oc.exchange_net_flow_btc > 0 else 'cooling activity — accumulation-leaning'})",
        f"  Activity-intensity proxy (in `whale_transactions_24h` field): {oc.whale_transactions_24h}",
    ]

    if ctx.macro is not None:
        m = ctx.macro
        parts.extend([
            "",
            "Macro backdrop:",
            f"  VIX: {m.vix:.2f}  DXY: {m.dxy:.2f}  10Y yield: {m.tnx_yield_pct:.2f}%",
            f"  S&P 500 20-session change: {m.spx_20d_change_pct:+.2f}%",
            f"  Regime classifier: {m.macro_bias}",
        ])
    else:
        parts.append("\nMacro backdrop: not provided.")

    parts.append("\nProvide your fundamental analysis signal as JSON.")
    return "\n".join(parts)


async def run_fundamental_analyst(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic,
) -> FundamentalAnalystOutput:
    """Run the fundamental analyst agent and return its structured output."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(ctx)
    return await call_agent(client, MODEL, system, user, FundamentalAnalystOutput)
