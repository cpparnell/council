"""
Reflection agent — runs after a position is closed.

Feeds the original council outputs, deliberation, and trade outcome to a
language model and returns a plain-text summary for the cycle log.  This
does NOT alter agent weights dynamically; summaries are for human review
and prompt refinement only.
"""

import os

import anthropic

from src.agents.base import AgentError, load_prompt
from src.models import CouncilOutputs, DeliberationOutput

MODEL = "claude-sonnet-4-6"
PROMPT_FILE = "reflection_v1.txt"


def _build_user_message(
    outputs: CouncilOutputs,
    deliberation: DeliberationOutput,
    trade: dict,
) -> str:
    pnl_usd = trade.get("pnl_usd", 0.0)
    pnl_pct = trade.get("pnl_pct", 0.0)
    outcome = "profit" if pnl_usd >= 0 else "loss"

    return (
        f"## Trade outcome\n"
        f"Side: {trade['side']}\n"
        f"Entry: {trade['entry_price']:,.2f} USD\n"
        f"Exit: {trade['exit_price']:,.2f} USD\n"
        f"Stop-loss: {trade['stop_loss']:,.2f} USD\n"
        f"Status: {trade['status']}\n"
        f"PnL: {pnl_usd:+,.2f} USD ({pnl_pct:+.2f}%) — {outcome}\n\n"
        f"## Council outputs at decision time\n\n"
        f"### Technical Analyst\n{outputs.technical.model_dump_json(indent=2)}\n\n"
        f"### Sentiment Analyst\n{outputs.sentiment.model_dump_json(indent=2)}\n\n"
        f"### Fundamental Analyst\n{outputs.fundamental.model_dump_json(indent=2)}\n\n"
        f"### Risk Manager\n{outputs.risk.model_dump_json(indent=2)}\n\n"
        f"## Deliberation summary\n{deliberation.model_dump_json(indent=2)}\n\n"
        "Please write your reflection now."
    )


async def run_reflection(
    outputs: CouncilOutputs,
    deliberation: DeliberationOutput,
    trade: dict,
    client: anthropic.AsyncAnthropic | None = None,
) -> str:
    """Generate a post-trade reflection summary.

    Args:
        outputs:      CouncilOutputs from the cycle that opened the trade.
        deliberation: DeliberationOutput from the same cycle.
        trade:        Closed trade dict from the DB (must include exit_price, pnl_usd, etc.).
        client:       Optional AsyncAnthropic client; created from env if not provided.

    Returns:
        Plain-text reflection summary (200–350 words).

    Raises:
        AgentError: If the API call fails.
    """
    if client is None:
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(outputs, deliberation, trade)

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as exc:
        raise AgentError(f"Reflection agent API error: {exc}") from exc

    return response.content[0].text.strip()
