"""Sentiment analyst agent — claude-haiku-4-5 (deepseek placeholder)."""

import anthropic

from src.agents.base import call_agent, load_prompt
from src.models import MarketContext, SentimentAnalystOutput

# Spec assigns deepseek-v3 here; using claude-haiku-4-5 as a drop-in until
# DeepSeek is wired up via their OpenAI-compatible API.
MODEL = "claude-haiku-4-5"
PROMPT_FILE = "default/sentiment_analyst.txt"


def _build_user_message(ctx: MarketContext) -> str:
    p = ctx.price
    sent = ctx.sentiment
    headlines = "\n".join(
        f"  [{item.source}] {item.headline}" for item in ctx.news[:20]
    )
    return (
        f"Asset: {ctx.asset}\n"
        f"Current price: {p.current:,.2f} USD  (24h change: {p.change_pct_24h:+.2f}%)\n\n"
        f"Sentiment data:\n"
        f"  Fear/Greed index: {sent.fear_greed_index}/100\n"
        f"  Reddit sentiment: {sent.reddit_sentiment}\n"
        f"  Social volume vs avg: {sent.social_volume_vs_avg:.2f}x\n\n"
        f"Top news headlines (last 24h):\n{headlines}\n\n"
        "Provide your sentiment analysis signal as JSON."
    )


async def run_sentiment_analyst(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic,
) -> SentimentAnalystOutput:
    """Run the sentiment analyst agent and return its structured output."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(ctx)
    return await call_agent(client, MODEL, system, user, SentimentAnalystOutput)
