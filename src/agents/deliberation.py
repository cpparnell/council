"""Deliberation (chair) agent — claude-sonnet-4-6."""

import json

import anthropic

from src.agents.base import call_agent, load_prompt
from src.models import CouncilOutputs, DeliberationOutput

MODEL = "claude-sonnet-4-6"
PROMPT_FILE = "deliberation_v2.txt"


def _build_user_message(outputs: CouncilOutputs) -> str:
    return (
        "Here are the four agent reports:\n\n"
        f"Technical analyst:\n{outputs.technical.model_dump_json(indent=2)}\n\n"
        f"Sentiment analyst:\n{outputs.sentiment.model_dump_json(indent=2)}\n\n"
        f"Fundamental analyst:\n{outputs.fundamental.model_dump_json(indent=2)}\n\n"
        f"Risk manager:\n{outputs.risk.model_dump_json(indent=2)}\n\n"
        "Synthesise these reports and provide the final trading decision as JSON."
    )


async def run_deliberation(
    outputs: CouncilOutputs,
    client: anthropic.AsyncAnthropic,
) -> DeliberationOutput:
    """Run the deliberation agent and return the final synthesised signal."""
    system = load_prompt(PROMPT_FILE)
    user = _build_user_message(outputs)
    return await call_agent(client, MODEL, system, user, DeliberationOutput)
