"""
Shared utilities for all council agents.
"""

import json
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    """Raised when an agent call fails or returns invalid output."""


def load_prompt(filename: str) -> str:
    """Load a system prompt from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text()


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the model wraps its JSON output."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def call_agent(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    user_message: str,
    output_model: type[T],
) -> T:
    """Call a single LLM agent and parse its output into a Pydantic model.

    Raises AgentError if the API call fails or the output does not validate.
    """
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise AgentError(f"API error calling {model}: {exc}") from exc

    raw = response.content[0].text
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise AgentError(
            f"Agent ({model}) returned non-JSON output: {raw!r}"
        ) from exc

    try:
        return output_model.model_validate(data)
    except ValidationError as exc:
        raise AgentError(
            f"Agent ({model}) output failed schema validation: {exc}"
        ) from exc
