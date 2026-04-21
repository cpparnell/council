"""
Shared utilities for all council agents.
"""

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


async def call_agent(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    user_message: str,
    output_model: type[T],
) -> T:
    """Call a single LLM agent and parse its output into a Pydantic model.

    Uses tool_use with tool_choice to force the API to return structured JSON,
    avoiding markdown-wrapped or prose-mixed responses.

    Raises AgentError if the API call fails or the output does not validate.
    """
    tool_name = "structured_output"
    tool_def = {
        "name": tool_name,
        "description": "Return your analysis as structured JSON.",
        "input_schema": output_model.model_json_schema(),
    }

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise AgentError(f"API error calling {model}: {exc}") from exc

    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise AgentError(
            f"Agent ({model}) did not return a tool_use block; "
            f"stop_reason={response.stop_reason!r}"
        )

    try:
        return output_model.model_validate(tool_block.input)
    except ValidationError as exc:
        raise AgentError(
            f"Agent ({model}) output failed schema validation: {exc}"
        ) from exc
