"""
Strategy YAML loader.

Loads a strategy config file, validates its schema, and checks that all
referenced prompt files exist on disk.
"""

from pathlib import Path

import yaml

from src.strategies.config import PROJECT_ROOT, StrategyConfig


def load_strategy(path: str | Path) -> StrategyConfig:
    """Load and validate a strategy YAML file.

    Prompt paths are resolved relative to the project root and validated
    to exist before the config is returned.

    Raises:
        FileNotFoundError: if the strategy file itself does not exist.
        ValueError: if schema validation fails or a prompt file is missing.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = StrategyConfig.model_validate(data)
    _validate_prompt_paths(config)
    return config


def _validate_prompt_paths(config: StrategyConfig) -> None:
    for agent in config.agents:
        p = PROJECT_ROOT / agent.prompt
        if not p.exists():
            raise ValueError(
                f"Agent '{agent.name}' prompt not found: {agent.prompt} "
                f"(resolved to {p})"
            )

    if config.deliberation and config.deliberation.enabled:
        p = PROJECT_ROOT / config.deliberation.prompt
        if not p.exists():
            raise ValueError(
                f"Deliberation prompt not found: {config.deliberation.prompt} "
                f"(resolved to {p})"
            )
