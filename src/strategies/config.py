"""
Strategy configuration schema.

A strategy is defined entirely in a YAML file. This module provides the
typed Pydantic models that a loaded YAML is validated against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).parent.parent.parent


class AgentConfig(BaseModel):
    name: str
    model: str
    prompt: str
    weight: float = 0.0
    is_veto: bool = False


class DeliberationConfig(BaseModel):
    enabled: bool = True
    model: str = "claude-sonnet-4-6"
    prompt: str = "prompts/personal/deliberation_generic_v1.txt"


class SLTPRule(BaseModel):
    type: Literal["atr_multiple", "fixed_pct", "none"]
    value: float = 0.0


class RiskRules(BaseModel):
    stop_loss: SLTPRule
    take_profit: SLTPRule
    max_position_pct: float = Field(default=20.0, ge=0.0, le=100.0)


class ScoringConfig(BaseModel):
    confidence_floor: int = Field(default=30, ge=0, le=100)
    trade_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class ValidationConfig(BaseModel):
    # Expressed as a percentage: 15.0 = 15% drawdown halt
    max_drawdown_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    atr_spike_multiplier: float = Field(default=2.0, ge=0.0)
    min_news_count: int = Field(default=5, ge=0)


class StrategyConfig(BaseModel):
    name: str
    version: str = "1"
    agents: list[AgentConfig]
    deliberation: DeliberationConfig | None = None
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    risk: RiskRules
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @model_validator(mode="after")
    def _validate_agents(self) -> "StrategyConfig":
        if not self.agents:
            raise ValueError("Strategy must define at least one agent")

        veto_agents = [a for a in self.agents if a.is_veto]
        if len(veto_agents) > 1:
            raise ValueError(
                f"At most one agent may have is_veto=true, "
                f"found {len(veto_agents)}: {[a.name for a in veto_agents]}"
            )

        directional = [a for a in self.agents if not a.is_veto]
        if directional:
            total = sum(a.weight for a in directional)
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"Directional agent weights must sum to 1.0, got {total:.4f}"
                )

        return self

    @property
    def directional_agents(self) -> list[AgentConfig]:
        return [a for a in self.agents if not a.is_veto]

    @property
    def veto_agent(self) -> AgentConfig | None:
        return next((a for a in self.agents if a.is_veto), None)
