"""
Generic strategy runner — drives a council cycle from a StrategyConfig.

Usage:
    from src.strategies.loader import load_strategy
    from src.strategies.runner import run_strategy_council

    config = load_strategy("strategies/my_strategy.yaml")
    result = await run_strategy_council(ctx, config)
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import anthropic

from src.agents.base import AgentError, call_agent
from src.models import (
    GenericAgentOutput,
    GenericCouncilOutputs,
    GenericDeliberationOutput,
    MarketContext,
    VetoAgentOutput,
)
from src.strategies.config import AgentConfig, DeliberationConfig, PROJECT_ROOT, StrategyConfig
from src.strategies.context import render_context_block
from src.strategies.risk_rules import compute_sl_tp
from src.strategies.scoring import compute_generic_score, score_to_position_size_pct

logger = logging.getLogger(__name__)


@dataclass
class GenericCouncilResult:
    """Full output of a single generic strategy council cycle."""
    signal: str
    score: float
    size_pct: float
    sl_price: float
    tp_price: float
    outputs: GenericCouncilOutputs
    narrative: str | None
    vetoed: bool

    @property
    def conviction(self) -> str:
        if abs(self.score) >= 0.6:
            return "high"
        if abs(self.score) >= 0.35:
            return "medium"
        return "low"


async def _call_directional(
    client: anthropic.AsyncAnthropic,
    agent: AgentConfig,
    context_block: str,
) -> GenericAgentOutput:
    system = (PROJECT_ROOT / agent.prompt).read_text()
    return await call_agent(client, agent.model, system, context_block, GenericAgentOutput)


async def _call_veto(
    client: anthropic.AsyncAnthropic,
    agent: AgentConfig,
    context_block: str,
) -> VetoAgentOutput:
    system = (PROJECT_ROOT / agent.prompt).read_text()
    return await call_agent(client, agent.model, system, context_block, VetoAgentOutput)


async def _call_deliberation(
    client: anthropic.AsyncAnthropic,
    config: DeliberationConfig,
    outputs: GenericCouncilOutputs,
) -> str:
    system = (PROJECT_ROOT / config.prompt).read_text()

    parts = ["Here are the agent outputs:\n"]
    for name, output in outputs.directional.items():
        parts.append(f"{name}:\n{output.model_dump_json(indent=2)}\n")
    if outputs.veto:
        parts.append(f"veto_agent:\n{outputs.veto.model_dump_json(indent=2)}\n")
    parts.append("Synthesise these outputs into a brief narrative.")

    user = "\n".join(parts)
    result = await call_agent(client, config.model, system, user, GenericDeliberationOutput)
    return result.narrative


async def run_strategy_council(
    ctx: MarketContext,
    config: StrategyConfig,
    client: anthropic.AsyncAnthropic | None = None,
) -> GenericCouncilResult:
    """Run a full council cycle driven by StrategyConfig.

    Steps:
    1. Render the context block once — all agents receive the same input.
    2. Run all agents in parallel (asyncio.gather).
    3. Short-circuit to HOLD if the veto agent fires.
    4. Compute signal and score deterministically from config weights.
    5. Optionally run deliberation agent for narrative.
    6. Compute SL/TP prices from config risk rules.
    """
    if client is None:
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    context_block = render_context_block(ctx)

    directional_agents = config.directional_agents
    veto_agent = config.veto_agent

    all_tasks: list = [
        _call_directional(client, agent, context_block)
        for agent in directional_agents
    ]
    if veto_agent:
        all_tasks.append(_call_veto(client, veto_agent, context_block))

    results = await asyncio.gather(*all_tasks)

    n_directional = len(directional_agents)
    directional_outputs: dict[str, GenericAgentOutput] = {
        directional_agents[i].name: results[i]
        for i in range(n_directional)
    }
    veto_output: VetoAgentOutput | None = results[n_directional] if veto_agent else None

    outputs = GenericCouncilOutputs(directional=directional_outputs, veto=veto_output)

    if veto_output and veto_output.veto:
        logger.info("Veto agent fired: %s", veto_output.veto_reason)
        sl_price, tp_price = compute_sl_tp(config.risk, ctx, ctx.price.current)
        return GenericCouncilResult(
            signal="HOLD",
            score=0.0,
            size_pct=0.0,
            sl_price=sl_price,
            tp_price=tp_price,
            outputs=outputs,
            narrative=f"Veto: {veto_output.veto_reason}",
            vetoed=True,
        )

    signal, score = compute_generic_score(outputs, config)
    size_pct = score_to_position_size_pct(score, config)

    narrative: str | None = None
    if config.deliberation and config.deliberation.enabled:
        try:
            narrative = await _call_deliberation(client, config.deliberation, outputs)
        except AgentError as exc:
            logger.warning("Deliberation agent failed (non-fatal): %s", exc)

    sl_price, tp_price = compute_sl_tp(config.risk, ctx, ctx.price.current)

    return GenericCouncilResult(
        signal=signal,
        score=score,
        size_pct=size_pct,
        sl_price=sl_price,
        tp_price=tp_price,
        outputs=outputs,
        narrative=narrative,
        vetoed=False,
    )
