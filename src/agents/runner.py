"""
Council runner — orchestrates all agents for a single trading cycle.

Usage:
    from src.agents.runner import run_council
    result = await run_council(ctx)
"""

import asyncio
import os

import anthropic

from src.agents.base import AgentError
from src.agents.deliberation import run_deliberation
from src.agents.fundamental import run_fundamental_analyst
from src.agents.risk import run_risk_manager
from src.agents.scoring import compute_signed_score
from src.agents.sentiment import run_sentiment_analyst
from src.agents.technical import run_technical_analyst
from src.models import CouncilOutputs, DeliberationOutput, MarketContext


class CouncilResult:
    """Full output of a single council cycle."""

    def __init__(
        self,
        outputs: CouncilOutputs,
        deliberation: DeliberationOutput,
    ) -> None:
        self.outputs = outputs
        self.deliberation = deliberation

    @property
    def signal(self) -> str:
        return self.deliberation.final_signal

    @property
    def vetoed(self) -> bool:
        return self.outputs.risk.veto


async def run_council(
    ctx: MarketContext,
    client: anthropic.AsyncAnthropic | None = None,
) -> CouncilResult:
    """Run a full council cycle: all 4 agents in parallel, then deliberation.

    Args:
        ctx: Assembled and validated MarketContext for this cycle.
        client: Optional AsyncAnthropic client (created from env if not provided).

    Returns:
        CouncilResult containing all agent outputs and the final signal.

    Raises:
        AgentError: If any agent call fails (API error or schema validation failure).
    """
    if client is None:
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Run the three directional agents and the risk manager in parallel
    technical_task = run_technical_analyst(ctx, client)
    sentiment_task = run_sentiment_analyst(ctx, client)
    fundamental_task = run_fundamental_analyst(ctx, client)
    risk_task = run_risk_manager(ctx, client)

    technical, sentiment, fundamental, risk = await asyncio.gather(
        technical_task,
        sentiment_task,
        fundamental_task,
        risk_task,
    )

    outputs = CouncilOutputs(
        technical=technical,
        sentiment=sentiment,
        fundamental=fundamental,
        risk=risk,
    )

    scored_signal, score = compute_signed_score(outputs)

    # If the risk manager has vetoed, skip deliberation entirely and return HOLD.
    # compute_signed_score already returns ("HOLD", 0.0) in this case; we also
    # short-circuit the deliberation LLM call to save tokens.
    if risk.veto:
        from src.models import AgentWeights

        deliberation = DeliberationOutput(
            final_signal="HOLD",
            conviction="low",
            consensus_summary=f"Risk manager veto: {risk.veto_reason}",
            key_disagreements=[],
            agent_weights_applied=AgentWeights(
                technical=0.0, sentiment=0.0, fundamental=0.0, risk=1.0
            ),
            score=0.0,
        )
        return CouncilResult(outputs=outputs, deliberation=deliberation)

    # Deliberation LLM supplies narrative (summary, disagreements, weights).
    # The deterministic scorer overrides final_signal and attaches the score.
    llm_deliberation = await run_deliberation(outputs, client)
    deliberation = llm_deliberation.model_copy(
        update={"final_signal": scored_signal, "score": score}
    )
    return CouncilResult(outputs=outputs, deliberation=deliberation)
