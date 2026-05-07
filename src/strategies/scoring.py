"""
Config-driven confidence-weighted signed-score aggregation.

Mirrors the logic in src/agents/scoring.py but reads weights and thresholds
from StrategyConfig rather than module-level constants, making them
per-strategy configurable.
"""

from src.models import GenericCouncilOutputs
from src.strategies.config import StrategyConfig

_SIGN: dict[str, int] = {"BUY": 1, "SELL": -1, "HOLD": 0}


def compute_generic_score(
    outputs: GenericCouncilOutputs,
    config: StrategyConfig,
) -> tuple[str, float]:
    """Compute (signal, score) from directional agent outputs.

    Weights are normalised at call time to guard against floating-point drift
    in the config. Agents below confidence_floor abstain from the score.
    Veto must be checked by the caller before invoking this function.

    Returns:
        (signal, score) where signal ∈ {"BUY", "SELL", "HOLD"} and
        score ∈ [-1.0, +1.0].
    """
    conf_floor = config.scoring.confidence_floor
    threshold = config.scoring.trade_threshold

    raw_weights = {a.name: a.weight for a in config.directional_agents}
    total_weight = sum(raw_weights.values())
    weights = (
        {k: v / total_weight for k, v in raw_weights.items()}
        if total_weight > 0
        else raw_weights
    )

    weighted_sum = 0.0
    weight_sum = 0.0

    for name, output in outputs.directional.items():
        if output.confidence < conf_floor:
            continue
        w = weights.get(name, 0.0)
        weighted_sum += w * _SIGN[output.direction] * (output.confidence / 100)
        weight_sum += w * (output.confidence / 100)

    score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    if score >= threshold:
        return "BUY", score
    if score <= -threshold:
        return "SELL", score
    return "HOLD", score


def score_to_position_size_pct(
    score: float,
    config: StrategyConfig,
) -> float:
    """Derive position size from score magnitude.

    size_pct = max_position_pct × |score|, clamped to [0, max_position_pct].
    Returns 0.0 when |score| is below the trade threshold.
    """
    threshold = config.scoring.trade_threshold
    max_pct = config.risk.max_position_pct
    abs_score = abs(score)
    if abs_score < threshold:
        return 0.0
    return min(max_pct, max_pct * abs_score)
