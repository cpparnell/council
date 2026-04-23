"""
Confidence-weighted signed-score aggregation for the council's three
directional agents.

v2 replaces the v1 hard consensus thresholds (all-3 agree + avg≥70 / 2-of-3 +
avg≥60) with a smooth scoring function. The LLM deliberation agent still runs
and produces narrative, but `final_signal` is set deterministically by
`compute_signed_score` so threshold changes are testable without re-prompting.

Formula:

    score = Σ (w_i · sign(dir_i) · conf_i/100)  / Σ (w_i · max(conf_i/100, CONF_FLOOR/100))

Agents with confidence below CONF_FLOOR are excluded from the sum entirely
(low-confidence neutrals should abstain, not drag the score toward zero).

    |score| ≥ TRADE_THRESHOLD → BUY/SELL
    |score| <  TRADE_THRESHOLD → HOLD

Risk-manager veto unconditionally forces HOLD.
"""

from src.models import CouncilOutputs

WEIGHTS = {"technical": 0.40, "sentiment": 0.25, "fundamental": 0.35}
CONF_FLOOR = 30            # agents below this confidence abstain
TRADE_THRESHOLD = 0.25     # |score| must clear this to trade

_SIGN = {"BUY": 1, "SELL": -1, "HOLD": 0}


def compute_signed_score(outputs: CouncilOutputs) -> tuple[str, float]:
    """Compute (final_signal, score) from directional agent outputs.

    Returns:
        (signal, score) where signal ∈ {"BUY", "SELL", "HOLD"} and
        score ∈ [-1.0, +1.0] is the raw weighted score (independent of threshold).

    Risk-veto always wins: if outputs.risk.veto is True, returns ("HOLD", 0.0).
    """
    if outputs.risk.veto:
        return "HOLD", 0.0

    agents = {
        "technical": (outputs.technical.direction, outputs.technical.confidence),
        "sentiment": (outputs.sentiment.direction, outputs.sentiment.confidence),
        "fundamental": (outputs.fundamental.direction, outputs.fundamental.confidence),
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    for name, (direction, conf) in agents.items():
        if conf < CONF_FLOOR:
            continue
        w = WEIGHTS[name]
        weighted_sum += w * _SIGN[direction] * (conf / 100)
        weight_sum += w * (conf / 100)

    score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    if score >= TRADE_THRESHOLD:
        return "BUY", score
    if score <= -TRADE_THRESHOLD:
        return "SELL", score
    return "HOLD", score


def score_to_position_size_pct(score: float, max_pct: float = 20.0) -> float:
    """Derive position size from score magnitude.

    size_pct = max_pct × |score|, clamped to [0, max_pct].
    Returns 0.0 when |score| is below the trade threshold.
    """
    abs_score = abs(score)
    if abs_score < TRADE_THRESHOLD:
        return 0.0
    return min(max_pct, max_pct * abs_score)
