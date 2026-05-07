"""
SL/TP price computation from strategy risk rules.
"""

from src.models import MarketContext
from src.strategies.config import RiskRules, SLTPRule


def compute_sl_tp(
    rules: RiskRules,
    ctx: MarketContext,
    entry_price: float,
) -> tuple[float, float]:
    """Compute stop-loss and take-profit prices from config rules.

    Returns:
        (sl_price, tp_price). Both are 0.0 when the respective rule type is 'none'.
        sl_price is clamped to > 0.
    """
    sl_price = _compute_level(rules.stop_loss, ctx, entry_price, is_stop=True)
    tp_price = _compute_level(rules.take_profit, ctx, entry_price, is_stop=False)
    return sl_price, tp_price


def _compute_level(
    rule: SLTPRule,
    ctx: MarketContext,
    entry_price: float,
    is_stop: bool,
) -> float:
    if rule.type == "none":
        return 0.0

    if rule.type == "atr_multiple":
        atr = ctx.indicators.atr_14
        if is_stop:
            return max(entry_price - rule.value * atr, 0.01)
        else:
            return entry_price + rule.value * atr

    if rule.type == "fixed_pct":
        if is_stop:
            return max(entry_price * (1.0 - rule.value), 0.01)
        else:
            return entry_price * (1.0 + rule.value)

    return 0.0
