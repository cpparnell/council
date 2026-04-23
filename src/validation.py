"""
Tier-0 validation and hard risk rules.

All checks run before any LLM call.  Any ValidationError or
HardRuleViolation must result in a HOLD signal — enforced by the caller.

Hard risk constants (spec §Risk Management):
  MAX_POSITION_SIZE_PCT = 0.20  — enforced by order router (Phase 3)
  MAX_TRADE_RISK_PCT    = 0.02  — enforced by order router (Phase 3)
  MAX_DRAWDOWN_HALT_PCT = 0.15  — enforced here (check_drawdown_halt)
  MAX_ATR_MULTIPLIER    = 2.0   — enforced here (check_volatility_halt)
  MIN_RISK_REWARD       = 1.5   — enforced by order router (Phase 3)
"""

from datetime import datetime, timedelta, timezone

from src.models import MarketContext

# ----------------------------------------------------------------- Constants
MAX_POSITION_SIZE_PCT = 0.20
MAX_TRADE_RISK_PCT = 0.02
MAX_DRAWDOWN_HALT_PCT = 0.15
MAX_ATR_MULTIPLIER = 2.0
MIN_RISK_REWARD = 1.5

# Plausible BTC price range used for sanity checks
_BTC_PRICE_MIN = 1_000.0
_BTC_PRICE_MAX = 1_000_000.0


# ---------------------------------------------------------------- Exceptions
class ValidationError(Exception):
    """Data quality failure — cycle should default to HOLD."""


class HardRuleViolation(Exception):
    """Hard risk rule breach — trading is halted for this cycle."""


# --------------------------------------------------------------- Validators
def validate_price_freshness(ctx: MarketContext, max_age_minutes: int = 5) -> None:
    """Reject the cycle if the context timestamp is older than *max_age_minutes*.

    The assembler stamps ctx.timestamp at the moment of assembly, so this
    check catches clock skew or a stale cached context being re-used.
    """
    ts = ctx.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - ts
    if age > timedelta(minutes=max_age_minutes):
        raise ValidationError(
            f"Context is stale: assembled {age.total_seconds():.0f}s ago "
            f"(max {max_age_minutes * 60}s)"
        )


def validate_news_count(ctx: MarketContext, min_items: int = 10) -> None:
    """Reject the cycle if fewer than *min_items* news articles were retrieved.

    Backtest callers pass a lower floor (5) because the GDELT allowlist can be
    thin on some historical dates. Live callers keep the default of 10.
    """
    if len(ctx.news) < min_items:
        raise ValidationError(
            f"Insufficient news: {len(ctx.news)} items retrieved, need ≥ {min_items}"
        )


def validate_numeric_fields(ctx: MarketContext) -> None:
    """Sanity-check all numeric fields against plausible ranges.

    These checks catch data-feed errors (e.g. a zero price from a failed
    API call) rather than normal market moves.
    """
    p = ctx.price
    ind = ctx.indicators
    sent = ctx.sentiment

    if not (_BTC_PRICE_MIN < p.current < _BTC_PRICE_MAX):
        raise ValidationError(
            f"BTC price {p.current} is outside plausible range "
            f"[{_BTC_PRICE_MIN}, {_BTC_PRICE_MAX}]"
        )
    if p.volume_24h_usd <= 0 or p.volume_24h_usd > 1e13:
        raise ValidationError(f"24h USD volume out of range: {p.volume_24h_usd}")

    if not (0.0 <= ind.rsi_14 <= 100.0):
        raise ValidationError(f"RSI-14 out of range [0, 100]: {ind.rsi_14}")
    if ind.atr_14 <= 0:
        raise ValidationError(f"ATR-14 must be positive, got {ind.atr_14}")
    if ind.ema_20 <= 0 or ind.ema_50 <= 0 or ind.ema_200 <= 0:
        raise ValidationError("One or more EMAs are ≤ 0")

    # Pydantic already validates fear_greed_index bounds (ge=0, le=100) on
    # model construction, but we re-check here for defence-in-depth.
    if not (0 <= sent.fear_greed_index <= 100):
        raise ValidationError(
            f"Fear/Greed index {sent.fear_greed_index} out of range [0, 100]"
        )


def check_drawdown_halt(ctx: MarketContext) -> None:
    """Halt trading if the portfolio has lost more than MAX_DRAWDOWN_HALT_PCT
    from its peak value.

    This is a hard rule — it cannot be overridden by any agent.
    """
    if ctx.portfolio.current_drawdown_pct > MAX_DRAWDOWN_HALT_PCT:
        raise HardRuleViolation(
            f"Drawdown halt: portfolio is down "
            f"{ctx.portfolio.current_drawdown_pct:.1%} from peak "
            f"(threshold {MAX_DRAWDOWN_HALT_PCT:.1%})"
        )


def check_volatility_halt(ctx: MarketContext) -> None:
    """Halt trading if current ATR-14 exceeds MAX_ATR_MULTIPLIER × 30-candle average.

    Uses ctx.indicators.atr_30_avg computed by the indicators module.
    Skips the check when atr_30_avg is zero (e.g. on a stub context).
    """
    avg = ctx.indicators.atr_30_avg
    if avg <= 0:
        return
    if ctx.indicators.atr_14 > MAX_ATR_MULTIPLIER * avg:
        raise HardRuleViolation(
            f"Volatility halt: ATR-14 {ctx.indicators.atr_14:.0f} > "
            f"{MAX_ATR_MULTIPLIER}× 30-candle average {avg:.0f}"
        )


def validate_context(
    ctx: MarketContext,
    skip_freshness: bool = False,
    min_news_items: int = 10,
) -> None:
    """Run all Tier-0 checks in order.

    Call this once after assembling the context and before invoking any agent.
    Any exception here must produce a HOLD signal.

    Args:
        ctx: The assembled MarketContext to validate.
        skip_freshness: When True, the price-freshness check is skipped.
                        Pass True for historical backtest contexts whose
                        timestamps are intentionally in the past.
        min_news_items: Minimum news count required. Live mode uses 10;
                        backtest mode uses 5 (GDELT historical coverage
                        can be thinner than live CryptoPanic).
    """
    if not skip_freshness:
        validate_price_freshness(ctx)
    validate_news_count(ctx, min_items=min_news_items)
    validate_numeric_fields(ctx)
    check_drawdown_halt(ctx)
    check_volatility_halt(ctx)
