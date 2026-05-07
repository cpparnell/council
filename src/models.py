from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------- Shared
Direction = Literal["BUY", "SELL", "HOLD"]
Timeframe = Literal["24h", "48h", "72h"]
OverallSentiment = Literal["bullish", "bearish", "neutral", "mixed"]
OnchainBias = Literal["accumulation", "distribution", "neutral"]
MacroBias = Literal["risk-on", "risk-off", "neutral"]
Conviction = Literal["high", "medium", "low"]


class PriceData(BaseModel):
    current: float
    open_24h: float
    high_24h: float
    low_24h: float
    change_pct_24h: float
    volume_24h_usd: float
    volume_vs_7d_avg: float


MACDSignal = Literal["bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"]
BBPosition = Literal["upper", "mid", "lower"]
Regime = Literal["trending", "ranging", "high_volatility"]


class IndicatorData(BaseModel):
    rsi_14: float
    macd_signal: MACDSignal
    bb_position: BBPosition
    ema_20: float
    ema_50: float
    ema_200: float
    atr_14: float
    # atr_30_avg is used internally by the Tier-0 volatility halt check;
    # it is included in the context object but agents should not trade on it directly.
    atr_30_avg: float
    regime: Regime


class NewsItem(BaseModel):
    headline: str
    source: str
    published_at: str


Sentiment = Literal["bullish", "bearish", "neutral"]


class SentimentData(BaseModel):
    fear_greed_index: int = Field(ge=0, le=100)
    reddit_sentiment: Sentiment
    social_volume_vs_avg: float


class OnchainData(BaseModel):
    exchange_net_flow_btc: float
    whale_transactions_24h: int
    sopr: float


class MacroData(BaseModel):
    """Macro context (v2). DXY, VIX, SPX 20-session change, 10Y yield, and
    a derived risk regime classifier."""
    vix: float
    dxy: float
    spx_20d_change_pct: float
    tnx_yield_pct: float          # ^TNX divided by 10 (e.g. 3.97, not 39.7)
    macro_bias: MacroBias


class PortfolioData(BaseModel):
    btc_position_usd: float
    cash_usd: float
    current_drawdown_pct: float = Field(ge=0.0)
    peak_portfolio_value: float


class MarketContext(BaseModel):
    timestamp: datetime
    asset: str = "BTC/USD"
    price: PriceData
    indicators: IndicatorData
    news: list[NewsItem]
    sentiment: SentimentData
    onchain: OnchainData
    portfolio: PortfolioData
    # v2 additive: macro context. None in live mode until fetcher wired;
    # populated by the backtest loop via src.data.macro.fetch_macro_for_date.
    macro: MacroData | None = None


# -------------------------------------------------------- Agent output models

class TechnicalAnalystOutput(BaseModel):
    direction: Direction
    confidence: int = Field(ge=0, le=100)
    timeframe: Timeframe
    key_signals: list[str]
    invalidation_level: float


class SentimentAnalystOutput(BaseModel):
    direction: Direction
    confidence: int = Field(ge=0, le=100)
    overall_sentiment: OverallSentiment
    dominant_narrative: str
    high_impact_events: list[str]
    sentiment_vs_price_divergence: bool


class FundamentalAnalystOutput(BaseModel):
    direction: Direction
    confidence: int = Field(ge=0, le=100)
    onchain_bias: OnchainBias
    macro_bias: MacroBias
    key_factors: list[str]


class RiskManagerOutput(BaseModel):
    veto: bool
    veto_reason: str | None
    approved_position_size_pct: float = Field(ge=0, le=100)
    recommended_stop_loss: float
    recommended_take_profit: float
    risk_reward_ratio: float
    notes: str

    @field_validator("recommended_stop_loss", "recommended_take_profit", "risk_reward_ratio", mode="before")
    @classmethod
    def strip_commas(cls, v):
        if isinstance(v, str):
            return v.replace(",", "")
        return v


# --------------------------------------------------------- Deliberation model

class AgentWeights(BaseModel):
    technical: float
    sentiment: float
    fundamental: float
    risk: float


class CouncilOutputs(BaseModel):
    technical: TechnicalAnalystOutput
    sentiment: SentimentAnalystOutput
    fundamental: FundamentalAnalystOutput
    risk: RiskManagerOutput


class DeliberationOutput(BaseModel):
    final_signal: Direction
    conviction: Conviction
    consensus_summary: str
    key_disagreements: list[str]
    agent_weights_applied: AgentWeights
    # Signed score in [-1, 1] from compute_signed_score. 0.0 when risk-vetoed
    # or when this output was constructed without running the scorer.
    score: float = 0.0


# ------------------------------------------------- Generic strategy models (v3)

class GenericAgentOutput(BaseModel):
    direction: Direction
    confidence: int = Field(ge=0, le=100)
    reasoning: str


class VetoAgentOutput(BaseModel):
    veto: bool
    veto_reason: str | None = None
    reasoning: str


class GenericCouncilOutputs(BaseModel):
    """All agent outputs from a generic strategy run, keyed by agent name."""
    directional: dict[str, GenericAgentOutput]
    veto: VetoAgentOutput | None = None


class GenericDeliberationOutput(BaseModel):
    narrative: str
    final_signal: Direction = "HOLD"
    score: float = 0.0
