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
