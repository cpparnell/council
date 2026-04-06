# Changelog

## Phase 3 — 2026-04-05

### Added
- `src/db/schema.py` — SQLAlchemy table definitions (`cycles`, `trades`, `portfolio_state`, `reflections`)
- `src/db/store.py` — CRUD operations: `log_cycle`, `open_trade`, `close_trade`, `get_open_trade`, `get_portfolio_state`, `update_portfolio_state`, `save_reflection`, `get_reflection`
- `src/execution/orders.py` — `place_market_buy`, `place_market_sell`, `place_stop_loss`, `cancel_order` with retry logic (max 3 attempts, exponential back-off)
- `src/execution/state.py` — `build_portfolio_dict`: loads live portfolio state from DB for context assembly
- `src/execution/router.py` — `route_signal`: translates `CouncilResult` → exchange orders + DB updates; `reconcile_open_position`: detects stop-outs at cycle start
- `src/agents/reflection.py` — post-trade reflection agent (claude-sonnet-4-6)
- `prompts/reflection_v1.txt` — system prompt for the reflection agent
- `src/main.py` — full cycle entry point; supports one-shot and `--schedule` (APScheduler daily at 00:05 UTC); `DRY_RUN=1` skips order execution
- `tests/test_db.py` — 16 tests for DB schema and CRUD
- `tests/test_execution.py` — 22 tests for orders, state, and router
- `tests/test_reflection.py` — 8 tests for the reflection agent
- `sqlalchemy>=2.0.0` and `apscheduler>=3.10.0` added to dependencies

## Phase 2 — Agent Framework

### Added
- `src/models.py` — extended with all agent output models: `TechnicalAnalystOutput`, `SentimentAnalystOutput`, `FundamentalAnalystOutput`, `RiskManagerOutput`, `CouncilOutputs`, `DeliberationOutput`, `AgentWeights`
- `src/agents/base.py` — `call_agent()` helper: loads prompt file, calls Anthropic API at `temperature=0`, strips markdown fences from response, validates JSON against Pydantic schema; raises `AgentError` on any failure
- `src/agents/technical.py` — Technical Analyst agent (claude-haiku-4-5); analyses RSI momentum, MACD crosses, Bollinger Band position, EMA alignment, ATR volatility
- `src/agents/sentiment.py` — Sentiment Analyst agent (claude-haiku-4-5); analyses news tone, fear/greed index, social volume, dominant narrative
- `src/agents/fundamental.py` — Fundamental Analyst agent (claude-haiku-4-5); analyses exchange net flows, whale transactions, SOPR, macro bias
- `src/agents/risk.py` — Risk Manager agent (claude-sonnet-4-6); evaluates portfolio risk, outputs position size, stop-loss, take-profit, R:R; has veto authority
- `src/agents/deliberation.py` — Deliberation Chair agent (claude-sonnet-4-6); synthesises all four agent outputs into final BUY/SELL/HOLD signal with conviction level
- `src/agents/runner.py` — `run_council()`: runs all four agents in parallel via `asyncio.gather()`; short-circuits to HOLD immediately if risk manager vetoes (skips deliberation); returns `CouncilResult`
- `prompts/technical_analyst_v1.txt` — system prompt for Technical Analyst
- `prompts/sentiment_analyst_v1.txt` — system prompt for Sentiment Analyst
- `prompts/fundamental_analyst_v1.txt` — system prompt for Fundamental Analyst
- `prompts/risk_manager_v1.txt` — system prompt for Risk Manager
- `prompts/deliberation_v1.txt` — system prompt for Deliberation Chair (includes consensus rules: all-3-agree ≥70% → full size; 2-of-3 ≥60% → 50% size; else HOLD)
- `tests/test_agents.py` — 38 tests: `call_agent` base utility, individual agent schema validation, deliberation consensus logic, runner veto short-circuit, prompt file existence checks

## Phase 1 — Data Pipeline + Tier-0 Validation

### Added
- `src/models.py` — Pydantic v2 models for the full context schema: `PriceData`, `IndicatorData`, `NewsItem`, `SentimentData`, `OnchainData`, `PortfolioData`, `MarketContext`; literal types for `Direction`, `MACDSignal`, `BBPosition`, `Regime`, etc.
- `src/data/price.py` — CCXT Binance testnet OHLCV fetcher (`fetch_ohlcv`, `build_price_data`); raises `ValueError` if fewer than 200 candles returned
- `src/data/indicators.py` — `compute_indicators()`: RSI-14, MACD cross detection, Bollinger Band position, EMA 20/50/200, ATR-14, ATR-30 rolling average, regime classifier (trending/ranging/high_volatility via ADX + ATR)
- `src/data/news.py` — async CryptoPanic fetcher; returns empty list if API key absent
- `src/data/sentiment.py` — async Alternative.me Fear/Greed Index fetcher; LunarCrush stub returning neutral defaults
- `src/data/onchain.py` — Glassnode/CryptoQuant stubs returning zero defaults
- `src/data/assembler.py` — `assemble_context()`: fetches OHLCV synchronously (ccxt is sync), fetches news/sentiment/on-chain in parallel with `asyncio.gather()`, assembles `MarketContext`, runs Tier-0 validation
- `src/validation.py` — `validate_context()`: price freshness check (< 5 min), minimum news count (≥ 10), numeric range checks; `HardRuleViolation` raised for drawdown halt (> 15% from peak) and ATR volatility halt (ATR-14 > 2× ATR-30 avg)
- `tests/test_indicators.py` — 35 tests for all indicator computations
- `tests/test_assembler.py` — context assembly tests with mocked data feeds and default portfolio
- `tests/test_validation.py` — 35 tests for price freshness, news count, numeric ranges, drawdown halt, volatility halt
