# Changelog

## Update — 2026-04-20 (per-cycle agent response logs)

### Added
- `src/backtest/signals.py` — `_append_agent_log(path, date, data)`: appends a JSON Lines entry `{"date": "...", "output": {...}}` to an agent file and closes immediately; `generate_signals` accepts a new `agent_log_dir: Path | None` parameter and calls this after each successful cycle, writing `technical_analyst.json`, `sentiment_analyst.json`, `fundamental_analyst.json`, `risk_manager.json`, and `deliberation.json` under `tmp/YYYYMMDD_HHMMSS/agents/`
- `src/backtest/engine.py` — derives `agent_log_dir = run_dir / "agents"` and passes it to `generate_signals`; end-of-run summary now prints the `agents/` folder

### Fixed
- `src/models.py` — `RiskManagerOutput`: added `field_validator` on `recommended_stop_loss`, `recommended_take_profit`, and `risk_reward_ratio` to strip commas before float parsing, fixing `AgentError` when the model returns formatted numbers like `'71,001.50'`

## Update — 2026-04-20 (backtest run output to tmp/, incremental CSV)

### Changed
- `src/backtest/engine.py` — `_setup_run_dir` default base changed from `runs/` to `tmp/`; timestamped run folders now live under `tmp/YYYYMMDD_HHMMSS/`
- `src/backtest/engine.py` — `run_backtest` now resolves `csv_path` before calling `generate_signals` and passes it in; the post-hoc `signals_df.to_csv()` call is removed
- `src/backtest/signals.py` — `generate_signals` accepts a new `csv_path: Path | None` parameter; when set, opens the file before the signal loop, writes the header row, then flushes one CSV row per completed cycle so the file is preserved if the run is killed mid-way

## Feature — 2026-04-20 (run output folder)

### Added
- `src/backtest/engine.py` — `_setup_run_dir(base)`: creates `runs/YYYYMMDD_HHMMSS/` on each invocation; adds a `logging.StreamHandler` pointing at `run.log` inside that directory (StreamHandler flushes after every record); tees `sys.stdout` and `sys.stderr` through `_TeeStream` so all `print()` output is also captured; uses `buffering=1` (line-buffered) so every newline triggers an OS write and output is preserved if the process is killed
- `_TeeStream`: lightweight wrapper that mirrors writes to two streams simultaneously
- `run_backtest` now accepts an optional `run_dir: Path` parameter; when provided, the signals CSV is written into that directory instead of the working directory
- CSV filename changed from `backtest_signals_<full-datetime-with-tz>.csv` (contained colons, broke on some OS) to `backtest_signals_YYYYMMDD_YYYYMMDD.csv` (start/end dates)
- End-of-run summary prints the run folder path, CSV filename, and log filename

### Fixed
- `run_backtest` previously called `datetime.now()` twice for the CSV filename, producing a different timestamp in the returned `csv_path` than the one used when writing the file

## Bugfix — 2026-04-20 (structured JSON output via tool use)

### Fixed
- `src/agents/base.py` — replaced free-form text generation + markdown-fence stripping with Anthropic tool use (`tools` + `tool_choice={"type": "tool"}`); the API now returns a `tool_use` block whose `.input` is a guaranteed parsed dict, eliminating the `non-JSON output` errors caused by models wrapping responses in ` ```json ``` ` fences with trailing prose
- Removed `_extract_json` helper and `import json` (no longer needed)
- `AgentError` now raised with `"tool_use block"` message when the response contains no tool-use block (e.g. unexpected `end_turn`)

### Tests
- `tests/test_agents.py` — updated `_mock_response`/`_mock_client` to return a mock with `type="tool_use"` and `.input=payload` instead of a text block; replaced `test_strips_markdown_fences` and `test_raises_on_non_json` (scenarios no longer possible) with `test_raises_on_missing_tool_use_block`; removed unused `import json`; 235 tests passing

## Bugfix — 2026-04-06 (fractional trading + dotenv)

### Fixed
- `src/backtest/engine.py` — switched from `Backtest` to `FractionalBacktest` (backtesting.lib); BTC trades in fractional units so any `--capital` value now works without orders being silently canceled
- `src/backtest/signals.py` — replaced `sl_price`/`tp_price` (raw USD) with `sl_pct`/`tp_pct` (ratio relative to close, e.g. 0.95 = 5% stop below close); `FractionalBacktest` scales OHLCV prices internally (to satoshi units) so absolute USD prices caused constraint failures (`SL < execution_price < TP` always failed against the scaled price)
- `src/backtest/engine.py` strategy — `next()` now computes `sl = Close * sl_pct` and `tp = Close * tp_pct`, which scales correctly under any price transformation
- `tests/test_backtest.py` — updated all test fixtures and `_make_signals_df` to use `sl_pct`/`tp_pct`; `_run_strategy` cash reverted to `$10,000` (fractional support makes large cash workaround unnecessary)

## Bugfix — 2026-04-06

### Fixed
- `src/main.py`, `src/backtest/engine.py` — added `load_dotenv()` call at startup; `.env` file was never being read, so `ANTHROPIC_API_KEY` and all other env vars were silently missing at runtime

## v1 Phase 6 (Historical Data Source) — 2026-04-06

### Changed
- `src/backtest/data.py` — `fetch_full_ohlcv` rewritten to use `yfinance.download("BTC-USD")` instead of CCXT; removes exchange parameter and pagination loop; handles MultiIndex columns (yfinance 0.2+); UTC-localizes index
- `src/backtest/engine.py` — `run_backtest` no longer creates a Kraken exchange for data fetching; `get_exchange` import removed
- `pyproject.toml` — added `yfinance>=0.2.0`

### Fixed
- Kraken's public OHLC API only serves the most recent ~720 daily candles regardless of the `since` parameter; requesting data older than that (e.g. `--start 2024-01-01` from April 2026) returned 0 candles. yfinance provides BTC-USD history back to 2014 with no API key.

### Tests
- `tests/test_backtest.py` — replaced CCXT-based `fetch_full_ohlcv` tests with yfinance mocks; removed `get_exchange` mock from `run_backtest` tests

## v1 Phase 5 (Exchange Migration) — 2026-04-06 (revised)

### Fixed
- `src/data/price.py` — corrected CCXT symbol from `"XBT/USD"` to `"BTC/USD"`; CCXT normalizes Kraken's internal XBT ticker to BTC in its unified API, so `XBT/USD` raises `BadSymbol` at runtime

## v1 Phase 5 (Exchange Migration) — 2026-04-06

### Changed
- `src/data/price.py` — replaced `ccxt.binance` with `ccxt.kraken`; `SYMBOL` changed from `"BTC/USDT"` to `"XBT/USD"`; removed Binance testnet URL logic; credentials now read from `KRAKEN_API_KEY` / `KRAKEN_SECRET`; `sandbox` param retained for interface compat but ignored (Kraken has no spot sandbox)
- `src/execution/router.py` — `SYMBOL = "XBT/USD"`
- `src/execution/state.py` — default symbol updated to `"XBT/USD"`
- `src/execution/orders.py` — docstring updated (Binance → Kraken)
- `src/db/store.py` — default symbol in `get_open_trade` updated to `"XBT/USD"`
- `src/main.py` — `SYMBOL = "XBT/USD"`; env var docs reflect Kraken credentials; paper trading note updated (DRY_RUN=1 replaces testnet)
- `src/data/assembler.py` — docstring updated
- `.env.example` — replaced `BINANCE_TESTNET_API_KEY/SECRET` and `BINANCE_API_KEY/SECRET` with `KRAKEN_API_KEY` / `KRAKEN_SECRET`
- `tests/test_execution.py`, `tests/test_db.py`, `tests/test_reporting.py` — all `"BTC/USDT"` references updated to `"XBT/USD"`

**Why:** Binance is geo-restricted in the United States. Kraken is fully US-accessible, free for market data, and provides BTC/USD history back to 2013 via CCXT public endpoints.

**Paper trading:** Kraken has no spot sandbox. Paper trading uses `DRY_RUN=1` (signals logged, no orders placed). Live trading requires `KRAKEN_API_KEY` + `KRAKEN_SECRET`.

## v1 Phase 1–4 (Backtesting) — 2026-04-06

### Added
- `src/backtest/__init__.py` — new backtest package
- `src/backtest/data.py` — `fetch_full_ohlcv` (paginated CCXT download), `slice_ohlcv` (strict no-lookahead enforcement), `fetch_historical_sentiment` (Alternative.me Fear & Greed history), `get_sentiment_for_date` (nearest-earlier fallback), `NEUTRAL_NEWS_STUB` (15 generic items), `NEUTRAL_ONCHAIN_STUB`
- `src/backtest/signals.py` — `generate_signals`: async LLM council loop over each historical date; `_PortfolioTracker` dataclass (cash/position/drawdown tracking); `SIGNAL_COLUMNS` constant; graceful HOLD fallback on `AgentError`
- `src/backtest/engine.py` — `LLMCouncilStrategy` (backtesting.py Strategy replaying pre-computed signals); `run_backtest` async runner (fetch → signals → simulate → formatted stats dict); CLI via `python -m src.backtest.engine --start --end --capital`
- `tests/test_backtest.py` — 26 tests: `TestBacktestData` (slice no-lookahead, pagination, sentiment fallbacks, news stub), `TestGenerateSignals` (column schema, date count, lookahead guard, error fallback, portfolio tracker), `TestRunBacktest` (strategy execution, stop-loss trigger, stats keys)
- `backtesting>=0.3.3` added to `pyproject.toml`

### Changed
- `src/data/price.py` — added `since: int | None = None` to `fetch_ohlcv` for historical backfill
- `src/data/assembler.py` — added `timestamp`, `news_override`, `sentiment_override`, `onchain_override`, `skip_freshness` optional params to `assemble_context`; backward compatible (all default to `None`/`False`)
- `src/validation.py` — added `skip_freshness: bool = False` to `validate_context`; when `True`, skips price freshness check (used by backtest engine)
- `tests/test_assembler.py` — updated to cover new override parameters
- `tests/test_validation.py` — updated to cover `skip_freshness` flag

### Fixed
- `src/backtest/data.py` — `fetch_full_ohlcv` now correctly handles timezone-aware `end_date` arguments using `tz_localize`/`tz_convert` instead of `pd.Timestamp(dt, tz=...)` which raises on already-aware datetimes
- `tests/test_backtest.py` — `_run_strategy` default cash raised from $10k to $1M so 20% position sizing can purchase at least one whole BTC unit at realistic price levels ($40k–$50k) without backtesting.py silently canceling orders

## v0 Phase 4 — 2026-04-05

### Added
- `src/db/schema.py` — added `weekly_summaries` table (`window_start`, `window_end`, `summary`, `metrics_json`)
- `src/db/store.py` — added `get_closed_trades_in_window`, `get_reflections_for_trades`, `get_cycles_in_window`, `save_weekly_summary`, `get_weekly_summaries`
- `src/reporting/__init__.py` — new reporting package
- `src/reporting/metrics.py` — `compute_sharpe_ratio` (annualised, sqrt(252)), `compute_max_drawdown` (equity-curve walk), `compute_expectancy` (avg_win × win_rate − avg_loss × loss_rate), `compute_council_consistency` (unanimous rate, confidence spread, veto rate), `compute_all_metrics` (all four, with target-met flags)
- `src/reporting/weekly.py` — `run_weekly_summary`: loads week's trades + reflections + cycles, computes metrics, calls claude-sonnet-4-6 with `weekly_summary_v1.txt`, persists summary to DB; runnable as `python -m src.reporting.weekly`
- `prompts/weekly_summary_v1.txt` — system prompt for the weekly summary agent (5 structured sections: performance verdict, pattern analysis, agent accuracy ranking, prompt refinement recommendations, next-period watchpoints)
- `src/main.py` — `TRADING_MODE` wiring (`paper` = testnet, `live` = mainnet with `COUNCIL_LIVE_CONFIRMED=1` safety gate); `--weekly` CLI flag; weekly APScheduler job (Sunday 08:00 UTC)
- `.env.example` — documented `TRADING_MODE`, `COUNCIL_LIVE_CONFIRMED`, `BINANCE_API_KEY/SECRET`, `COUNCIL_DB_PATH`, `DRY_RUN`
- `tests/test_reporting.py` — 49 tests: Sharpe, max drawdown, expectancy, consistency, all-metrics, weekly message builder, `run_weekly_summary`, store windowed queries

## v0 Phase 3 — 2026-04-05

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

## v0 Phase 2 — Agent Framework

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

## v0 Phase 1 — Data Pipeline + Tier-0 Validation

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
