# Council — BTC LLM Trading Bot

A Bitcoin swing trading bot that uses a council of specialised LLM agents to generate daily BUY/SELL/HOLD signals and execute them against Binance testnet (paper) or mainnet (live).

## Architecture

```
Data Pipeline (CCXT/Kraken, CryptoPanic, Alternative.me, Glassnode)
    │
    ▼
Tier-0 Validation (price freshness, news count, drawdown halt, ATR halt)
    │
    ▼
Council of LLMs (parallel)
    ├── Technical Analyst  (claude-haiku-4-5)
    ├── Sentiment Analyst  (claude-haiku-4-5)
    ├── Fundamental Analyst (claude-haiku-4-5)
    └── Risk Manager       (claude-sonnet-4-6)
    │
    ▼
Deliberation / Chair (claude-sonnet-4-6) → BUY / SELL / HOLD
    │
    ▼
Execution Layer (Binance testnet/mainnet) + SQLite Logging
    │
    ├── Reflection Loop (post-trade analysis, claude-sonnet-4-6)
    └── Weekly Summary  (performance review, claude-sonnet-4-6)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
```

## Running

```bash
# One-shot cycle (run once and exit)
python -m src.main

# Dry run (logs council decision, no orders placed)
DRY_RUN=1 python -m src.main

# Daily scheduler (00:05 UTC) + weekly summary (Sunday 08:00 UTC)
python -m src.main --schedule

# Weekly summary report only
python -m src.main --weekly
```

## Backtesting

Run the LLM council against historical BTC data without executing real orders:

```bash
# Backtest over 2024 (requires ANTHROPIC_API_KEY only — no Binance credentials needed)
python -m src.backtest.engine \
    --start 2024-01-01 \
    --end   2025-01-01 \
    --capital 250000

# Shorter date range for a quick smoke test
python -m src.backtest.engine \
    --start 2024-06-01 \
    --end   2024-09-01 \
    --capital 250000 \
    --no-save-csv
```

**Cost:** ~$10–15 in Anthropic API calls per year of daily cycles (~365 × ~$0.03).

**Capital note:** Set `--capital` high enough that 20% position sizing can buy at least 1 BTC. At ~$50k/BTC you need at least `--capital 250000`.

Each run creates a timestamped folder under `tmp/YYYYMMDD_HHMMSS/` containing:
- `backtest_signals_<start>_<end>.csv` — per-day signal log, written one row per cycle (flushed immediately) so it survives a kill
- `run.log` — full console output, written continuously (line-buffered) so it survives a kill

Output includes Return%, Sharpe, Max Drawdown, Win Rate, Avg Trade, and Expectancy.

## Testing

```bash
pytest -v                                    # all 235 tests
pytest tests/test_db.py -v                  # DB layer
pytest tests/test_execution.py -v           # execution + router
pytest tests/test_reporting.py -v           # metrics + weekly summary
pytest tests/test_agents.py -v              # agent framework
pytest tests/test_backtest.py -v            # backtesting module
pytest tests/test_agents.py::test_run_council_hold_on_veto -v  # single test
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `KRAKEN_API_KEY` | Live mode | — | Kraken API key |
| `KRAKEN_SECRET` | Live mode | — | Kraken private key |
| `COUNCIL_DB_PATH` | No | `./council.db` | SQLite database path |
| `INITIAL_CAPITAL` | No | `10000` | Starting paper capital (USD) |
| `TRADING_MODE` | No | `paper` | `paper` (testnet) or `live` (mainnet) |
| `COUNCIL_LIVE_CONFIRMED` | Live mode | — | Must be `1` to enable live trading |
| `DRY_RUN` | No | `0` | Set to `1` to skip order execution entirely |
| `CRYPTOPANIC_API_KEY` | No | — | News feed (falls back to empty list) |
| `LUNARCRUSH_API_KEY` | No | — | Sentiment (falls back to Fear/Greed proxy) |
| `GLASSNODE_API_KEY` | No | — | On-chain data (falls back to zeros) |

## Development Phases

### v0 — Live Trading Bot
- **Phase 1** ✅ Data pipeline + Tier-0 validation
- **Phase 2** ✅ Agent framework (4 agents + deliberation)
- **Phase 3** ✅ Execution layer + SQLite logging + reflection loop
- **Phase 4** ✅ Performance metrics + weekly summary agent
- **Phase 5** ✅ Exchange migration: Binance → Kraken (US-accessible, free)
- **Phase 6** Paper trading (60 days minimum, DRY_RUN=1)
- **Phase 7** Live deployment (≤ $500 initial capital, Kraken mainnet)

### v1 — Backtesting
- **Phase 1** ✅ Patch data layer (override params, `skip_freshness`)
- **Phase 2** ✅ Historical data module (`fetch_full_ohlcv`, `slice_ohlcv`, sentiment history)
- **Phase 3** ✅ Async signal generation loop (`generate_signals`, `_PortfolioTracker`)
- **Phase 4** ✅ `backtesting.py` strategy + CLI runner (`run_backtest`)
- **Phase 5** ✅ Exchange migration: Binance → Kraken
- **Phase 6** ✅ Historical data source: Kraken OHLCV → Yahoo Finance (yfinance; unlimited history)

## Success Targets (Phase 5 evaluation)

| Metric | Target |
|---|---|
| Sharpe ratio | ≥ 1.5 |
| Max drawdown | < 20% |
| Expectancy | Positive |
| Council consistency | ≥ 40% unanimous cycles |
