# Council — BTC LLM Trading Bot

A Bitcoin swing trading bot that uses a council of specialised LLM agents to generate daily BUY/SELL/HOLD signals and execute them against Binance testnet (paper) or mainnet (live).

## Architecture

```
Data Pipeline (CCXT/Binance, CryptoPanic, Alternative.me, Glassnode)
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

## Testing

```bash
pytest -v                                    # all 203 tests
pytest tests/test_db.py -v                  # DB layer
pytest tests/test_execution.py -v           # execution + router
pytest tests/test_reporting.py -v           # metrics + weekly summary
pytest tests/test_agents.py -v              # agent framework
pytest tests/test_agents.py::test_run_council_hold_on_veto -v  # single test
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `BINANCE_TESTNET_API_KEY` | Paper mode | — | Binance testnet key |
| `BINANCE_TESTNET_SECRET` | Paper mode | — | Binance testnet secret |
| `BINANCE_API_KEY` | Live mode | — | Binance mainnet key |
| `BINANCE_SECRET` | Live mode | — | Binance mainnet secret |
| `COUNCIL_DB_PATH` | No | `./council.db` | SQLite database path |
| `INITIAL_CAPITAL` | No | `10000` | Starting paper capital (USD) |
| `TRADING_MODE` | No | `paper` | `paper` (testnet) or `live` (mainnet) |
| `COUNCIL_LIVE_CONFIRMED` | Live mode | — | Must be `1` to enable live trading |
| `DRY_RUN` | No | `0` | Set to `1` to skip order execution entirely |
| `CRYPTOPANIC_API_KEY` | No | — | News feed (falls back to empty list) |
| `LUNARCRUSH_API_KEY` | No | — | Sentiment (falls back to Fear/Greed proxy) |
| `GLASSNODE_API_KEY` | No | — | On-chain data (falls back to zeros) |

## Development Phases

- **Phase 1** ✅ Data pipeline + Tier-0 validation
- **Phase 2** ✅ Agent framework (4 agents + deliberation)
- **Phase 3** ✅ Execution layer + SQLite logging + reflection loop
- **Phase 4** ✅ Performance metrics + weekly summary agent
- **Phase 5** Paper trading (60 days minimum on live data with simulated capital)
- **Phase 6** Live deployment (≤ $500 initial capital)

## Success Targets (Phase 5 evaluation)

| Metric | Target |
|---|---|
| Sharpe ratio | ≥ 1.5 |
| Max drawdown | < 20% |
| Expectancy | Positive |
| Council consistency | ≥ 40% unanimous cycles |
