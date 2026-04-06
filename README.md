# Council — BTC LLM Trading Bot

A Bitcoin swing trading bot that uses a council of specialised LLM agents to generate daily BUY/SELL/HOLD signals and execute them against Binance testnet.

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
Execution Layer (Binance testnet) + SQLite Logging
    │
    └── Reflection Loop (post-trade analysis)
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

# Dry run (logs only, no orders placed)
DRY_RUN=1 python -m src.main

# Daily scheduler (00:05 UTC)
python -m src.main --schedule
```

## Testing

```bash
pytest -v                                    # all tests
pytest tests/test_db.py -v                  # DB layer
pytest tests/test_execution.py -v           # execution + router
pytest tests/test_agents.py -v              # agent framework
pytest tests/test_agents.py::test_run_council_hold_on_veto -v  # single test
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `BINANCE_TESTNET_API_KEY` | For execution | — | Binance testnet key |
| `BINANCE_TESTNET_SECRET` | For execution | — | Binance testnet secret |
| `COUNCIL_DB_PATH` | No | `./council.db` | SQLite database path |
| `INITIAL_CAPITAL` | No | `10000` | Starting paper capital (USD) |
| `DRY_RUN` | No | `0` | Set to `1` to skip order execution |
| `CRYPTOPANIC_API_KEY` | No | — | News feed (falls back to empty list) |
| `LUNARCRUSH_API_KEY` | No | — | Sentiment (falls back to Fear/Greed proxy) |
| `GLASSNODE_API_KEY` | No | — | On-chain data (falls back to zeros) |

## Development Phases

- **Phase 1** ✅ Data pipeline + Tier-0 validation
- **Phase 2** ✅ Agent framework (4 agents + deliberation)
- **Phase 3** ✅ Execution layer + SQLite logging + reflection loop
- **Phase 4** Paper trading (60 days minimum)
- **Phase 5** Live deployment (≤ $500 initial capital)
