# Council — BTC LLM Trading Bot

A Bitcoin swing trading framework built around a **council of LLM agents** that vote on daily BUY/SELL/HOLD signals. Fully configurable — define your own agents, weights, and risk rules in a single YAML file without writing code.

## Architecture

```
strategy.yaml  (agents, weights, SL/TP rules, validation thresholds)
    │
    ▼
Data Pipeline (CCXT/Kraken, GDELT news, Alternative.me, CoinMetrics, yfinance)
    │
    ▼
Tier-0 Validation (price freshness, news count, drawdown halt, ATR halt)
    │                          ↑ thresholds from strategy.yaml
    ▼
Generic Council Runner — agents defined in strategy.yaml, run in parallel
    ├── Directional agents  (any number, any model, user-defined prompts)
    └── Veto agent          (optional; forces HOLD if veto=true)
    │
    ▼
Deterministic Scorer (confidence-weighted signed-score from strategy weights)
    │
    ▼
Optional Deliberation (narrative synthesis, claude-sonnet-4-6) → BUY / SELL / HOLD
    │
    ▼
Execution Layer (Kraken paper/live) + SQLite Logging
    │
    ├── Reflection Loop (post-trade analysis, claude-sonnet-4-6)
    └── Weekly Summary  (performance review, claude-sonnet-4-6)
```

The default strategy (`strategies/default.yaml`) reproduces the original v2 council: Technical (40%) + Sentiment (25%) + Fundamental (35%) + Risk veto.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
```

## Custom Strategies (v3)

Define a strategy as a YAML file — no Python required.

```yaml
# strategies/my_strategy.yaml
name: "My Momentum Strategy"
agents:
  - name: trend
    model: claude-haiku-4-5
    prompt: prompts/agent_template.txt   # copy & customise
    weight: 0.60
  - name: macro
    model: claude-haiku-4-5
    prompt: prompts/agent_template.txt
    weight: 0.40
  - name: risk_guard
    model: claude-sonnet-4-6
    prompt: prompts/veto_agent_template.txt
    is_veto: true
scoring:
  confidence_floor: 30
  trade_threshold: 0.25
risk:
  stop_loss:
    type: atr_multiple
    value: 2.0
  take_profit:
    type: fixed_pct
    value: 0.08
  max_position_pct: 15.0
validation:
  max_drawdown_pct: 12.0
  atr_spike_multiplier: 2.5
```

Start from `strategies/example.yaml` (fully commented) and `prompts/agent_template.txt` / `prompts/veto_agent_template.txt`.

## Running

```bash
# One-shot cycle — default v2 council
python -m src.main

# One-shot cycle — custom strategy
python -m src.main --strategy strategies/my_strategy.yaml

# Dry run (logs council decision, no orders placed)
DRY_RUN=1 python -m src.main
DRY_RUN=1 python -m src.main --strategy strategies/my_strategy.yaml

# Daily scheduler (00:05 UTC) + weekly summary (Sunday 08:00 UTC)
python -m src.main --schedule

# Weekly summary report only
python -m src.main --weekly
```

## Backtesting

Run the LLM council against historical BTC data without executing real orders:

```bash
# Backtest over 2024 — default council
python -m src.backtest.engine \
    --start 2024-01-01 \
    --end   2025-01-01 \
    --capital 250000

# Backtest with a custom strategy
python -m src.backtest.engine \
    --start 2023-01-01 \
    --end   2024-01-01 \
    --capital 250000 \
    --strategy strategies/my_strategy.yaml

# Quick smoke test
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
pytest -v                                    # all 344 tests
pytest tests/test_db.py -v                  # DB layer
pytest tests/test_execution.py -v           # execution + router
pytest tests/test_reporting.py -v           # metrics + weekly summary
pytest tests/test_agents.py -v              # agent framework
pytest tests/test_backtest.py -v            # backtesting module
pytest tests/test_agents.py::test_run_council_hold_on_veto -v  # single test
pytest tests/test_strategy_loader.py -v     # strategy YAML loader
pytest tests/test_strategy_scoring.py -v   # generic scoring + SL/TP rules
pytest tests/test_strategy_runner.py -v    # generic council runner
pytest tests/test_strategy_parity.py -v    # legacy vs generic scorer parity
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

