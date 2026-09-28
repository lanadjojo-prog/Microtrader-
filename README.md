# MicroTrader

A small, Render-ready automated trading engine for Alpaca. It is deliberately safe-by-default: **paper mode is on, live trading is off, and the engine does not auto-start**.

## What it does

- Scans a configurable list of liquid US stocks every few seconds.
- Uses 1-minute bars and a simple fast/slow moving-average momentum strategy.
- Opens tiny fractional market positions by dollar notional.
- Closes positions on signal fade, take-profit, or stop-loss.
- Enforces maximum open positions, maximum trades/day, per-trade notional, cooldowns, and a daily-loss kill switch.
- Provides a password-protected dashboard/API for status, start, stop, emergency flatten, and recent executions.
- Stores recent signal price versus actual fill in memory and calculates adverse slippage in basis points.
- Never commits broker credentials.

## Important

The included strategy is a **technical starter strategy, not a claim of profitability**. Before live use, measure spread, slippage, fills, rejected orders, and net expectancy. Paper execution does not reproduce every live-market effect.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Export the variables from .env or configure them in your shell.
uvicorn app:app --reload
```

Open http://localhost:8000 and enter `DASHBOARD_TOKEN`.

## Render

Create a Python Web Service from this repo:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Region: Frankfurt is a sensible default for a Netherlands-based operator.

Add these secrets directly in Render Environment:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `DASHBOARD_TOKEN`

Start with:

- `ALPACA_PAPER=true`
- `LIVE_TRADING_ENABLED=false`
- `AUTO_START=false`

When live credentials are eventually used, `ALPACA_PAPER=false` alone is **not enough** to send live orders. `LIVE_TRADING_ENABLED=true` is also required.

## Default risk settings

- $3 notional per entry
- 5 open positions max
- 100 submitted trades/day max
- $5 daily equity loss kill switch
- 0.4% take-profit
- 0.3% stop-loss
- 60-second symbol cooldown

Every value is configurable with environment variables in `.env.example`.

## Endpoints

- `GET /health` — public health check
- `GET /api/status` — protected
- `GET /api/executions` — protected; recent order/fill/slippage records
- `POST /api/start` — protected
- `POST /api/stop` — protected
- `POST /api/flatten` — protected, closes all positions and cancels open orders

Protected calls use `Authorization: Bearer <DASHBOARD_TOKEN>`.

## Next engineering upgrades

For serious evaluation, add persistent PostgreSQL execution logging, theoretical-vs-fill slippage tracking, bid/ask spread filters, websocket quotes/trade updates, strategy backtesting, and a dedicated order-state machine.

## Strategy Lab

MicroTrader v0.2 adds a protected Strategy Lab for research before paper/live execution.

- 10 initial candidate configurations across momentum and mean-reversion families.
- Chronological 70/30 train/out-of-sample split.
- Next-bar-open fills to avoid same-bar look-ahead.
- Configurable round-trip transaction-cost assumptions.
- A stressed-cost pass with a higher friction assumption.
- Automatic rejection when out-of-sample expectancy is non-positive, trade count is too small, profit factor is weak, drawdown is excessive, or the stressed-cost result fails.
- The lab does **not** automatically enable live trading or promote a result into the live engine.

Dashboard controls are available under **Strategy Lab**. API endpoints: `GET /api/lab/status`, `GET /api/lab/results`, `POST /api/lab/start`, and `POST /api/lab/stop`.
