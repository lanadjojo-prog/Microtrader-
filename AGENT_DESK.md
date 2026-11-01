# MicroTrader · Five-agent forex tester

Open `/desk` and use the existing dashboard token. This is a **forward paper
tester with its own virtual ledger**, not a cTrader broker-order bot. cTrader
remains read-only/demo. It does not read or modify the existing Paper engine's
capital, strategies or positions.

## Frozen v1 policy

- EUR/USD and GBP/USD. Closed M1 candles aggregated into complete M5 candles.
- M5 EMA50 trend: close above/below EMA50 and EMA50 rising/falling vs 3 M5 bars ago.
- Entry candidates on M1 and M5: close beyond the prior 20-bar high/low, or
  a candle touching EMA20 and closing back on the trend side with matching candle direction.
- Stop: three-bar extreme plus 0.1 ATR14 buffer. Distance must be 0.7–3 ATR.
- Five LLM agents: Scout, Charts, Risk, Sceptic, Chief. All assess independently,
  then all five reconsider using all first-round opinions. Final unanimity is
  mandatory. Any failed provider call in either round vetoes the candidate.
- No changing a strategy in flight, no random/default trade if consensus fails.
- Fresh post-debate bid/ask is required. Setup expiry is 90 seconds after its
  candle closes. A move beyond 0.1 ATR from the approved fill rejects execution.
- Requested risk budget = 10% of current virtual balance. Order size rounds
  **down** to broker min/step/max and is capped by modelled margin; actual risk
  may be substantially lower. Commission and modelled stop slippage are inside R.
- Target is designed for **2R net**, after modelled spread/slippage/commission.
- One open position across both pairs/timeframes/sessions. At most two entries
  per trading day. Stop new entries after one >=2R winning trade or two losses.
  A loss followed by a 2R winner also ends the day; daily aggregate need not be +2R.
- Trading day = Europe/London calendar date. London window 08:00–17:00
  Europe/London, New York 08:00–17:00 America/New_York, Monday–Friday.
  DST follows each zone independently. Daily limits span BOTH sessions.
- At the end of the combined session window the paper position is closed at
  the next available quote. There is no guaranteed fill at the scheduled time
  during downtime or a feed failure. No trade is forced on quiet days.
- Pause disables new entries but keeps existing paper positions managed.

## Activation

Existing runtime needs `DATABASE_URL` (or independent `DESK_DATABASE_URL`), `DASHBOARD_TOKEN`, cTrader client
credentials and a completed demo OAuth authorization. These remain secret
environment settings and must never be committed to this public repository.

Add `OPENAI_API_KEY` in the service's secret environment settings. Model defaults
to `gpt-4.1-mini`; override with `DESK_LLM_MODEL` if needed. These are real
provider calls, billed separately. There is no fake-AI fallback.

The new tester defaults to **entries paused**. Click "Tester starten" when the
status shows no blockers. State, daily counters, decisions and trades persist
in PostgreSQL. `DESK_DATABASE_URL` isolates the new ledger from an unavailable
legacy research database; new cTrader OAuth tokens are saved there as well.
The old database and research records are not replaced. For explicit restart activation set `DESK_AUTO_START=true`.
Do not enable auto-start if you need a manual pause to persist across restarts.

Optional model assumptions (verify against the actual broker account before
interpreting results):

| Environment setting | Default |
|---|---|
| `DESK_COMMISSION_USD_PER_LOT` | 4.5 round turn per 100,000 base units |
| `DESK_SLIPPAGE_PIPS` | 0.2 on each fill |
| `DESK_MAX_SPREAD_PIPS` | 1.5 |
| `DESK_MODEL_LEVERAGE` | 30 |
| `DESK_MAX_DEBATES_PER_DAY` | 12 (up to 120 LLM calls/day) |

The start balance for a NEW desk ledger is EUR 50. It is not read from the
broker or reset every day. Changing code does not reset persisted balance.
The export endpoint includes the complete append-only audit journal; the
dashboard holds the latest 200 trades and 30 debates for bounded state size.

## Execution-model limits

Quote checks run approximately every five seconds except during retrieval or
agent debate. Completed M1 OHLC candles are used to recover missed touches.
Ask OHLC is modelled from bid OHLC using the larger of entry/current spread,
not actual historical ask extremes. If both stop and target touch within one
candle, stop wins. For a partial entry candle, pre-entry highs cannot establish
a win: a stop touch is conservatively counted and tagged
`stop_entry_ambiguous`. Such trades require separate analysis. This is not
tick-accurate brokerage simulation and not proof of a scalping edge.

Stops can exceed -1R on gaps; the ledger records this rather than clipping it.
Currency conversion uses the current EUR/USD quote. A target is 2R under the
entry conversion assumption; actual reported EUR R can differ slightly as FX
moves. Target hits end the day regardless of that conversion difference.

No current news/calendar feed is supplied in v1. Agents are explicitly forbidden
to invent news or infer a measured win probability from their own confidence.
No historical performance or forward profitability has yet been established.

## Reliability and operation

An expiring PostgreSQL lease and compare-and-swap state write prevent overlapping
workers from committing duplicate paper entries. State and journal commit in
one transaction. A pause revokes in-flight leases. Credentials/provider exception
messages are not exposed through public status. All state/control/export API
routes require the existing dashboard Bearer token. RLS is enabled on the two
new tables and no public policies are granted.

This process needs continuously running hosting and a writable database. A free
web service that sleeps cannot reliably scan sessions or manage M1 positions.
Do not infer daily operation merely because the page is online. Check last-cycle
time, enabled state, blockers and journal. No hosting plan is changed by this code.

Validation: `python -m unittest discover -s tests -v` and
`python -m compileall -q .`. Tests use controlled data/provider doubles; they do
not represent actual LLM decisions, broker fills or profitable results.
