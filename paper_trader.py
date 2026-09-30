from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from config import Settings
from ctrader_client import CTraderClient
from forex_store import ForexStrategyStore
from paper_store import PaperTradingStore

log = logging.getLogger("microtrader.paper")


@dataclass
class PaperTradingState:
    running: bool = False
    stage: str = "idle"
    message: str = "Waiting for promoted strategies"
    strategies: int = 0
    last_cycle_at: Optional[str] = None
    last_error: Optional[str] = None


def _paper_id(row: dict) -> str:
    params = {k: v for k, v in dict(row.get("params") or {}).items() if k != "_phase"}
    raw = json.dumps(
        {
            "strategy": row.get("strategy"),
            "params": params,
            "pairs": sorted(row.get("pairs") or []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "paper-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _dt(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals: List[float] = []
    for j in range(max(1, i - window), i):
        high = float(bars[j]["h"])
        low = float(bars[j]["l"])
        prev = float(bars[j - 1]["c"])
        vals.append(max(high - low, abs(high - prev), abs(low - prev)))
    return mean(vals) if vals else 0.0


def _exit_management(params: dict) -> tuple[float | None, float | None]:
    mode = str(params.get("exit_mode") or "baseline")
    if mode == "breakeven_2r":
        return 2.0, 0.0
    if mode == "protect_2r_025r":
        return 2.0, 0.25
    if mode == "lock_2r_05r":
        return 2.0, 0.50
    return None, None


def _signal(strategy: str, params: dict, bars: List[dict], i: int) -> int:
    if i < 2:
        return 0
    if strategy == "range_reversal":
        window = int(params.get("window", 30))
        if i < window:
            return 0
        closes = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(closes)
        sigma = pstdev(closes)
        if sigma <= 0:
            return 0
        z = (closes[-1] - mu) / sigma
        z_entry = float(params.get("z_entry", 1.8))
        return 1 if z <= -z_entry else (-1 if z >= z_entry else 0)

    if strategy == "trend_pullback":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 30))
        if i < slow:
            return 0
        closes = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(closes[-fast:])
        slow_ma = mean(closes)
        prev_close = closes[-1]
        if fast_ma > slow_ma and slow_ma < prev_close <= fast_ma:
            return 1
        if fast_ma < slow_ma and fast_ma <= prev_close < slow_ma:
            return -1
        return 0

    if strategy == "asymmetric_breakout":
        window = int(params.get("window", 20))
        if i < window + 2:
            return 0
        prior = bars[i - window - 1:i - 1]
        prev_close = float(bars[i - 1]["c"])
        high_break = max(float(x["h"]) for x in prior)
        low_break = min(float(x["l"]) for x in prior)
        return 1 if prev_close > high_break else (-1 if prev_close < low_break else 0)

    return 0


class PaperTradingEngine:
    """Forward-only simulated execution for promoted Forex strategies.

    It never sends broker orders. New promoted configurations are frozen and
    tracked from EUR 50 using subsequent cTrader market bars only.
    """

    SUPPORTED = {"range_reversal", "trend_pullback", "asymmetric_breakout"}

    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.research_store = ForexStrategyStore(settings.database_url)
        self.store = PaperTradingStore(settings.database_url)
        self.state = PaperTradingState()
        self._task: Optional[asyncio.Task] = None

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["start_balance_eur"] = self.settings.paper_start_balance
        payload["poll_seconds"] = self.settings.paper_poll_seconds
        payload["execution"] = "simulated-only; no broker orders"
        return payload

    async def start(self) -> None:
        if self.state.running:
            return
        if not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = "Paper trading waits for cTrader demo market-data authorization."
            return
        self.state = PaperTradingState(
            running=True,
            stage="starting",
            message="Starting promoted-strategy paper trading",
        )
        self._task = asyncio.create_task(self._run(), name="microtrader-paper")

    async def stop(self) -> None:
        self.state.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def results(self) -> dict:
        return {
            "state": self.public_state(),
            **(await self.store.dashboard()),
        }

    async def _discover(self) -> List[dict]:
        promoted = await self.research_store.load_promoted(limit=100)
        for row in promoted:
            strategy = str(row.get("strategy") or "")
            if strategy not in self.SUPPORTED:
                continue
            base_params = {
                k: v for k, v in dict(row.get("params") or {}).items()
                if k != "_phase"
            }
            # Older promoted runs predate exit-management testing. Keep their
            # frozen baseline untouched and add a simultaneous +2R -> +0.25R
            # protection variant for clean forward comparison.
            if "exit_mode" in base_params:
                variants = [base_params]
            else:
                variants = [
                    dict(base_params),
                    {**base_params, "exit_mode": "protect_2r_025r"},
                ]

            for params in variants:
                variant_row = {**row, "params": params}
                await self.store.ensure_strategy(
                    paper_id=_paper_id(variant_row),
                    promoted_run_id=str(row.get("run_id") or ""),
                    strategy=strategy,
                    params=params,
                    pairs=list(row.get("pairs") or self.settings.forex_pairs),
                    timeframe_min=int(
                        row.get("timeframe_min")
                        or params.get("timeframe_min")
                        or 1
                    ),
                    start_balance=float(self.settings.paper_start_balance),
                )
        strategies = await self.store.list_strategies()
        self.state.strategies = len(strategies)
        return strategies

    async def _run(self) -> None:
        try:
            await self.store.init()
            await self.research_store.init()
            while self.state.running:
                try:
                    strategies = await self._discover()
                    if not strategies:
                        self.state.stage = "waiting_promoted"
                        self.state.message = "No promoted strategies yet."
                    else:
                        self.state.stage = "paper_trading"
                        self.state.message = (
                            f"Paper trading {len(strategies)} frozen promoted "
                            f"strateg{'y' if len(strategies)==1 else 'ies'} from €{self.settings.paper_start_balance:.0f}."
                        )
                        for row in strategies:
                            if not self.state.running:
                                break
                            if str(row.get("status")) == "ruined":
                                continue
                            try:
                                await self._process_strategy(row)
                            except Exception as exc:
                                await self.store.set_error(str(row["paper_id"]), str(exc))
                                log.exception("Paper strategy failed: %s", row.get("paper_id"))
                    self.state.last_error = None
                    self.state.last_cycle_at = datetime.now(timezone.utc).isoformat()
                except Exception as exc:
                    self.state.stage = "error"
                    self.state.last_error = str(exc)
                    self.state.message = str(exc)
                    log.exception("Paper trading cycle failed")
                await asyncio.sleep(max(30, int(self.settings.paper_poll_seconds)))
        except asyncio.CancelledError:
            raise
        finally:
            self.state.running = False

    async def _process_strategy(self, row: dict) -> None:
        paper_id = str(row["paper_id"])
        strategy = str(row["strategy"])
        params = dict(row.get("params") or {})
        pairs = list(row.get("pairs") or self.settings.forex_pairs)
        tf = int(row.get("timeframe_min") or params.get("timeframe_min") or 1)
        cursors = dict(row.get("last_bar_times") or {})
        positions = {p["pair"]: p for p in await self.store.list_positions(paper_id)}

        for pair in pairs:
            bars = await self.client.historical_bars(
                pair,
                timeframe_min=tf,
                max_bars=600,
                lookback_days=7,
            )
            if len(bars) < 50:
                continue

            # Only bars whose interval is fully closed are eligible.
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=tf)
            closed = [b for b in bars if _dt(b["t"]) <= cutoff]
            if len(closed) < 40:
                continue

            cursor = str(cursors.get(pair) or "")
            if not cursor:
                # Strict forward start: do not retroactively paper-trade data
                # from before this strategy was enrolled.
                await self.store.set_cursor(paper_id, pair, str(closed[-1]["t"]))
                cursors[pair] = str(closed[-1]["t"])
                continue

            new_indices = [
                i for i, bar in enumerate(closed)
                if str(bar.get("t") or "") > cursor
            ]
            for i in new_indices:
                bar = closed[i]
                pos = positions.get(pair)
                if pos:
                    exited = await self._manage_existing(
                        paper_id, pair, pos, bar, params
                    )
                    if exited:
                        positions.pop(pair, None)
                    else:
                        pos = dict(pos)
                        pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
                        await self.store.upsert_position(paper_id, pair, pos)
                        positions[pair] = pos
                else:
                    direction = _signal(strategy, params, closed, i)
                    if direction:
                        atr = _atr(closed, i, 14)
                        risk_distance = atr * float(params.get("stop_atr", 1.0))
                        if risk_distance > 0:
                            entry = float(bar["o"])
                            target_r = float(params.get("target_r", 2.0))
                            risk_eur = float(params.get("risk_eur", 2.0))
                            if direction > 0:
                                stop = entry - risk_distance
                                target = entry + risk_distance * target_r
                            else:
                                stop = entry + risk_distance
                                target = entry - risk_distance * target_r
                            new_pos = {
                                "direction": direction,
                                "entry_time": str(bar["t"]),
                                "entry_price": entry,
                                "risk_distance": risk_distance,
                                "stop_price": stop,
                                "target_price": target,
                                "bars_held": 0,
                                "risk_eur": risk_eur,
                            }
                            immediate = await self._manage_entry_bar(
                                paper_id, pair, new_pos, bar, params
                            )
                            if not immediate:
                                await self.store.upsert_position(
                                    paper_id, pair, new_pos
                                )
                                positions[pair] = new_pos

                await self.store.set_cursor(paper_id, pair, str(bar["t"]))
                cursors[pair] = str(bar["t"])

        await self.store.touch_daily(paper_id, len(positions))

    def _maybe_protect_stop(self, pos: dict, bar: dict, params: dict) -> bool:
        trigger_r, lock_r = _exit_management(params)
        if trigger_r is None or lock_r is None:
            return False
        direction = int(pos["direction"])
        entry = float(pos["entry_price"])
        risk_distance = float(pos["risk_distance"])
        current_stop = float(pos["stop_price"])
        protected_stop = (
            entry + risk_distance * lock_r
            if direction > 0
            else entry - risk_distance * lock_r
        )
        # Already at least this well protected.
        if direction > 0 and current_stop >= protected_stop:
            return False
        if direction < 0 and current_stop <= protected_stop:
            return False
        trigger_price = (
            entry + risk_distance * trigger_r
            if direction > 0
            else entry - risk_distance * trigger_r
        )
        high, low = float(bar["h"]), float(bar["l"])
        reached = high >= trigger_price if direction > 0 else low <= trigger_price
        if reached:
            pos["stop_price"] = protected_stop
            return True
        return False

    async def _manage_entry_bar(
        self, paper_id: str, pair: str, pos: dict, bar: dict, params: dict
    ) -> bool:
        direction = int(pos["direction"])
        low, high = float(bar["l"]), float(bar["h"])
        stop, target = float(pos["stop_price"]), float(pos["target_price"])
        if direction > 0 and low <= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction < 0 and high >= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction > 0 and high >= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        if direction < 0 and low <= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        # Protection earned on this candle becomes active from the next candle.
        self._maybe_protect_stop(pos, bar, params)
        return False

    async def _manage_existing(
        self, paper_id: str, pair: str, pos: dict, bar: dict, params: dict
    ) -> bool:
        direction = int(pos["direction"])
        low, high = float(bar["l"]), float(bar["h"])
        stop, target = float(pos["stop_price"]), float(pos["target_price"])
        if direction > 0 and low <= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction < 0 and high >= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction > 0 and high >= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        if direction < 0 and low <= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True

        # Same conservative rule as the backtest: a stop improvement earned
        # within this candle becomes active on the next candle.
        self._maybe_protect_stop(pos, bar, params)

        held = int(pos.get("bars_held", 0)) + 1
        if held >= int(params.get("max_hold", 36)):
            await self._close(
                paper_id, pair, pos, bar, float(bar["c"]), "max_hold"
            )
            return True
        return False

    async def _close(
        self,
        paper_id: str,
        pair: str,
        pos: dict,
        bar: dict,
        exit_price: float,
        reason: str,
    ) -> None:
        direction = int(pos["direction"])
        entry = float(pos["entry_price"])
        risk_distance = float(pos["risk_distance"])
        gross = direction * ((float(exit_price) / entry) - 1.0)
        net = gross - (2.0 * float(self.settings.forex_cost_bps) / 10_000.0)
        risk_pct = risk_distance / entry if entry > 0 else 0.0
        r_multiple = net / risk_pct if risk_pct > 0 else 0.0
        risk_eur = float(pos["risk_eur"])
        pnl = r_multiple * risk_eur
        await self.store.record_trade(
            paper_id=paper_id,
            pair=pair,
            side="long" if direction > 0 else "short",
            entry_time=str(pos["entry_time"]),
            exit_time=str(bar["t"]),
            entry_price=entry,
            exit_price=float(exit_price),
            exit_reason=reason,
            risk_eur=risk_eur,
            r_multiple=round(r_multiple, 6),
            pnl=round(pnl, 6),
        )
        await self.store.delete_position(paper_id, pair)
