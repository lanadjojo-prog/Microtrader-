from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import Settings
from ctrader_client import CTraderClient
from forex_backtest import (
    ForexCandidate,
    candidate_grid,
    candidate_signature,
    evaluate_candidate,
)
from forex_store import ForexStrategyStore

log = logging.getLogger("microtrader.forex_lab")


@dataclass
class ForexLabState:
    running: bool = False
    stage: str = "idle"
    message: str = "Waiting for cTrader credentials"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_error: Optional[str] = None
    generation: int = 0
    tested_total: int = 0
    promoted_total: int = 0
    progress: int = 0
    total: int = 0
    current_candidate: str = ""
    current_params: dict | None = None
    current_pair: str = ""
    pairs_loaded: int = 0
    pairs_total: int = 0
    last_completed_candidate: str = ""
    last_completed_at: Optional[str] = None


class ForexStrategyLab:
    """CPU-conscious forex strategy research.

    It loads one 1-minute source dataset per pair, then aggregates locally for
    5m/15m candidates. This prevents repeated broker downloads and lets the
    small Render instance spend its CPU on strategy evaluation instead.
    """

    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.store = ForexStrategyStore(settings.database_url)
        self.state = ForexLabState(pairs_total=len(settings.forex_pairs))
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._bars_1m: Dict[str, List[dict]] = {}

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["broker"] = self.client.public_state()
        payload["pairs"] = self.settings.forex_pairs
        payload["start_capital_eur"] = self.settings.forex_start_capital
        payload["risk_eur"] = self.settings.forex_risk_eur
        payload["cost_bps_per_side"] = self.settings.forex_cost_bps
        payload["results_loaded"] = len(self._results)
        return payload

    def results(self) -> List[dict]:
        return list(self._results)

    async def start(self) -> None:
        if self.state.running:
            return
        if not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = (
                "Forex Lab is built but cTrader credentials/access token are not configured yet."
            )
            return

        self.state = ForexLabState(
            running=True,
            stage="starting",
            message="Preparing Forex Strategy Lab",
            started_at=datetime.now(timezone.utc).isoformat(),
            pairs_total=len(self.settings.forex_pairs),
        )
        self._task = asyncio.create_task(self._run(), name="microtrader-forex-lab")

    async def stop(self) -> None:
        self.state.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _load_source_data(self) -> Dict[str, List[dict]]:
        if self._bars_1m:
            return self._bars_1m
        self.state.stage = "loading_data"
        self.state.message = "Loading cTrader 1-minute FX history"
        await self.client.connect_and_authenticate()

        bars_by_pair: Dict[str, List[dict]] = {}
        for pair in self.settings.forex_pairs:
            self.state.current_pair = pair
            self.state.message = f"Loading {pair}"
            bars = await self.client.historical_bars(
                pair,
                timeframe_min=1,
                max_bars=self.settings.forex_max_bars_per_pair,
                lookback_days=self.settings.forex_lookback_days,
            )
            if len(bars) >= 300:
                bars_by_pair[pair] = bars
            self.state.pairs_loaded += 1
            await asyncio.sleep(0)
        self.state.current_pair = ""
        if not bars_by_pair:
            raise RuntimeError("No usable cTrader FX bars were returned")
        self._bars_1m = bars_by_pair
        return bars_by_pair

    async def _run(self) -> None:
        try:
            await self.store.init()
            state = await self.store.load_state()
            self.state.generation = int(state.get("generation", 0))
            self.state.tested_total = int(state.get("tested_total", 0))
            self.state.promoted_total = int(state.get("promoted_total", 0))
            self._results = await self.store.load_results(limit=250)
            seen = await self.store.load_signatures()

            source = await self._load_source_data()
            configured_candidates = [
                ForexCandidate(
                    candidate.strategy,
                    {**candidate.params, "risk_eur": self.settings.forex_risk_eur},
                )
                for candidate in candidate_grid()
            ]
            all_candidates = [
                candidate for candidate in configured_candidates
                if evaluation_signature(candidate, source) not in seen
            ]
            batch_size = max(1, self.settings.forex_lab_batch_size)
            batch = all_candidates[:batch_size]

            if not batch:
                self.state.stage = "search_exhausted"
                self.state.message = "All configured forex candidates have been tested."
                return

            self.state.generation += 1
            self.state.stage = "testing"
            self.state.total = len(batch)
            self.state.progress = 0

            for idx, candidate in enumerate(batch, start=1):
                if not self.state.running:
                    break
                tf = int(candidate.params.get("timeframe_min", 1))
                bars = {
                    pair: aggregate_bars(pair_bars, tf)
                    for pair, pair_bars in source.items()
                }
                self.state.current_candidate = candidate.strategy
                self.state.current_params = dict(candidate.params)
                self.state.message = (
                    f"Testing {idx}/{len(batch)} · {candidate.strategy} · "
                    f"{tf}m · target {candidate.params.get('target_r')}R"
                )

                result = await asyncio.to_thread(
                    evaluate_candidate,
                    candidate,
                    bars,
                    cost_bps=self.settings.forex_cost_bps,
                    stress_multiplier=self.settings.forex_stress_cost_multiplier,
                    min_oos_trades=self.settings.forex_min_oos_trades,
                    min_profit_factor=self.settings.forex_min_profit_factor,
                    min_payoff_ratio=self.settings.forex_min_payoff_ratio,
                    start_capital=self.settings.forex_start_capital,
                )
                signature = evaluation_signature(candidate, source)
                result["dataset"]["version"] = dataset_version(source)
                run_id = await self.store.save_run(signature, result)
                if run_id:
                    result["run_id"] = run_id

                self.state.tested_total += 1
                if result.get("promoted"):
                    self.state.promoted_total += 1
                self.state.progress = idx
                self.state.last_completed_candidate = candidate.strategy
                self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                self._results.append(result)
                self._results.sort(
                    key=lambda row: (
                        bool(row.get("promoted")),
                        float(row.get("funnel_score") or 0),
                        float((row.get("oos") or {}).get("expectancy_r") or 0),
                    ),
                    reverse=True,
                )
                self._results = self._results[:250]
                await self.store.save_state(
                    self.state.generation,
                    self.state.tested_total,
                    self.state.promoted_total,
                )

                m = result.get("oos") or {}
                log.info(
                    "Forex candidate complete: strategy=%s tf=%sm target_r=%s "
                    "stage=%s score=%s trades=%s trades_day=%s win=%s pf=%s "
                    "payoff=%s exp_r=%s avg_win_r=%s avg_loss_r=%s best_r=%s worst_r=%s",
                    candidate.strategy,
                    tf,
                    candidate.params.get("target_r"),
                    result.get("funnel_stage"),
                    result.get("funnel_score"),
                    m.get("trades"),
                    m.get("avg_trades_per_day"),
                    m.get("win_rate_pct"),
                    m.get("profit_factor"),
                    m.get("payoff_ratio"),
                    m.get("expectancy_r"),
                    m.get("avg_win_r"),
                    m.get("avg_loss_r"),
                    m.get("best_trade_r"),
                    m.get("worst_trade_r"),
                )
                await asyncio.sleep(0.05)

            self.state.stage = "batch_complete"
            self.state.message = (
                f"Forex batch complete: {self.state.progress}/{self.state.total} tested. "
                "Every result is stored append-only per strategy."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Forex Strategy Lab failed")
        finally:
            self.state.running = False
            self.state.current_candidate = ""
            self.state.current_params = None
            self.state.current_pair = ""
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


def dataset_version(bars_by_pair: Dict[str, List[dict]]) -> str:
    payload = {
        pair: {
            "count": len(bars),
            "first": str(bars[0].get("t") or "") if bars else "",
            "last": str(bars[-1].get("t") or "") if bars else "",
        }
        for pair, bars in sorted(bars_by_pair.items())
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluation_signature(
    candidate: ForexCandidate,
    bars_by_pair: Dict[str, List[dict]],
) -> str:
    raw = candidate_signature(candidate) + ":" + dataset_version(bars_by_pair)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def aggregate_bars(bars: List[dict], minutes: int) -> List[dict]:
    if minutes <= 1:
        return list(bars)

    buckets: Dict[str, List[dict]] = {}
    for bar in bars:
        raw = str(bar.get("t") or "")
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            continue
        minute = (dt.minute // minutes) * minutes
        key_dt = dt.replace(minute=minute, second=0, microsecond=0)
        key = key_dt.isoformat()
        buckets.setdefault(key, []).append(bar)

    out: List[dict] = []
    for key in sorted(buckets):
        group = buckets[key]
        if not group:
            continue
        out.append({
            "t": key,
            "o": float(group[0]["o"]),
            "h": max(float(x["h"]) for x in group),
            "l": min(float(x["l"]) for x in group),
            "c": float(group[-1]["c"]),
            "v": sum(float(x.get("v") or 0) for x in group),
        })
    return out
