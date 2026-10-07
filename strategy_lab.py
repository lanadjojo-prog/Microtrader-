from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from ctrader_client import CTraderClient
from config import Settings
from forex_research_store import ForexResearchStore
from market_filters import entry_allowed
from adaptive_router import (
    ADAPTIVE_CONTEXT_VERSION,
    adaptive_signal,
    context_key,
    execution_policy,
    market_context,
)

log = logging.getLogger("microtrader.strategy_lab")


@dataclass(frozen=True)
class Candidate:
    strategy: str
    params: dict


@dataclass
class LabState:
    running: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_error: Optional[str] = None
    progress: int = 0
    total: int = 0
    stage: str = "idle"
    message: str = "Not started"
    symbols_loaded: int = 0
    symbols_total: int = 0
    generation: int = 0
    tested_total: int = 0
    promoted_total: int = 0
    target_promoted: int = 0
    discovery_total: int = 0
    discovery_done: int = 0
    incubator_total: int = 0
    incubator_done: int = 0
    deep_total: int = 0
    deep_done: int = 0
    current_candidate: str = ""
    current_params: dict | None = None
    current_symbol: str = ""
    current_symbol_index: int = 0
    current_symbol_total: int = 0
    current_candidate_started_at: Optional[str] = None
    last_completed_candidate: str = ""
    last_completed_at: Optional[str] = None
    last_progress_at: Optional[str] = None
    candidate_seconds: float = 0.0
    persistence_status: str = "idle"
    persistence_pending: int = 0
    last_persist_error: Optional[str] = None
    paused: bool = False
    pause_reason: str = ""


class StrategyLab:
    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.state = LabState()
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._analysis_results: List[dict] = []
        self._recent_results = deque(maxlen=100)
        self._summary: dict = {}
        self._agent_focus: dict = {
            "families": [],
            "timeframes": [],
            "reason": "",
            "adaptive_policy": {},
        }
        self.store = ForexResearchStore(settings.database_url)
        self._persist_queue: asyncio.Queue = asyncio.Queue()
        self._persist_task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()

    def public_state(self) -> dict:
        payload = asdict(self.state)
        if self.state.current_candidate_started_at and self.state.running:
            try:
                started = datetime.fromisoformat(self.state.current_candidate_started_at)
                payload["candidate_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
            except Exception:
                pass
        payload["summary"] = self._summary
        payload["market"] = "forex"
        payload["data_source"] = "cTrader / Fusion demo"
        payload["pairs"] = list(self.settings.forex_pairs)
        payload["start_capital_eur"] = self.settings.forex_start_capital
        payload["allowed_timeframes_min"] = [1, 5]
        payload["min_trades_per_day"] = self.settings.strategy_min_trades_per_day
        payload["preferred_trades_per_day"] = self.settings.strategy_preferred_trades_per_day
        payload["target_trades_per_day"] = self.settings.strategy_target_trades_per_day
        payload["portfolio_min_trades_per_day"] = self.settings.portfolio_min_trades_per_day
        payload["entry_sessions"] = ["London", "New York"]
        payload["min_volume_ratio"] = self.settings.strategy_min_volume_ratio
        payload["max_loss_streak"] = self.settings.strategy_max_loss_streak
        payload["max_loss_streak_asymmetric"] = self.settings.strategy_max_loss_streak_asymmetric
        payload["adaptive_context_version"] = ADAPTIVE_CONTEXT_VERSION
        return payload

    def results(self) -> List[dict]:
        """Small dashboard-facing result set."""
        return list(self._results)

    def analysis_results(self) -> List[dict]:
        """Bounded but diverse research memory for agents and validation."""
        return list(self._analysis_results)

    def recent_results(self) -> List[dict]:
        """Unranked recent observations for the critic, avoiding survivor bias."""
        return list(self._recent_results)

    def set_agent_focus(
        self,
        families: List[str],
        timeframes: List[int],
        reason: str = "",
        adaptive_policy: Optional[dict] = None,
    ) -> None:
        self._agent_focus = {
            "families": [str(x) for x in families][:6],
            "timeframes": [int(x) for x in timeframes if int(x) in {1, 5}][:2],
            "reason": str(reason)[:500],
            "adaptive_policy": dict(adaptive_policy or {}),
        }

    def agent_focus(self) -> dict:
        return dict(self._agent_focus)

    async def start(self):
        if self.state.running:
            return
        self.state = LabState(
            running=True,
            started_at=datetime.now(timezone.utc).isoformat(),
            stage="starting",
            message="Preparing Forex Research Lab",
            symbols_total=len(self.settings.forex_pairs),
            target_promoted=self.settings.lab_target_promoted,
        )
        self._results = []
        self._analysis_results = []
        self._summary = {}
        self._pause_event.set()
        if not self._persist_task or self._persist_task.done():
            self._persist_task = asyncio.create_task(
                self._persistence_worker(), name="microtrader-strategy-persistence"
            )
        self._task = asyncio.create_task(self._run(), name="microtrader-strategy-lab")

    async def pause(self, reason: str = ""):
        if not self.state.running:
            return
        self.state.paused = True
        self.state.pause_reason = reason or "Paused by research scheduler"
        self._pause_event.clear()

    async def resume(self):
        if not self.state.running:
            await self.start()
            return
        self.state.paused = False
        self.state.pause_reason = ""
        self._pause_event.set()

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self._task = None
        if self._persist_task and not self._persist_task.done():
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
        self._persist_task = None

    async def _persistence_worker(self):
        """Persist checkpoints independently so database latency cannot stop research."""
        while True:
            item = await self._persist_queue.get()
            signature, result, generation, tested_total, promoted_total = item
            self.state.persistence_pending = self._persist_queue.qsize() + 1
            self.state.persistence_status = "saving"
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.store.save_checkpoint_sync,
                        signature, result, generation, tested_total, promoted_total,
                    ),
                    timeout=8.0,
                )
                self.state.persistence_status = "ok"
                self.state.last_persist_error = None
                log.info(
                    "Strategy Lab persisted: tested_total=%s promoted_total=%s generation=%s",
                    tested_total, promoted_total, generation,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.persistence_status = "delayed"
                self.state.last_persist_error = str(exc)[:300]
                log.warning("Strategy Lab persistence delayed: %s", exc)
                # Requeue once at the back; research itself keeps moving.
                await asyncio.sleep(1.0)
                self._persist_queue.put_nowait(item)
            finally:
                self._persist_queue.task_done()
                self.state.persistence_pending = self._persist_queue.qsize()

    async def _run(self):
        try:
            bars_by_symbol: Dict[str, List[dict]] = {}
            self.state.stage = "loading_data"
            self.state.message = "Loading cTrader forex market data"
            for pair in self.settings.forex_pairs:
                self.state.message = f"Loading {pair} from cTrader"
                bars = await self.client.historical_bars(
                    pair,
                    timeframe_min=1,
                    max_bars=self.settings.forex_max_bars_per_pair,
                    lookback_days=self.settings.forex_lookback_days,
                )
                if len(bars) >= 100:
                    bars_by_symbol[pair] = bars
                self.state.symbols_loaded += 1
                await asyncio.sleep(0)

            if not bars_by_symbol:
                raise RuntimeError("No usable cTrader forex bars returned for Research Lab")

            await self.store.init()
            try:
                persisted = await asyncio.wait_for(
                    self.store.load_research_memory(
                        per_family_stage=40,
                        limit=RESEARCH_MEMORY_LIMIT,
                    ),
                    timeout=60.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                persisted = []
                self.state.persistence_status = "delayed"
                self.state.last_persist_error = (
                    "resume memory unavailable: " + str(exc)
                )[:300]
                log.warning(
                    "Strategy Lab resume memory unavailable; continuing without cached rows: %s",
                    exc,
                )

            # Old runs remain in the database for audit/history, but they must
            # never re-enter the active funnel after the policy change.
            persisted = [
                row for row in persisted
                if int((row.get("params") or {}).get("timeframe_min") or 0) in (1, 5)
                and str((row.get("params") or {}).get("entry_sessions") or "") == "london_new_york"
                and str((row.get("params") or {}).get("market") or "") == "forex"
                and str((row.get("params") or {}).get("data_source") or "") == "ctrader"
                and str((row.get("params") or {}).get("direction_mode") or "") == "long_short"
                and str((row.get("params") or {}).get("_policy_version") or "") == RESEARCH_POLICY_VERSION
                and str((row.get("adaptive_diagnostics") or {}).get("context_version") or "") == ADAPTIVE_CONTEXT_VERSION
            ]

            try:
                persisted_signatures = await asyncio.wait_for(
                    self.store.load_signatures(),
                    timeout=15.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                persisted_signatures = {
                    str(row.get("signature"))
                    for row in persisted
                    if row.get("signature")
                }
                self.state.persistence_status = "delayed"
                self.state.last_persist_error = (
                    "resume signatures unavailable: " + str(exc)
                )[:300]
                log.warning(
                    "Strategy Lab signature resume unavailable; using cached-memory signatures: %s",
                    exc,
                )

            try:
                persisted_state = await asyncio.wait_for(
                    self.store.load_state(),
                    timeout=10.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                persisted_state = {
                    "generation": 0,
                    "tested_total": len(persisted_signatures),
                    "promoted_total": sum(1 for row in persisted if row.get("promoted")),
                }
                self.state.persistence_status = "delayed"
                self.state.last_persist_error = (
                    "resume state unavailable: " + str(exc)
                )[:300]
                log.warning(
                    "Strategy Lab state resume unavailable; continuing from safe counters: %s",
                    exc,
                )

            seen = set(persisted_signatures)
            results: List[dict] = list(persisted)
            promoted: List[dict] = [r for r in results if r.get("promoted")]
            batch_size = max(1, self.settings.lab_batch_size)

            self.state.generation = int(persisted_state.get("generation", 0))
            self.state.tested_total = max(
                int(persisted_state.get("tested_total", len(seen))),
                len(seen),
            )
            self.state.promoted_total = len(promoted)
            self._analysis_results = list(results)
            self._results = prune_research_results(
                results,
                limit=DASHBOARD_RESULT_LIMIT,
                per_family_stage=30,
            )
            log.info(
                "Strategy Lab resume: loaded_results=%s loaded_signatures=%s generation=%s tested_total=%s promoted_total=%s",
                len(results), len(seen), self.state.generation, self.state.tested_total, self.state.promoted_total
            )

            self.state.stage = "testing"
            self.state.total = batch_size

            while self.state.running:
                await self._pause_event.wait()
                self.state.generation += 1
                self.state.progress = 0
                self.state.total = batch_size
                batch = choose_batch(
                    results, seen, self.state.generation, batch_size,
                    focus=self._agent_focus,
                )
                phase = str(batch[0].params.get("_phase", "discovery")) if batch else "discovery"
                self.state.stage = phase
                self.state.total = len(batch)
                self.state.message = (
                    f"{phase.replace('_',' ').title()}: testing {len(batch)} candidates"
                )
                if not batch:
                    self.state.stage = "expanding_search"
                    self.state.message = (
                        f"Generation {self.state.generation} exhausted; expanding search space"
                    )
                    log.info(
                        "Strategy Lab generation exhausted: generation=%s; advancing",
                        self.state.generation,
                    )
                    await asyncio.sleep(30.0)
                    continue
                for candidate in batch:
                    seen.add(candidate_signature(candidate))

                for idx, candidate in enumerate(batch, start=1):
                    await self._pause_event.wait()
                    candidate_phase = str(candidate.params.get("_phase", phase))
                    # Fast funnel: Discovery gets a smaller but still broad sample,
                    # Incubator gets more history, Deep Search must confirm on the
                    # full configured dataset before promotion is possible.
                    phase_bar_budget = {
                        "discovery": min(DISCOVERY_SOURCE_BARS, self.settings.forex_max_bars_per_pair),
                        "incubator": min(INCUBATOR_SOURCE_BARS, self.settings.forex_max_bars_per_pair),
                        "deep_search": self.settings.forex_max_bars_per_pair,
                    }.get(candidate_phase, self.settings.forex_max_bars_per_pair)
                    tf = int(candidate.params.get("timeframe_min", 1))
                    candidate_bars: Dict[str, List[dict]] = {}
                    for s, raw_bars in bars_by_symbol.items():
                        if candidate_phase == "deep_search":
                            selected = raw_bars
                        else:
                            holdout_start = max(2, int(len(raw_bars) * (1.0 - FINAL_HOLDOUT_FRACTION)))
                            pre_holdout = raw_bars[:holdout_start]
                            selected = pre_holdout[-phase_bar_budget:]
                        candidate_bars[s] = aggregate_bars(selected, tf)

                    self.state.current_candidate = candidate.strategy
                    self.state.current_params = dict(candidate.params)
                    self.state.current_symbol = ""
                    self.state.current_symbol_index = 0
                    self.state.current_symbol_total = len(candidate_bars)
                    self.state.current_candidate_started_at = datetime.now(timezone.utc).isoformat()
                    self.state.last_progress_at = self.state.current_candidate_started_at
                    self.state.message = (
                        f"{candidate_phase.replace('_',' ').title()}: candidate {idx}/{len(batch)} · "
                        f"{candidate.strategy} · {tf}m · {phase_bar_budget} source bars/symbol"
                    )
                    log.info(
                        "Strategy Lab candidate start: phase=%s strategy=%s tf=%sm bars_per_symbol=%s batch=%s/%s",
                        candidate_phase, candidate.strategy, tf, phase_bar_budget, idx, len(batch)
                    )

                    def symbol_progress(symbol_index: int, symbol_total: int, symbol: str) -> None:
                        self.state.current_symbol_index = symbol_index
                        self.state.current_symbol_total = symbol_total
                        self.state.current_symbol = symbol
                        self.state.last_progress_at = datetime.now(timezone.utc).isoformat()

                    started_candidate = datetime.now(timezone.utc)
                    result = await asyncio.to_thread(
                        evaluate_candidate,
                        candidate,
                        candidate_bars,
                        self.settings.forex_cost_bps,
                        self.settings.forex_stress_cost_multiplier,
                        self.settings.forex_min_oos_trades,
                        self.settings.forex_min_profit_factor,
                        self.settings.lab_max_drawdown_pct,
                        self.settings.lab_min_positive_symbol_ratio,
                        symbol_progress,
                        min_trades_per_day=self.settings.strategy_min_trades_per_day,
                        preferred_trades_per_day=self.settings.strategy_preferred_trades_per_day,
                        target_trades_per_day=self.settings.strategy_target_trades_per_day,
                        max_loss_streak=self.settings.strategy_max_loss_streak,
                        max_loss_streak_asymmetric=self.settings.strategy_max_loss_streak_asymmetric,
                    )
                    elapsed = (datetime.now(timezone.utc) - started_candidate).total_seconds()
                    self.state.candidate_seconds = round(elapsed, 2)
                    self.state.last_completed_candidate = candidate.strategy
                    self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                    self.state.last_progress_at = self.state.last_completed_at
                    self.state.current_symbol = ""
                    log.info(
                        "Strategy Lab candidate complete: strategy=%s phase=%s seconds=%.2f funnel_stage=%s score=%s pf=%s exp_bps=%s oos_trades=%s",
                        candidate.strategy, candidate_phase, elapsed, result.get("funnel_stage"),
                        result.get("funnel_score"), (result.get("oos") or {}).get("profit_factor"),
                        (result.get("oos") or {}).get("expectancy_bps"),
                        (result.get("oos") or {}).get("trades", 0)
                    )
                    self._recent_results.append(result)
                    results.append(result)
                    results = prune_research_results(
                        results,
                        limit=RESEARCH_MEMORY_LIMIT,
                        per_family_stage=80,
                    )
                    signature = candidate_signature(candidate)
                    if result["promoted"]:
                        promoted.append(result)

                    # Update live state immediately. Persistence should never make
                    # the dashboard look frozen while a DB pooler is slow.
                    self.state.progress = idx
                    self.state.tested_total += 1
                    self.state.promoted_total = len(promoted)
                    self.state.message = (
                        f"{phase.replace('_',' ').title()}: {idx}/{len(batch)} tested · "
                        f"{len(promoted)}/{self.settings.lab_target_promoted} promoted"
                    )

                    stage_rank = {"promoted": 4, "deep_search": 3, "incubator": 2, "rejected": 1}
                    results.sort(
                        key=lambda row: (
                            stage_rank.get(row.get("funnel_stage", "rejected"), 0),
                            float(row.get("funnel_score") or 0),
                            float((row.get("oos") or {}).get("expectancy_bps") or 0),
                        ),
                        reverse=True,
                    )
                    self._analysis_results = list(results)
                    self._results = prune_research_results(
                        results,
                        limit=DASHBOARD_RESULT_LIMIT,
                        per_family_stage=30,
                    )

                    # Queue persistence instead of awaiting the database. A slow
                    # Supabase pooler must never pause strategy discovery.
                    if self.store.enabled:
                        self._persist_queue.put_nowait((
                            signature,
                            result,
                            self.state.generation,
                            self.state.tested_total,
                            self.state.promoted_total,
                        ))
                        self.state.persistence_pending = self._persist_queue.qsize()
                        if self.state.persistence_status == "idle":
                            self.state.persistence_status = "queued"
                    else:
                        self.state.persistence_status = "disabled"

                    self._summary = {
                        "symbols": list(bars_by_symbol.keys()),
                        "bars": {s: len(v) for s, v in bars_by_symbol.items()},
                        "market": "forex",
                        "data_source": "cTrader / Fusion demo",
                        "pairs": list(self.settings.forex_pairs),
                        "source_timeframe": "1Min",
                        "agent_focus": self.agent_focus(),
                        "candidates_tested": self.state.tested_total,
                        "promoted_count": len(promoted),
                        "target_promoted": self.settings.lab_target_promoted,
                        "generation": self.state.generation,
                        "funnel_counts": {
                            "promoted": sum(1 for r in results if r.get("funnel_stage") == "promoted"),
                            "deep_search": sum(1 for r in results if r.get("funnel_stage") == "deep_search"),
                            "incubator": sum(1 for r in results if r.get("funnel_stage") == "incubator"),
                            "rejected": sum(1 for r in results if r.get("funnel_stage", "rejected") == "rejected"),
                        },
                        "near_misses": sorted(
                            [r for r in results if not r.get("promoted")],
                            key=lambda r: float(r.get("funnel_score") or 0),
                            reverse=True,
                        )[:10],
                        "best_candidate": promoted[0] if promoted else (results[0] if results else None),
                        "cost_bps_per_side": self.settings.forex_cost_bps,
                        "stress_cost_multiplier": self.settings.forex_stress_cost_multiplier,
                        "method": (
                            "forex-only cTrader research; chronological 70/30 holdout; "
                            "1m/5m; London/New York entries; relative-volume filter; "
                            "next-bar-open fills; per-pair robustness filter"
                        ),
                    }
                    await asyncio.sleep(0)

                    # In continuous mode the promotion target is a milestone, not a
                    # stop condition. Research must keep exploring/refining after the
                    # first promoted candidates have been found.
                    if (
                        not self.settings.lab_continuous
                        and self.settings.lab_target_promoted > 0
                        and len(promoted) >= self.settings.lab_target_promoted
                    ):
                        self.state.stage = "target_reached"
                        self.state.message = (
                            f"Target reached: {len(promoted)} robust candidates found after "
                            f"{self.state.tested_total} tests"
                        )
                        return

                if not self.settings.lab_continuous:
                    self.state.stage = "completed"
                    self.state.message = (
                        f"Batch completed: {self.state.tested_total} candidates tested"
                    )
                    return

        except asyncio.CancelledError:
            self.state.stage = "stopped"
            self.state.message = "Strategy Lab stopped"
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Strategy Lab failed")
        finally:
            self.state.running = False
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


RESEARCH_POLICY_VERSION = "forex-ctrader-v7-net-r-risk-holdout20-oos20"
FINAL_HOLDOUT_FRACTION = 0.20
DISCOVERY_SOURCE_BARS = 10000
INCUBATOR_SOURCE_BARS = 20000
RESEARCH_MEMORY_LIMIT = 1000
DASHBOARD_RESULT_LIMIT = 250
MIN_DEEP_OOS_DAYS = 3

def candidate_signature(candidate: Candidate) -> str:
    import json
    return (
        candidate.strategy + ":" +
        json.dumps(candidate.params, sort_keys=True, separators=(",", ":")) +
        ":" + RESEARCH_POLICY_VERSION +
        ":" + ADAPTIVE_CONTEXT_VERSION
    )


def discovery_candidates() -> List[Candidate]:
    out: List[Candidate] = []
    for tf in (1, 5):
        common = {"timeframe_min": tf, "_phase": "discovery", "_policy_version": RESEARCH_POLICY_VERSION, "market": "forex", "data_source": "ctrader", "direction_mode": "long_short", "entry_sessions": "london_new_york", "min_volume_ratio": 0.70, "volume_window": 50}
        out.extend([
            Candidate("momentum", {**common, "fast": 4, "slow": 16, "entry_bps": 8.0, "max_hold": 16}),
            Candidate("mean_reversion", {**common, "window": 20, "z_entry": 1.5, "z_exit": 0.25, "max_hold": 20}),
            Candidate("breakout", {**common, "window": 20, "buffer_bps": 3.0, "max_hold": 24}),
            Candidate("extreme_reversal", {**common, "window": 12, "shock_z": 2.0, "max_hold": 12}),
            Candidate("volatility_breakout", {**common, "window": 20, "vol_mult": 1.6, "max_hold": 20}),
            Candidate("trend_pullback", {**common, "fast": 8, "slow": 30, "pullback_z": 1.0, "max_hold": 24}),
            Candidate("regime_ensemble", {
                **common,
                "regime_atr_short": 14, "regime_atr_long": 50,
                "regime_vol_ratio": 1.20, "regime_trend_atr": 0.55,
                "fast": 8, "slow": 30, "pullback_z": 1.0,
                "breakout_window": 20, "buffer_bps": 2.0,
                "vwap_window": 60, "z_entry": 1.4,
                "stop_atr": 1.0, "target_r": 2.0, "max_hold": 24,
            }),
            Candidate("vwap_reversion", {**common, "window": 60, "z_entry": 1.5, "max_hold": 20}),
            Candidate("vwap_momentum", {**common, "window": 60, "buffer_bps": 8.0, "max_hold": 20}),
            Candidate("asymmetric_breakout", {**common, "window": 20, "stop_atr": 0.7, "target_r": 3.0, "max_hold": 40}),
            Candidate("asymmetric_breakout", {**common, "window": 20, "stop_atr": 0.6, "target_r": 5.0, "max_hold": 60}),
            Candidate("asymmetric_breakout", {**common, "window": 40, "stop_atr": 0.5, "target_r": 8.0, "max_hold": 90}),
        ])
    return out


def parameter_variants(row: dict, phase: str, generation: int = 1) -> List[Candidate]:
    """Create deterministic local variants around a promising candidate.

    Generation changes the perturbation grid so Incubator/Deep Search never
    silently exhaust after a handful of batches, while keeping changes local.
    """
    strategy = str(row.get("strategy"))
    base = dict(row.get("params") or {})
    base["_phase"] = phase
    out: List[Candidate] = []

    # Cycle through increasingly fine local offsets. This produces genuinely
    # different trading parameters without adding signature-only metadata.
    g = max(1, int(generation))
    band = 0.30 if phase == "incubator" else 0.16
    slot = ((g - 1) % 31) - 15
    center_shift = (slot / 15.0) * band if slot else 0.0
    local_offsets = (
        center_shift - band / 5.0,
        center_shift,
        center_shift + band / 5.0,
    )

    keys = [
        k for k, v in base.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and k not in {
            "timeframe_min", "target_r", "min_volume_ratio", "volume_window",
            "risk_eur", "start_capital_eur", "direction_mode",
        }
    ][:4]

    for key in keys:
        for offset in local_offsets:
            q = dict(base)
            v = base[key]
            nv = float(v) * (1.0 + offset)
            q[key] = max(1, int(round(nv))) if isinstance(v, int) else round(max(0.01, nv), 4)
            q["_phase"] = phase
            out.append(Candidate(strategy, q))

    if strategy == "regime_ensemble":
        ensemble_keys = [
            "regime_vol_ratio", "regime_trend_atr", "pullback_z",
            "buffer_bps", "z_entry", "target_r",
        ]
        for key in ensemble_keys:
            if key not in base:
                continue
            for offset in local_offsets:
                q = dict(base)
                v = float(base[key])
                q[key] = round(max(0.01, v * (1.0 + offset)), 4)
                q["_phase"] = phase
                out.append(Candidate(strategy, q))

    if strategy == "adaptive_router":
        for key in (
            "regime_vol_ratio", "regime_trend_atr",
            "adaptive_exit_target_scale", "adaptive_exit_stop_scale",
            "adaptive_exit_hold_scale",
        ):
            if key not in base:
                continue
            for offset in local_offsets:
                q = dict(base)
                v = float(base[key])
                q[key] = round(max(0.05, v * (1.0 + offset)), 4)
                q["_phase"] = phase
                out.append(Candidate(strategy, q))

    if strategy == "asymmetric_breakout":
        # Explore asymmetric payoffs densely around the current target as well
        # as a few canonical R multiples.
        base_r = float(base.get("target_r", 3.0))
        r_shift = 0.25 * (((g - 1) % 21) - 10)
        for r in sorted({2.0, 3.0, 5.0, 8.0, 10.0, max(1.25, base_r + r_shift)}):
            q = dict(base)
            q["target_r"] = round(r, 2)
            q["_phase"] = phase
            out.append(Candidate(strategy, q))
    return out

def _candidate_priority(candidate: Candidate, focus_families: set[str], focus_timeframes: set[int]) -> tuple:
    return (
        1 if candidate.strategy in focus_families else 0,
        1 if int(candidate.params.get("timeframe_min", 0)) in focus_timeframes else 0,
    )


def _round_robin_candidates(
    candidates: List[Candidate],
    *,
    batch_size: int,
    focus_families: set[str],
    focus_timeframes: set[int],
) -> List[Candidate]:
    groups: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.strategy, []).append(candidate)
    family_order = sorted(
        groups,
        key=lambda family: (
            1 if family in focus_families else 0,
            max(
                (
                    1
                    if int(c.params.get("timeframe_min", 0)) in focus_timeframes
                    else 0
                )
                for c in groups[family]
            ),
            family,
        ),
        reverse=True,
    )
    out: List[Candidate] = []
    while family_order and len(out) < batch_size:
        next_round: List[str] = []
        for family in family_order:
            if groups[family] and len(out) < batch_size:
                out.append(groups[family].pop(0))
            if groups[family]:
                next_round.append(family)
        family_order = next_round
    return out


def _context_backfill_candidates(
    results: List[dict],
    seen: set[str],
) -> List[Candidate]:
    """Re-run historical configurations to recover context-level evidence.

    Old aggregate rows do not contain the per-trade context needed by the new
    router. Re-evaluate the same trading parameters on the pre-holdout
    discovery slice, including historically rejected configurations: a model
    that is weak globally may still be a useful specialist in one regime.
    """
    rows = sorted(
        results,
        key=lambda row: (
            float(row.get("funnel_score") or 0.0),
            float((row.get("oos") or {}).get("trades") or 0.0),
        ),
        reverse=True,
    )
    out: List[Candidate] = []
    local_seen: set[str] = set()
    for row in rows:
        strategy = str(row.get("strategy") or "")
        if not strategy or strategy == "adaptive_router":
            continue
        diagnostics = dict(row.get("adaptive_diagnostics") or {})
        if (
            diagnostics.get("context_entry_breakdown")
            and diagnostics.get("context_entry_management_breakdown")
            and diagnostics.get("context_entry_exit_breakdown")
            and diagnostics.get("research_context_entry_breakdown")
            and diagnostics.get("research_context_entry_management_breakdown")
            and diagnostics.get("research_context_entry_exit_breakdown")
            and str(diagnostics.get("context_version") or "") == ADAPTIVE_CONTEXT_VERSION
        ):
            continue
        params = dict(row.get("params") or {})
        if not params:
            continue
        if str(params.get("_policy_version") or "") != RESEARCH_POLICY_VERSION:
            continue
        # Context backfill deliberately stays on pre-holdout discovery data.
        params["_phase"] = "discovery"
        params["_policy_version"] = RESEARCH_POLICY_VERSION
        # Signature-only migration marker: forces one fresh evaluation for
        # historical rows whose stored diagnostics predate the combined
        # pre-holdout context sample. It does not change trading behavior.
        params["_context_backfill_version"] = "preholdout-sample-v2"
        candidate = Candidate(strategy, params)
        signature = candidate_signature(candidate)
        if signature in seen or signature in local_seen:
            continue
        local_seen.add(signature)
        out.append(candidate)
    return out


def choose_batch(
    results: List[dict],
    seen: set[str],
    generation: int,
    batch_size: int,
    focus: Optional[dict] = None,
) -> List[Candidate]:
    focus = focus or {}
    focus_families = set(focus.get("families") or [])
    focus_timeframes = {
        int(x) for x in (focus.get("timeframes") or []) if int(x) in {1, 5}
    }

    adaptive_policy_bundle = dict(focus.get("adaptive_policy") or {})
    timeframe_policies = dict(adaptive_policy_bundle.get("timeframes") or {})

    # Harvest context evidence from historical strategies without monopolizing
    # the lab. Before an adaptive policy exists, alternate backfill with normal
    # research. Once specialists exist, spend roughly one generation in three
    # on backfill so the router can keep improving while adaptive candidates
    # are already being tested.
    context_backfill = _context_backfill_candidates(results, seen)
    backfill_every = 3 if timeframe_policies else 2
    if context_backfill and generation % backfill_every == 0:
        context_backfill.sort(
            key=lambda candidate: _candidate_priority(
                candidate, focus_families, focus_timeframes
            ),
            reverse=True,
        )
        return _round_robin_candidates(
            context_backfill,
            batch_size=batch_size,
            focus_families=focus_families,
            focus_timeframes=focus_timeframes,
        )

    # Once the research agent has enough context-labelled evidence, inject one
    # data-driven adaptive-router candidate per active timeframe. The complete
    # specialist policy is part of the signature, so unchanged policy is never
    # re-tested merely because a new generation started.
    if timeframe_policies:
        adaptive_candidates: List[Candidate] = []
        adaptive_tfs = sorted(focus_timeframes or {1, 5})
        for tf in adaptive_tfs:
            learned_policy = dict(timeframe_policies.get(str(tf)) or {})
            if not learned_policy.get("routes"):
                continue
            adaptive_policy = execution_policy(learned_policy)
            params = {
                "timeframe_min": int(tf),
                "_phase": "discovery",
                "_policy_version": RESEARCH_POLICY_VERSION,
                "market": "forex",
                "data_source": "ctrader",
                "direction_mode": "long_short",
                "entry_sessions": "london_new_york",
                "min_volume_ratio": 0.70,
                "volume_window": 50,
                "regime_atr_short": 14,
                "regime_atr_long": 50,
                "regime_vol_ratio": 1.20,
                "regime_trend_atr": 0.55,
                "fast": 8,
                "slow": 30,
                "vwap_window": 60,
                "adaptive_exit_target_scale": 1.0,
                "adaptive_exit_stop_scale": 1.0,
                "adaptive_exit_hold_scale": 1.0,
                "_adaptive_research_mode": False,
                "adaptive_policy": adaptive_policy,
            }
            candidate = Candidate("adaptive_router", params)
            if candidate_signature(candidate) not in seen:
                adaptive_candidates.append(candidate)
        if adaptive_candidates:
            return adaptive_candidates[:batch_size]

    # Only continue broad standalone discovery when there is no new
    # evidence-backed adaptive router waiting to be evaluated. This prevents
    # the final adaptive strategy from being starved by an effectively endless
    # stream of lower-priority standalone candidates.
    pending_discovery = [
        candidate for candidate in discovery_candidates()
        if candidate_signature(candidate) not in seen
    ]
    if pending_discovery:
        pending_discovery.sort(
            key=lambda candidate: _candidate_priority(
                candidate, focus_families, focus_timeframes
            ),
            reverse=True,
        )
        return _round_robin_candidates(
            pending_discovery,
            batch_size=batch_size,
            focus_families=focus_families,
            focus_timeframes=focus_timeframes,
        )

    # A configuration that passed Incubator is frozen before Deep Search.
    # Deep Search never mutates parameters: it is the final holdout test.
    frozen_deep: List[Candidate] = []
    for row in results:
        params = dict(row.get("params") or {})
        if (
            row.get("funnel_stage") == "deep_search"
            and params.get("_phase") == "incubator"
        ):
            params["_phase"] = "deep_search"
            candidate = Candidate(str(row.get("strategy") or ""), params)
            if candidate.strategy and candidate_signature(candidate) not in seen:
                frozen_deep.append(candidate)
    if frozen_deep:
        frozen_deep.sort(
            key=lambda candidate: _candidate_priority(
                candidate, focus_families, focus_timeframes
            ),
            reverse=True,
        )
        return _round_robin_candidates(
            frozen_deep,
            batch_size=batch_size,
            focus_families=focus_families,
            focus_timeframes=focus_timeframes,
        )

    # Only Discovery winners are locally varied in Incubator.
    parents = [
        row for row in results
        if row.get("funnel_stage") == "incubator"
        and (row.get("params") or {}).get("_phase") == "discovery"
    ]
    parents.sort(
        key=lambda row: (
            1 if row.get("strategy") in focus_families else 0,
            1 if int((row.get("params") or {}).get("timeframe_min", 0)) in focus_timeframes else 0,
            float(row.get("funnel_score") or 0),
        ),
        reverse=True,
    )
    # Keep several parents per family, not just the global top rows.
    parent_counts: Dict[str, int] = {}
    incubator: List[Candidate] = []
    for row in parents:
        family = str(row.get("strategy") or "")
        if parent_counts.get(family, 0) >= 3:
            continue
        parent_counts[family] = parent_counts.get(family, 0) + 1
        incubator.extend(parameter_variants(row, "incubator", generation))

    local_seen: set[str] = set()
    incubator = [
        candidate for candidate in incubator
        if candidate_signature(candidate) not in seen
        and not (
            candidate_signature(candidate) in local_seen
            or local_seen.add(candidate_signature(candidate))
        )
    ]
    if incubator:
        return _round_robin_candidates(
            incubator,
            batch_size=batch_size,
            focus_families=focus_families,
            focus_timeframes=focus_timeframes,
        )

    # No promising parent left: broaden Discovery deterministically.
    g = max(1, generation)
    # Search within plausible intraday ranges, even after thousands of cycles.
    # Unbounded epoch growth made thresholds unreachable and 5m warmups
    # longer than the OOS sample. Seeded jitter explores without that drift.
    rng = random.Random(g)
    epoch = rng.randrange(9)
    wave = rng.randrange(1, 7)
    broad: List[Candidate] = []
    for tf in (1, 5):
        templates = [
            Candidate("momentum", {"timeframe_min": tf, "_phase": "discovery",
                "fast": 2 + (wave % 5), "slow": 10 + 3*wave + 2*epoch,
                "entry_bps": round(3.0 + 1.5*wave + 0.25*epoch, 2), "max_hold": 8 + 3*wave + epoch}),
            Candidate("mean_reversion", {"timeframe_min": tf, "_phase": "discovery",
                "window": 10 + 4*wave + 2*epoch, "z_entry": round(0.8 + 0.18*wave + 0.03*epoch, 2),
                "z_exit": round(0.05*((wave + epoch) % 7), 2), "max_hold": 10 + 3*wave + epoch}),
            Candidate("breakout", {"timeframe_min": tf, "_phase": "discovery",
                "window": 10 + 5*wave + 2*epoch, "buffer_bps": round(2.0 + wave + 0.25*epoch, 2),
                "max_hold": 12 + 4*wave + epoch}),
            Candidate("extreme_reversal", {"timeframe_min": tf, "_phase": "discovery",
                "window": 8 + 2*wave + 2*epoch, "shock_z": round(1.4 + 0.15*wave + 0.03*epoch, 2),
                "max_hold": 8 + 2*wave + epoch}),
            Candidate("volatility_breakout", {"timeframe_min": tf, "_phase": "discovery",
                "window": 12 + 3*wave + 2*epoch, "vol_mult": round(1.1 + 0.1*wave + 0.02*epoch, 2),
                "max_hold": 12 + 3*wave + epoch}),
            Candidate("trend_pullback", {"timeframe_min": tf, "_phase": "discovery",
                "fast": 4 + wave, "slow": 20 + 4*wave + 2*epoch,
                "pullback_z": round(0.6 + 0.12*wave + 0.02*epoch, 2), "max_hold": 12 + 3*wave + epoch}),
            Candidate("regime_ensemble", {"timeframe_min": tf, "_phase": "discovery",
                "regime_atr_short": 8 + wave,
                "regime_atr_long": 40 + 5*wave + 2*epoch,
                "regime_vol_ratio": round(1.05 + 0.05*wave + 0.01*epoch, 2),
                "regime_trend_atr": round(0.25 + 0.07*wave + 0.01*epoch, 2),
                "fast": 4 + wave, "slow": 20 + 4*wave + 2*epoch,
                "pullback_z": round(0.6 + 0.10*wave + 0.02*epoch, 2),
                "breakout_window": 12 + 3*wave + 2*epoch,
                "buffer_bps": round(0.5 + 0.5*wave + 0.10*epoch, 2),
                "vwap_window": 30 + 5*wave + 2*epoch,
                "z_entry": round(0.9 + 0.12*wave + 0.02*epoch, 2),
                "stop_atr": 1.0, "target_r": 2.0, "max_hold": 24}),
            Candidate("vwap_reversion", {"timeframe_min": tf, "_phase": "discovery",
                "window": 30 + 5*wave + 2*epoch, "z_entry": round(0.9 + 0.12*wave + 0.02*epoch, 2),
                "max_hold": 10 + 3*wave + epoch}),
            Candidate("vwap_momentum", {"timeframe_min": tf, "_phase": "discovery",
                "window": 30 + 5*wave + 2*epoch, "buffer_bps": round(3.0 + wave + 0.2*epoch, 2),
                "max_hold": 10 + 3*wave + epoch}),
            Candidate("asymmetric_breakout", {"timeframe_min": tf, "_phase": "discovery",
                "window": 15 + 5*wave + 2*epoch, "stop_atr": round(0.4 + 0.05*wave + 0.01*epoch, 2),
                "target_r": float((3, 5, 8, 10, 12, 15)[(wave - 1) % 6]),
                "max_hold": 30 + 5*wave + 2*epoch}),
        ]
        for candidate in templates:
            varied = dict(candidate.params)
            for key, value in list(varied.items()):
                if key in {"timeframe_min", "target_r"} or not isinstance(value, (int, float)):
                    continue
                value2 = value * rng.uniform(0.80, 1.20)
                varied[key] = max(1, round(value2)) if isinstance(value, int) else round(value2, 4)
            candidate = Candidate(candidate.strategy, varied)
            candidate = Candidate(candidate.strategy, {
                **candidate.params,
                "_policy_version": RESEARCH_POLICY_VERSION,
                "market": "forex",
                "data_source": "ctrader",
                "direction_mode": "long_short",
                "entry_sessions": "london_new_york",
                "min_volume_ratio": 0.70,
                "volume_window": 50,
            })
            if candidate_signature(candidate) not in seen:
                broad.append(candidate)
    return _round_robin_candidates(
        broad,
        batch_size=batch_size,
        focus_families=focus_families,
        focus_timeframes=focus_timeframes,
    )


def prune_research_results(
    rows: List[dict],
    *,
    limit: int,
    per_family_stage: int,
) -> List[dict]:
    """Keep a bounded, diverse research set without starving a timeframe."""
    stage_rank = {"promoted": 4, "deep_search": 3, "incubator": 2, "rejected": 1}
    ordered = sorted(
        rows,
        key=lambda row: (
            stage_rank.get(str(row.get("funnel_stage") or "rejected"), 0),
            bool((row.get("adaptive_diagnostics") or {}).get("context_entry_breakdown")),
            int((row.get("oos") or {}).get("trades") or 0) > 0,
            float(row.get("funnel_score") or 0),
            float((row.get("oos") or {}).get("expectancy_bps") or 0),
        ),
        reverse=True,
    )
    counts: Dict[tuple[str, str, int], int] = {}
    timeframe_counts = {1: 0, 5: 0}
    timeframe_quota = {1: max(1, limit // 2), 5: max(1, limit - (limit // 2))}
    kept: List[dict] = []
    kept_ids: set[int] = set()

    def bucket(row: dict) -> tuple[str, str, int]:
        return (
            str(row.get("family") or row.get("strategy") or "unknown"),
            str(row.get("funnel_stage") or "rejected"),
            int((row.get("params") or {}).get("timeframe_min") or 0),
        )

    # First pass reserves capacity for both supported timeframes.
    for row in ordered:
        key = bucket(row)
        tf = key[2]
        cap = limit if bool(row.get("promoted")) else per_family_stage
        if counts.get(key, 0) >= cap:
            continue
        if tf in timeframe_quota and timeframe_counts[tf] >= timeframe_quota[tf]:
            continue
        counts[key] = counts.get(key, 0) + 1
        if tf in timeframe_counts:
            timeframe_counts[tf] += 1
        kept.append(row)
        kept_ids.add(id(row))
        if len(kept) >= limit:
            return kept

    # Reallocate unused capacity to the best remaining rows.
    for row in ordered:
        if id(row) in kept_ids:
            continue
        key = bucket(row)
        cap = limit if bool(row.get("promoted")) else per_family_stage
        if counts.get(key, 0) >= cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(row)
        if len(kept) >= limit:
            break
    return kept

def candidate_stream():
    for c in discovery_candidates():
        yield c


def candidate_grid() -> List[Candidate]:
    return discovery_candidates()


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

def _candidate_exit_model(candidate: Candidate) -> str:
    p = candidate.params
    if candidate.strategy in {"regime_ensemble", "asymmetric_breakout"}:
        return (
            f"atr_rr_stop{float(p.get('stop_atr', 1.0)):.2f}_"
            f"target{float(p.get('target_r', 2.0)):.2f}_hold{int(p.get('max_hold', 24))}"
        )
    if candidate.strategy == "momentum":
        return f"momentum_flip_hold{int(p.get('max_hold', 16))}"
    if candidate.strategy == "mean_reversion":
        return (
            f"mean_reversion_z{float(p.get('z_exit', 0.25)):.2f}_"
            f"hold{int(p.get('max_hold', 20))}"
        )
    return f"time_exit_hold{int(p.get('max_hold', 24))}"


MANAGEMENT_PROFILES = {
    "baseline": {"name": "baseline"},
    "protect_1r_be": {
        "name": "protect_1r_be",
        "management_trigger_r": 1.0,
        "management_lock_net_r": 0.0,
    },
    "protect_1_5r_0_2r": {
        "name": "protect_1_5r_0_2r",
        "management_trigger_r": 1.5,
        "management_lock_net_r": 0.2,
    },
    "protect_2r_0_5r": {
        "name": "protect_2r_0_5r",
        "management_trigger_r": 2.0,
        "management_lock_net_r": 0.5,
    },
}


EXIT_PROFILES = {
    "compact_1_5r": {
        "name": "compact_1_5r",
        "stop_atr": 0.9,
        "target_r": 1.5,
        "max_hold": 18,
    },
    "balanced_2_5r": {
        "name": "balanced_2_5r",
        "stop_atr": 1.0,
        "target_r": 2.5,
        "max_hold": 32,
    },
    "asymmetric_3r": {
        "name": "asymmetric_3r",
        "stop_atr": 0.8,
        "target_r": 3.0,
        "max_hold": 36,
    },
}


def _max_loss_stop_price(
    *,
    entry: float,
    risk_distance: float,
    direction: int,
    cost_bps: float,
    max_loss_r: float = 1.0,
) -> float:
    """Price that limits NET modeled loss to max_loss_r after round-trip costs."""
    if entry <= 0 or risk_distance <= 0 or direction not in (-1, 1):
        return entry
    risk_pct = risk_distance / entry
    roundtrip_cost_pct = 2.0 * float(cost_bps) / 10_000.0
    desired_net = -abs(float(max_loss_r)) * risk_pct
    signed_gross = desired_net + roundtrip_cost_pct
    return entry * (1.0 + signed_gross / float(direction))


def _net_r_multiple(
    *,
    entry: float,
    exit_price: float,
    risk_distance: float,
    direction: int,
    cost_bps: float,
) -> float:
    if entry <= 0 or risk_distance <= 0 or direction not in (-1, 1):
        return 0.0
    gross = direction * ((float(exit_price) / entry) - 1.0)
    net = gross - (2.0 * float(cost_bps) / 10_000.0)
    risk_pct = risk_distance / entry
    return net / risk_pct if risk_pct > 0 else 0.0


def _exit_outcomes(
    trade: dict,
    bars: List[dict],
    entry_idx: int,
    cost_bps: float,
) -> dict:
    """Counterfactual simple exits from the same causal entry.

    These are discovery diagnostics only. The selected entry + management +
    exit combination must later pass a full combined backtest and Validation.
    """
    entry = float(trade.get("entry_price") or 0.0)
    direction = int(trade.get("direction") or 0)
    atr = _atr(bars, entry_idx, 14)
    if entry <= 0 or direction not in (-1, 1) or atr <= 0:
        return {}

    outcomes = {}
    for name, profile in EXIT_PROFILES.items():
        risk_distance = atr * float(profile["stop_atr"])
        if risk_distance <= 0:
            continue
        stop = _max_loss_stop_price(
            entry=entry,
            risk_distance=risk_distance,
            direction=direction,
            cost_bps=float(cost_bps),
            max_loss_r=1.0,
        )
        target_r = float(profile["target_r"])
        target = (
            entry + risk_distance * target_r
            if direction > 0 else entry - risk_distance * target_r
        )
        last_idx = min(
            entry_idx + int(profile["max_hold"]),
            len(bars) - 1,
        )
        exit_price = float(bars[last_idx]["c"])
        reason = "max_hold"

        for j in range(entry_idx, last_idx + 1):
            low = float(bars[j]["l"])
            high = float(bars[j]["h"])
            stop_hit = low <= stop if direction > 0 else high >= stop
            target_hit = high >= target if direction > 0 else low <= target
            # Conservative ambiguity handling.
            if stop_hit:
                exit_price = stop
                reason = "stop"
                break
            if target_hit:
                exit_price = target
                reason = "target"
                break

        gross = (
            direction * ((exit_price / entry) - 1.0)
            if entry > 0 else 0.0
        )
        net = gross - (2.0 * float(cost_bps) / 10_000.0)
        risk_pct = risk_distance / entry if entry > 0 else 0.0
        outcomes[name] = {
            "net_return": net,
            "r_multiple": net / risk_pct if risk_pct > 0 else 0.0,
            "exit_reason": reason,
        }
    return outcomes


def _trade_lifecycle(
    trade: dict,
    bars: List[dict],
    entry_idx: int,
    exit_idx: int,
    cost_bps: float,
) -> dict:
    """Measure the full path and counterfactual profit-protection outcomes.

    Management is intentionally simple. A protection level earned on a bar
    becomes active on the next bar, matching forward paper semantics and
    avoiding same-bar look-ahead.
    """
    entry = float(trade.get("entry_price") or 0.0)
    direction = int(trade.get("direction") or 0)
    risk_distance = float(trade.get("risk_distance") or 0.0) or _atr(
        bars, entry_idx, 14
    )
    if entry <= 0 or direction not in (-1, 1) or risk_distance <= 0:
        return {
            "mfe_r": 0.0,
            "mae_r": 0.0,
            "giveback_r": 0.0,
            "risk_distance": 0.0,
            "management_outcomes": {},
        }

    risk_pct = risk_distance / entry
    roundtrip_cost_pct = 2.0 * float(cost_bps) / 10_000.0
    baseline_net_r = (
        float(trade.get("net_return") or 0.0) / risk_pct
        if risk_pct > 0 else 0.0
    )
    last_idx = min(exit_idx, len(bars) - 1)
    exit_price = float(trade.get("exit_price") or entry)

    mfe_r = 0.0
    mae_r = 0.0
    # Full OHLC ranges are only observable while the position survives the bar.
    # On the actual exit bar use only the exit price, never later candle range.
    for j in range(entry_idx, last_idx):
        hi = float(bars[j]["h"])
        lo = float(bars[j]["l"])
        favorable = (
            (hi - entry) / risk_distance
            if direction > 0 else (entry - lo) / risk_distance
        )
        adverse = (
            (entry - lo) / risk_distance
            if direction > 0 else (hi - entry) / risk_distance
        )
        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)

    exit_move_r = direction * (exit_price - entry) / risk_distance
    mfe_r = max(mfe_r, exit_move_r)
    mae_r = max(mae_r, -exit_move_r)

    outcomes = {}
    for name, profile in MANAGEMENT_PROFILES.items():
        if name == "baseline":
            outcomes[name] = {
                "net_return": float(trade.get("net_return") or 0.0),
                "r_multiple": baseline_net_r,
            }
            continue

        trigger = float(profile["management_trigger_r"])
        lock_net_r = float(profile["management_lock_net_r"])
        triggered = False
        managed_net_r = baseline_net_r
        for j in range(entry_idx, last_idx + 1):
            hi = float(bars[j]["h"])
            lo = float(bars[j]["l"])

            # A protection level earned on an earlier bar is active now.
            if triggered:
                lock_gross_return = lock_net_r * risk_pct + roundtrip_cost_pct
                lock_price = entry * (
                    1.0 + (lock_gross_return / float(direction))
                )
                lock_hit = lo <= lock_price if direction > 0 else hi >= lock_price
                if lock_hit:
                    managed_net_r = lock_net_r
                    break

            # Do not create a trigger from movement after the actual exit.
            if j >= last_idx:
                break

            favorable = (
                (hi - entry) / risk_distance
                if direction > 0 else (entry - lo) / risk_distance
            )
            if not triggered and favorable >= trigger:
                triggered = True

        outcomes[name] = {
            "net_return": managed_net_r * risk_pct,
            "r_multiple": managed_net_r,
        }

    return {
        "mfe_r": round(mfe_r, 4),
        "mae_r": round(mae_r, 4),
        "giveback_r": round(max(0.0, mfe_r - baseline_net_r), 4),
        "risk_distance": risk_distance,
        "management_outcomes": outcomes,
    }


def _annotate_trade_context(
    candidate: Candidate,
    trades: List[dict],
    bars: List[dict],
    cost_bps: float = 0.0,
) -> None:
    if not trades or not bars:
        return
    by_time = {str(row.get("t") or ""): idx for idx, row in enumerate(bars)}
    default_exit = _candidate_exit_model(candidate)
    for trade in trades:
        idx = by_time.get(str(trade.get("entry_time") or ""))
        if idx is None:
            continue
        exit_idx = by_time.get(str(trade.get("exit_time") or ""), idx)
        context = dict(trade.get("entry_context") or market_context(bars, idx, candidate.params))
        trade.setdefault("entry_model", candidate.strategy)
        trade.setdefault("entry_regime", str(context.get("regime") or "unknown"))
        trade["entry_context"] = context
        trade.setdefault("context_key", context_key(context))
        trade.setdefault("exit_model", default_exit)
        trade["time_bucket"] = str(context.get("time_bucket") or "unknown")
        trade["volume_bucket"] = str(context.get("volume_bucket") or "unknown")
        trade["volatility_bucket"] = str(context.get("volatility") or "unknown")
        trade["trend_direction"] = str(context.get("trend_direction") or "unknown")
        lifecycle = _trade_lifecycle(
            trade, bars, idx, exit_idx, float(cost_bps)
        )
        trade.update(lifecycle)
        trade["exit_outcomes"] = _exit_outcomes(
            trade, bars, idx, float(cost_bps)
        )


def evaluate_candidate(
    candidate: Candidate,
    bars_by_symbol: Dict[str, List[dict]],
    cost_bps: float,
    stress_cost_multiplier: float,
    min_oos_trades: int,
    min_profit_factor: float = 1.15,
    max_drawdown_pct: float = 6.0,
    min_positive_symbol_ratio: float = 0.60,
    progress_callback=None,
    min_trades_per_day: float = 3.0,
    preferred_trades_per_day: float = 5.0,
    target_trades_per_day: float = 10.0,
    max_loss_streak: int = 5,
    max_loss_streak_asymmetric: int = 7,
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_oos_trades: List[dict] = []
    per_symbol: Dict[str, dict] = {}
    train_days: set[str] = set()
    oos_days: set[str] = set()

    candidate_phase = str(candidate.params.get("_phase", "discovery"))
    symbol_items = list(bars_by_symbol.items())
    for symbol_index, (symbol, bars) in enumerate(symbol_items, start=1):
        if progress_callback:
            progress_callback(symbol_index, len(symbol_items), symbol)
        split_fraction = (
            1.0 - FINAL_HOLDOUT_FRACTION
            if candidate_phase == "deep_search"
            else 0.70
        )
        split = max(2, int(len(bars) * split_fraction))
        train = bars[:split]
        test = bars[split:]
        train_days.update(str(x.get("t") or "")[:10] for x in train if x.get("t"))
        oos_days.update(str(x.get("t") or "")[:10] for x in test if x.get("t"))
        symbol_train = simulate(candidate, symbol, train, cost_bps)
        symbol_oos = simulate(candidate, symbol, test, cost_bps)
        symbol_stress = simulate(candidate, symbol, test, cost_bps * stress_cost_multiplier)
        _annotate_trade_context(candidate, symbol_train, train, cost_bps)
        _annotate_trade_context(candidate, symbol_oos, test, cost_bps)
        _annotate_trade_context(
            candidate, symbol_stress, test, cost_bps * stress_cost_multiplier
        )
        train_trades.extend(symbol_train)
        oos_trades.extend(symbol_oos)
        stress_oos_trades.extend(symbol_stress)
        per_symbol[symbol] = {
            "oos": metrics(symbol_oos),
            "stress_oos": metrics(symbol_stress),
        }

    train_metrics = metrics(train_trades, train_days)
    oos_metrics = metrics(oos_trades, oos_days)
    stress_metrics = metrics(stress_oos_trades, oos_days)

    def _breakdown(rows: List[dict], key: str) -> Dict[str, dict]:
        groups: Dict[str, List[dict]] = {}
        for trade in rows:
            label = str(trade.get(key) or "unknown")
            groups.setdefault(label, []).append(trade)
        return {label: metrics(group) for label, group in groups.items()}

    def _combo_breakdown(rows: List[dict], a: str, b: str) -> Dict[str, dict]:
        groups: Dict[str, List[dict]] = {}
        for trade in rows:
            left = str(trade.get(a) or "unknown")
            right = str(trade.get(b) or "unknown")
            groups.setdefault(f"{left}||{right}", []).append(trade)
        return {label: metrics(group) for label, group in groups.items()}

    def _triple_breakdown(
        rows: List[dict], a: str, b: str, c: str
    ) -> Dict[str, dict]:
        groups: Dict[str, List[dict]] = {}
        for trade in rows:
            labels = [
                str(trade.get(a) or "unknown"),
                str(trade.get(b) or "unknown"),
                str(trade.get(c) or "unknown"),
            ]
            groups.setdefault("||".join(labels), []).append(trade)
        return {label: metrics(group) for label, group in groups.items()}

    def _management_breakdown(rows: List[dict]) -> tuple[Dict[str, dict], Dict[str, dict]]:
        by_combo: Dict[str, List[dict]] = {}
        by_profile: Dict[str, List[dict]] = {}
        for trade in rows:
            ctx = str(trade.get("context_key") or "unknown")
            model = str(trade.get("entry_model") or "unknown")
            for profile, outcome in dict(trade.get("management_outcomes") or {}).items():
                synthetic = dict(trade)
                synthetic["net_return"] = float((outcome or {}).get("net_return") or 0.0)
                synthetic["r_multiple"] = float((outcome or {}).get("r_multiple") or 0.0)
                by_combo.setdefault(f"{ctx}||{model}||{profile}", []).append(synthetic)
                by_profile.setdefault(str(profile), []).append(synthetic)
        return (
            {label: metrics(group) for label, group in by_combo.items()},
            {label: metrics(group) for label, group in by_profile.items()},
        )

    def _exit_counterfactual_breakdown(
        rows: List[dict],
    ) -> tuple[Dict[str, dict], Dict[str, dict]]:
        by_combo: Dict[str, List[dict]] = {}
        by_profile: Dict[str, List[dict]] = {}
        for trade in rows:
            ctx = str(trade.get("context_key") or "unknown")
            model = str(trade.get("entry_model") or "unknown")
            for profile, outcome in dict(trade.get("exit_outcomes") or {}).items():
                synthetic = dict(trade)
                synthetic["net_return"] = float(
                    (outcome or {}).get("net_return") or 0.0
                )
                synthetic["r_multiple"] = float(
                    (outcome or {}).get("r_multiple") or 0.0
                )
                by_combo.setdefault(
                    f"{ctx}||{model}||{profile}", []
                ).append(synthetic)
                by_profile.setdefault(str(profile), []).append(synthetic)
        return (
            {label: metrics(group) for label, group in by_combo.items()},
            {label: metrics(group) for label, group in by_profile.items()},
        )

    entry_model_breakdown = _breakdown(oos_trades, "entry_model")
    regime_breakdown = _breakdown(oos_trades, "entry_regime")
    context_breakdown = _breakdown(oos_trades, "context_key")
    exit_model_breakdown = _breakdown(oos_trades, "exit_model")
    context_entry_breakdown = _combo_breakdown(oos_trades, "context_key", "entry_model")
    context_exit_breakdown = _combo_breakdown(oos_trades, "context_key", "exit_model")
    time_breakdown = _breakdown(oos_trades, "time_bucket")
    volume_breakdown = _breakdown(oos_trades, "volume_bucket")
    volatility_breakdown = _breakdown(oos_trades, "volatility_bucket")
    trend_direction_breakdown = _breakdown(oos_trades, "trend_direction")
    context_entry_time_breakdown = _triple_breakdown(
        oos_trades, "context_key", "entry_model", "time_bucket"
    )
    context_entry_volume_breakdown = _triple_breakdown(
        oos_trades, "context_key", "entry_model", "volume_bucket"
    )
    context_entry_management_breakdown, management_model_breakdown = _management_breakdown(oos_trades)
    context_entry_exit_breakdown, exit_counterfactual_breakdown = (
        _exit_counterfactual_breakdown(oos_trades)
    )
    # Distillation may learn from all data before the untouched final
    # holdout. OOS metrics below remain the quality gate; these combined
    # breakdowns are used only to establish that a context has enough sample.
    research_trades = train_trades + oos_trades
    research_context_entry_breakdown = _combo_breakdown(
        research_trades, "context_key", "entry_model"
    )
    research_context_entry_management_breakdown, _ = _management_breakdown(
        research_trades
    )
    research_context_entry_exit_breakdown, _ = _exit_counterfactual_breakdown(
        research_trades
    )
    stress_context_breakdown = _breakdown(stress_oos_trades, "context_key")
    stress_context_entry_breakdown = _combo_breakdown(
        stress_oos_trades, "context_key", "entry_model"
    )
    stress_context_exit_breakdown = _combo_breakdown(
        stress_oos_trades, "context_key", "exit_model"
    )
    stress_context_entry_management_breakdown, _ = _management_breakdown(stress_oos_trades)
    stress_context_entry_exit_breakdown, _ = _exit_counterfactual_breakdown(
        stress_oos_trades
    )
    stress_context_entry_time_breakdown = _triple_breakdown(
        stress_oos_trades, "context_key", "entry_model", "time_bucket"
    )
    stress_context_entry_volume_breakdown = _triple_breakdown(
        stress_oos_trades, "context_key", "entry_model", "volume_bucket"
    )

    ensemble_pass = True
    ensemble_positive_models: List[str] = []
    ensemble_active_models: List[str] = []
    ensemble_dominant_share = 0.0
    if candidate.strategy in {"regime_ensemble", "adaptive_router"}:
        total_model_trades = max(1, int(oos_metrics.get("trades") or 0))
        min_component_trades = 3
        ensemble_active_models = [
            name for name, row in entry_model_breakdown.items()
            if int(row.get("trades") or 0) >= min_component_trades
        ]
        ensemble_positive_models = [
            name for name in ensemble_active_models
            if float(entry_model_breakdown[name].get("expectancy_bps") or 0.0) > 0
        ]
        ensemble_dominant_share = max(
            [
                float(row.get("trades") or 0) / total_model_trades
                for row in entry_model_breakdown.values()
            ] or [0.0]
        )
        ensemble_pass = (
            len(ensemble_active_models) >= 2
            and len(ensemble_positive_models) >= 2
            and ensemble_dominant_share <= 0.90
        )

    router_active_contexts = [
        name for name, row in context_breakdown.items()
        if name.split("|", 1)[0] in {"expansion", "trend", "range"}
        and int(row.get("trades") or 0) >= 3
    ]
    router_positive_contexts = [
        name for name in router_active_contexts
        if float(context_breakdown[name].get("expectancy_bps") or 0.0) > 0
    ]
    adaptive_bootstrap = (
        candidate.strategy == "adaptive_router"
        and bool(candidate.params.get("_adaptive_research_mode", False))
    )
    learned_route_count = len(
        dict((candidate.params.get("adaptive_policy") or {}).get("routes") or {})
    )
    router_pass = (
        candidate.strategy != "adaptive_router"
        or (
            not adaptive_bootstrap
            and learned_route_count > 0
            and ensemble_pass
            and len(router_active_contexts) >= 2
            and len(router_positive_contexts) >= 2
        )
    )

    positive_symbols = sum(
        1 for row in per_symbol.values()
        if row["oos"]["trades"] > 0 and row["oos"]["expectancy_bps"] > 0
    )
    symbol_count = max(1, len(per_symbol))
    positive_symbol_ratio = positive_symbols / symbol_count

    reasons: List[str] = []
    if train_metrics["expectancy_bps"] <= 0:
        reasons.append("negative in-sample expectancy")
    if oos_metrics["trades"] < min_oos_trades:
        reasons.append(f"fewer than {min_oos_trades} out-of-sample trades")
    hard_frequency_pass = float(oos_metrics.get("avg_trades_per_day") or 0.0) >= float(min_trades_per_day)
    if not hard_frequency_pass:
        reasons.append(f"average trades/day below hard minimum {float(min_trades_per_day):g}")

    allowed_loss_streak = (
        int(max_loss_streak_asymmetric)
        if candidate.strategy == "asymmetric_breakout"
        else int(max_loss_streak)
    )
    observed_loss_streak = int(oos_metrics.get("max_loss_streak") or 0)
    streak_pass = observed_loss_streak <= allowed_loss_streak
    if not streak_pass:
        reasons.append(
            f"out-of-sample max loss streak {observed_loss_streak} exceeds {allowed_loss_streak}"
        )
    if oos_metrics["expectancy_bps"] <= 0:
        reasons.append("negative out-of-sample expectancy")
    if oos_metrics["profit_factor"] < min_profit_factor:
        reasons.append(f"out-of-sample profit factor below {min_profit_factor:.2f}")
    if oos_metrics["max_drawdown_pct"] > max_drawdown_pct:
        reasons.append(f"out-of-sample drawdown above {max_drawdown_pct:.1f}%")
    if positive_symbol_ratio < min_positive_symbol_ratio:
        reasons.append(
            f"positive on only {positive_symbols}/{len(per_symbol)} symbols "
            f"(< {min_positive_symbol_ratio:.0%})"
        )
    if stress_metrics["expectancy_bps"] <= 0:
        reasons.append("fails stressed transaction-cost test")

    deep_oos_days = int(oos_metrics.get("trading_days") or 0)
    if candidate_phase == "deep_search" and deep_oos_days < MIN_DEEP_OOS_DAYS:
        reasons.append(
            f"final holdout has fewer than {MIN_DEEP_OOS_DAYS} trading days"
        )
    if candidate.strategy in {"regime_ensemble", "adaptive_router"} and not ensemble_pass:
        reasons.append(
            f"{candidate.strategy} lacks robust multi-entry contribution "
            f"(active={len(ensemble_active_models)}, positive={len(ensemble_positive_models)}, "
            f"dominant_share={ensemble_dominant_share:.0%})"
        )
    if adaptive_bootstrap:
        reasons.append(
            "adaptive research bootstrap collects entry/exit/management evidence only; "
            "it cannot be promoted"
        )
    if candidate.strategy == "adaptive_router" and not router_pass:
        reasons.append(
            "adaptive router lacks positive evidence across at least two market contexts "
            f"(active_contexts={len(router_active_contexts)}, "
            f"positive_contexts={len(router_positive_contexts)})"
        )

    raw_pass = not reasons
    score = funnel_score(
        oos_metrics,
        stress_metrics,
        positive_symbol_ratio,
        min_oos_trades,
        preferred_trades_per_day,
        target_trades_per_day,
    )
    discovery_ok = (
        hard_frequency_pass
        and streak_pass
        and ensemble_pass
        and router_pass
        and oos_metrics["trades"] >= 15
        and (
            oos_metrics["expectancy_bps"] > 0
            or oos_metrics["profit_factor"] >= 1.05
        )
        and oos_metrics["max_drawdown_pct"] <= 12.0
    )
    incubator_ok = (
        hard_frequency_pass
        and streak_pass
        and ensemble_pass
        and router_pass
        and oos_metrics["trades"] >= max(20, min_oos_trades // 2)
        and oos_metrics["expectancy_bps"] > 0
        and oos_metrics["profit_factor"] >= 1.20
        and stress_metrics["expectancy_bps"] > -1.0
        and positive_symbol_ratio >= 0.40
    )

    if candidate_phase == "deep_search":
        promoted = raw_pass
        funnel_stage = "promoted" if promoted else "rejected"
    elif candidate_phase == "incubator":
        promoted = False
        funnel_stage = "deep_search" if (raw_pass or incubator_ok) else "rejected"
        if funnel_stage == "deep_search" and raw_pass:
            reasons.append("incubator passed; exact parameters frozen for final holdout")
    else:
        promoted = False
        funnel_stage = "incubator" if (raw_pass or incubator_ok or discovery_ok) else "rejected"
        if funnel_stage == "incubator" and raw_pass:
            reasons.append("discovery passed; local stability variants required")

    return {
        "strategy": candidate.strategy,
        "family": candidate.strategy,
        "params": candidate.params,
        "promoted": promoted,
        "funnel_stage": funnel_stage,
        "funnel_score": score,
        "rejection_reasons": reasons,
        "train": train_metrics,
        "oos": oos_metrics,
        "stress_oos": stress_metrics,
        "positive_symbol_ratio": round(positive_symbol_ratio, 3),
        "positive_symbols": positive_symbols,
        "symbol_count": len(per_symbol),
        "per_symbol": per_symbol,
        "validation_policy": {
            "policy_version": RESEARCH_POLICY_VERSION,
            "phase": candidate_phase,
            "final_holdout_fraction": FINAL_HOLDOUT_FRACTION,
            "final_holdout_days": deep_oos_days if candidate_phase == "deep_search" else None,
            "deep_search_parameters_frozen": candidate_phase == "deep_search",
        },
        "frequency_policy": {
            "hard_min_trades_per_day": float(min_trades_per_day),
            "preferred_from_trades_per_day": float(preferred_trades_per_day),
            "target_trades_per_day": float(target_trades_per_day),
        },
        "streak_policy": {
            "max_loss_streak_allowed": allowed_loss_streak,
            "max_loss_streak_observed": observed_loss_streak,
            "max_win_streak_observed": int(oos_metrics.get("max_win_streak") or 0),
            "passed": streak_pass,
        },
        "entry_model_breakdown": entry_model_breakdown,
        "regime_breakdown": regime_breakdown,
        "ensemble_policy": {
            "applies": candidate.strategy in {"regime_ensemble", "adaptive_router"},
            "passed": ensemble_pass,
            "active_models": ensemble_active_models,
            "positive_models": ensemble_positive_models,
            "dominant_trade_share": round(ensemble_dominant_share, 3),
        },
        "adaptive_diagnostics": {
            "context_version": ADAPTIVE_CONTEXT_VERSION,
            "context_breakdown": context_breakdown,
            "entry_model_breakdown": entry_model_breakdown,
            "exit_model_breakdown": exit_model_breakdown,
            "context_entry_breakdown": context_entry_breakdown,
            "research_context_entry_breakdown": research_context_entry_breakdown,
            "context_exit_breakdown": context_exit_breakdown,
            "time_breakdown": time_breakdown,
            "volume_breakdown": volume_breakdown,
            "volatility_breakdown": volatility_breakdown,
            "trend_direction_breakdown": trend_direction_breakdown,
            "context_entry_time_breakdown": context_entry_time_breakdown,
            "context_entry_volume_breakdown": context_entry_volume_breakdown,
            "context_entry_management_breakdown": context_entry_management_breakdown,
            "research_context_entry_management_breakdown": research_context_entry_management_breakdown,
            "management_model_breakdown": management_model_breakdown,
            "context_entry_exit_breakdown": context_entry_exit_breakdown,
            "research_context_entry_exit_breakdown": research_context_entry_exit_breakdown,
            "exit_counterfactual_breakdown": exit_counterfactual_breakdown,
            "stress_context_breakdown": stress_context_breakdown,
            "stress_context_entry_breakdown": stress_context_entry_breakdown,
            "stress_context_exit_breakdown": stress_context_exit_breakdown,
            "stress_context_entry_management_breakdown": stress_context_entry_management_breakdown,
            "stress_context_entry_exit_breakdown": stress_context_entry_exit_breakdown,
            "stress_context_entry_time_breakdown": stress_context_entry_time_breakdown,
            "stress_context_entry_volume_breakdown": stress_context_entry_volume_breakdown,
            "management_profiles": MANAGEMENT_PROFILES,
            "exit_profiles": EXIT_PROFILES,
            "lifecycle_summary": {
                "avg_mfe_r": round(mean([float(t.get("mfe_r") or 0.0) for t in oos_trades]), 3) if oos_trades else 0.0,
                "avg_mae_r": round(mean([float(t.get("mae_r") or 0.0) for t in oos_trades]), 3) if oos_trades else 0.0,
                "avg_giveback_r": round(mean([float(t.get("giveback_r") or 0.0) for t in oos_trades]), 3) if oos_trades else 0.0,
                "reached_1_5r_then_lost": sum(
                    1 for t in oos_trades
                    if float(t.get("mfe_r") or 0.0) >= 1.5 and float(t.get("net_return") or 0.0) < 0
                ),
            },
            "router_passed": router_pass,
            "active_contexts": router_active_contexts,
            "positive_contexts": router_positive_contexts,
            "observed_exit_profiles": {
                str(t.get("exit_model")): dict(t.get("exit_profile") or {})
                for t in oos_trades
                if t.get("exit_model") and t.get("exit_profile")
            },
        },
    }


def funnel_score(
    oos: dict,
    stress: dict,
    positive_ratio: float,
    min_trades: int,
    preferred_trades_per_day: float = 5.0,
    target_trades_per_day: float = 10.0,
) -> float:
    exp = max(0.0, min(35.0, 17.5 + float(oos.get("expectancy_bps", 0))))
    pf = min(25.0, max(0.0, (float(oos.get("profit_factor", 0)) - 0.8) * 25.0))
    stress_score = min(15.0, max(0.0, 7.5 + float(stress.get("expectancy_bps", 0))))
    robustness = 15.0 * max(0.0, min(1.0, positive_ratio))
    trade_sample = 5.0 * min(1.0, float(oos.get("trades", 0)) / max(1, min_trades))
    avg_tpd = float(oos.get("avg_trades_per_day") or 0.0)
    if avg_tpd >= preferred_trades_per_day:
        span = max(0.1, target_trades_per_day - preferred_trades_per_day)
        frequency_bonus = 2.5 + 2.5 * min(
            1.0, max(0.0, (avg_tpd - preferred_trades_per_day) / span)
        )
    else:
        frequency_bonus = 0.0
    return round(min(100.0, exp + pf + stress_score + robustness + trade_sample + frequency_bonus), 2)


def simulate(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    if candidate.strategy == "momentum":
        return _simulate_momentum(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "mean_reversion":
        return _simulate_mean_reversion(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "breakout":
        return _simulate_breakout(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "extreme_reversal":
        return _simulate_extreme_reversal(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "volatility_breakout":
        return _simulate_volatility_breakout(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "trend_pullback":
        return _simulate_trend_pullback(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "regime_ensemble":
        return _simulate_regime_ensemble(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "adaptive_router":
        return _simulate_adaptive_router(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "vwap_reversion":
        return _simulate_vwap_reversion(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "vwap_momentum":
        return _simulate_vwap_momentum(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "asymmetric_breakout":
        return _simulate_asymmetric_breakout(candidate, symbol, bars, cost_bps)
    raise ValueError(f"Unknown strategy: {candidate.strategy}")


def _simulate_momentum(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    slow = int(p["slow"])
    trades: List[dict] = []
    i = slow

    while i < len(bars) - 1:
        if not _entry_ok(p, bars, i):
            i += 1
            continue
        history = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(history[-int(p["fast"]):])
        slow_ma = mean(history)
        edge_bps = ((fast_ma / slow_ma) - 1.0) * 10_000 if slow_ma else 0.0
        threshold = float(p["entry_bps"])
        direction = 1 if edge_bps >= threshold else (-1 if edge_bps <= -threshold else 0)
        if not direction:
            i += 1
            continue

        entry_idx = i
        entry_price = float(bars[entry_idx]["o"])
        exit_idx = min(entry_idx + int(p["max_hold"]), len(bars) - 1)

        for j in range(entry_idx + 1, exit_idx + 1):
            trailing = [float(x["c"]) for x in bars[max(0, j - slow):j]]
            if len(trailing) < slow:
                continue
            f = mean(trailing[-int(p["fast"]):])
            s = mean(trailing)
            edge = ((f / s) - 1.0) * 10_000 if s else 0.0
            if (direction > 0 and edge <= 0) or (direction < 0 and edge >= 0):
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(
            symbol, bars[entry_idx], bars[exit_idx],
            entry_price, exit_price, cost_bps, direction=direction
        ))
        i = exit_idx + 1
    return trades

def _simulate_mean_reversion(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    window = int(p["window"])
    trades: List[dict] = []
    i = window

    while i < len(bars) - 1:
        if not _entry_ok(p, bars, i):
            i += 1
            continue
        history = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(history)
        sigma = pstdev(history)
        if sigma <= 0:
            i += 1
            continue
        z = (float(bars[i - 1]["c"]) - mu) / sigma
        z_entry = float(p["z_entry"])
        direction = 1 if z <= -z_entry else (-1 if z >= z_entry else 0)
        if not direction:
            i += 1
            continue

        entry_idx = i
        entry_price = float(bars[entry_idx]["o"])
        exit_idx = min(entry_idx + int(p["max_hold"]), len(bars) - 1)

        for j in range(entry_idx + 1, exit_idx + 1):
            trailing = [float(x["c"]) for x in bars[max(0, j - window):j]]
            if len(trailing) < window:
                continue
            mu_j = mean(trailing)
            sigma_j = pstdev(trailing)
            if sigma_j <= 0:
                continue
            z_j = (float(bars[j - 1]["c"]) - mu_j) / sigma_j
            z_exit = float(p["z_exit"])
            if (direction > 0 and z_j >= -z_exit) or (direction < 0 and z_j <= z_exit):
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(
            symbol, bars[entry_idx], bars[exit_idx],
            entry_price, exit_price, cost_bps, direction=direction
        ))
        i = exit_idx + 1
    return trades

def _entry_ok(params: dict, bars: List[dict], i: int) -> bool:
    allowed, _, _ = entry_allowed(
        bars,
        i,
        min_volume_ratio=float(params.get("min_volume_ratio", 0.70)),
        volume_window=int(params.get("volume_window", 50)),
    )
    return allowed


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals = []
    for j in range(max(1, i-window), i):
        h, l = float(bars[j]["h"]), float(bars[j]["l"])
        prev = float(bars[j-1]["c"])
        vals.append(max(h-l, abs(h-prev), abs(l-prev)))
    return mean(vals) if vals else 0.0


def _simulate_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w+1
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        prior=bars[i-w-1:i-1]
        buf=float(p.get("buffer_bps",0))/10000
        high_level=max(float(x["h"]) for x in prior)*(1+buf)
        low_level=min(float(x["l"]) for x in prior)*(1-buf)
        px=float(bars[i-1]["c"])
        direction=1 if px>high_level else (-1 if px<low_level else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades

def _simulate_extreme_reversal(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w+1
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        rets=[float(bars[j]["c"])/float(bars[j-1]["c"])-1 for j in range(i-w,i)]
        sd=pstdev(rets) if len(rets)>1 else 0.0
        if sd<=0: i+=1; continue
        threshold=float(p["shock_z"])*sd
        direction=1 if rets[-1] <= -threshold else (-1 if rets[-1] >= threshold else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades

def _simulate_volatility_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w+1
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        ranges=[float(x["h"])-float(x["l"]) for x in bars[i-w:i]]
        cur=float(bars[i-1]["h"])-float(bars[i-1]["l"])
        if mean(ranges)<=0 or cur < mean(ranges)*float(p["vol_mult"]): i+=1; continue
        o=float(bars[i-1]["o"]); cl=float(bars[i-1]["c"])
        direction=1 if cl>o else (-1 if cl<o else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades

def _simulate_trend_pullback(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; slow=int(p["slow"]); fast=int(p["fast"]); trades=[]; i=slow
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        closes=[float(x["c"]) for x in bars[i-slow:i]]
        f=mean(closes[-fast:]); s=mean(closes); sd=pstdev(closes)
        z=(closes[-1]-f)/sd if sd>0 else 0
        threshold=float(p["pullback_z"])
        direction=1 if (f>s and z<=-threshold) else (-1 if (f<s and z>=threshold) else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades


def _regime_ensemble_signal(params: dict, bars: List[dict], i: int) -> tuple[int, str, str]:
    """Causal regime router: returns (direction, entry_model, regime).

    The router uses only bars strictly before entry index i. It chooses a
    deterministic entry model; no agent/model is allowed to pick trades after
    seeing their outcome.
    """
    atr_short_n = int(params.get("regime_atr_short", 14))
    atr_long_n = int(params.get("regime_atr_long", 50))
    slow = int(params.get("slow", 30))
    fast = int(params.get("fast", 8))
    breakout_window = int(params.get("breakout_window", 20))
    vwap_window = int(params.get("vwap_window", 60))
    need = max(atr_long_n + 1, slow, breakout_window + 1, vwap_window)
    if i < need:
        return 0, "none", "warmup"

    atr_short = _atr(bars, i, atr_short_n)
    atr_long = _atr(bars, i, atr_long_n)
    if atr_short <= 0 or atr_long <= 0:
        return 0, "none", "invalid"

    closes = [float(x["c"]) for x in bars[i-slow:i]]
    fast_ma = mean(closes[-fast:])
    slow_ma = mean(closes)
    trend_strength = abs(fast_ma - slow_ma) / atr_long
    vol_ratio = atr_short / atr_long

    # 1) Volatility expansion -> price breakout entry.
    if vol_ratio >= float(params.get("regime_vol_ratio", 1.20)):
        prior = bars[i-breakout_window-1:i-1]
        if not prior:
            return 0, "breakout", "expansion"
        px = float(bars[i-1]["c"])
        buf = float(params.get("buffer_bps", 2.0)) / 10_000.0
        hi = max(float(x["h"]) for x in prior) * (1.0 + buf)
        lo = min(float(x["l"]) for x in prior) * (1.0 - buf)
        direction = 1 if px > hi else (-1 if px < lo else 0)
        return direction, "breakout", "expansion"

    # 2) Directional market -> pullback entry in prevailing trend.
    if trend_strength >= float(params.get("regime_trend_atr", 0.55)):
        sd = pstdev(closes)
        z = (closes[-1] - fast_ma) / sd if sd > 0 else 0.0
        threshold = float(params.get("pullback_z", 1.0))
        direction = (
            1 if (fast_ma > slow_ma and z <= -threshold)
            else (-1 if (fast_ma < slow_ma and z >= threshold) else 0)
        )
        return direction, "trend_pullback", "trend"

    # 3) Quiet/ranging market -> VWAP mean-reversion entry.
    vw = _rolling_vwap(bars, i-vwap_window, i)
    range_closes = [float(x["c"]) for x in bars[i-vwap_window:i]]
    sd = pstdev(range_closes)
    z = (range_closes[-1] - vw) / sd if sd > 0 else 0.0
    threshold = float(params.get("z_entry", 1.4))
    direction = 1 if z <= -threshold else (-1 if z >= threshold else 0)
    return direction, "vwap_reversion", "range"


def _simulate_regime_ensemble(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    """Common risk/exit model, varying only the regime-routed entry model."""
    p = candidate.params
    warmup = max(
        int(p.get("regime_atr_long", 50)) + 1,
        int(p.get("slow", 30)),
        int(p.get("breakout_window", 20)) + 1,
        int(p.get("vwap_window", 60)),
    )
    trades: List[dict] = []
    i = warmup
    while i < len(bars) - 2:
        if not _entry_ok(p, bars, i):
            i += 1
            continue
        direction, entry_model, regime = _regime_ensemble_signal(p, bars, i)
        if not direction:
            i += 1
            continue

        entry = float(bars[i]["o"])
        atr = _atr(bars, i, 14)
        if atr <= 0:
            i += 1
            continue
        risk = atr * float(p.get("stop_atr", 1.0))
        target_r = float(p.get("target_r", 2.0))
        stop = _max_loss_stop_price(
            entry=entry,
            risk_distance=risk,
            direction=direction,
            cost_bps=float(cost_bps),
            max_loss_r=1.0,
        )
        target = entry + risk * target_r if direction > 0 else entry - risk * target_r

        e = i
        x = min(e + int(p.get("max_hold", 24)), len(bars) - 1)
        exit_price = float(bars[x]["o"])
        r_mult = None
        exit_reason = "max_hold"
        for j in range(e, x + 1):
            lo = float(bars[j]["l"])
            hi = float(bars[j]["h"])
            stop_hit = (lo <= stop) if direction > 0 else (hi >= stop)
            target_hit = (hi >= target) if direction > 0 else (lo <= target)
            # Conservative same-bar assumption: stop is evaluated first.
            if stop_hit:
                exit_price = stop
                x = j
                r_mult = _net_r_multiple(
                    entry=entry, exit_price=stop, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                )
                exit_reason = "stop"
                break
            if target_hit:
                exit_price = target
                x = j
                r_mult = _net_r_multiple(
                    entry=entry, exit_price=target, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                )
                exit_reason = "target"
                break
        if r_mult is None:
            r_mult = _net_r_multiple(
                entry=entry, exit_price=exit_price, risk_distance=risk,
                direction=direction, cost_bps=float(cost_bps),
            )

        trade = _trade(
            symbol, bars[e], bars[x], entry, exit_price, cost_bps,
            r_mult, direction=direction,
        )
        trade["risk_distance"] = risk
        trade["entry_model"] = entry_model
        trade["entry_regime"] = regime
        trade["exit_reason"] = exit_reason
        trades.append(trade)
        i = x + 1
    return trades


def _simulate_adaptive_router(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    """Context-aware multi-entry router with independently selected RR exits."""
    p = candidate.params
    warmup = max(
        int(p.get("regime_atr_long", 50)) + 1,
        int(p.get("slow", 30)),
        int(p.get("vwap_window", 60)),
        61,
    )
    trades: List[dict] = []
    i = warmup
    while i < len(bars) - 2:
        if not _entry_ok(p, bars, i):
            i += 1
            continue
        direction, entry_model, context, exit_profile, route_meta = adaptive_signal(
            p, bars, i
        )
        if not direction:
            i += 1
            continue

        entry = float(bars[i]["o"])
        atr = _atr(bars, i, 14)
        stop_atr = max(0.10, float(exit_profile.get("stop_atr", 1.0)))
        risk = atr * stop_atr
        if risk <= 0:
            i += 1
            continue
        target_r = max(0.25, float(exit_profile.get("target_r", 2.0)))
        max_hold = max(2, int(exit_profile.get("max_hold", 24)))
        stop = _max_loss_stop_price(
            entry=entry,
            risk_distance=risk,
            direction=direction,
            cost_bps=float(cost_bps),
            max_loss_r=1.0,
        )
        target = entry + risk * target_r if direction > 0 else entry - risk * target_r

        e = i
        x = min(e + max_hold, len(bars) - 1)
        exit_price = float(bars[x]["o"])
        r_mult = None
        exit_reason = "max_hold"
        management = dict(route_meta.get("management_profile") or {"name": "baseline"})
        trigger_r = management.get("management_trigger_r")
        lock_net_r = management.get("management_lock_net_r")
        protection_active = False
        risk_pct = risk / entry if entry > 0 else 0.0
        cost_r = (
            (2.0 * float(cost_bps) / 10_000.0) / risk_pct
            if risk_pct > 0 else 0.0
        )
        for j in range(e, x + 1):
            lo = float(bars[j]["l"])
            hi = float(bars[j]["h"])
            stop_hit = (lo <= stop) if direction > 0 else (hi >= stop)
            target_hit = (hi >= target) if direction > 0 else (lo <= target)
            if stop_hit:
                exit_price = stop
                x = j
                r_mult = _net_r_multiple(
                    entry=entry, exit_price=stop, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                )
                exit_reason = "management_stop" if protection_active else "stop"
                break
            if target_hit:
                exit_price = target
                x = j
                r_mult = _net_r_multiple(
                    entry=entry, exit_price=target, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                )
                exit_reason = "target"
                break
            if (
                trigger_r is not None
                and lock_net_r is not None
                and not protection_active
            ):
                trigger_price = (
                    entry + risk * float(trigger_r)
                    if direction > 0 else entry - risk * float(trigger_r)
                )
                reached = hi >= trigger_price if direction > 0 else lo <= trigger_price
                if reached:
                    lock_gross_r = float(lock_net_r) + cost_r
                    protected = (
                        entry + risk * lock_gross_r
                        if direction > 0 else entry - risk * lock_gross_r
                    )
                    stop = max(stop, protected) if direction > 0 else min(stop, protected)
                    protection_active = True
        if r_mult is None:
            r_mult = _net_r_multiple(
                entry=entry, exit_price=exit_price, risk_distance=risk,
                direction=direction, cost_bps=float(cost_bps),
            )

        trade = _trade(
            symbol, bars[e], bars[x], entry, exit_price, cost_bps,
            r_mult, direction=direction,
        )
        trade["risk_distance"] = risk
        trade["entry_model"] = entry_model
        trade["entry_regime"] = str(context.get("regime") or "unknown")
        trade["entry_context"] = context
        trade["context_key"] = context_key(context)
        trade["exit_model"] = str(exit_profile.get("name") or "adaptive_rr")
        trade["exit_profile"] = dict(exit_profile)
        trade["exit_reason"] = exit_reason
        trade["route_evidence_score"] = float(route_meta.get("evidence_score") or 0.0)
        trade["route_source_strategy"] = str(route_meta.get("source_strategy") or entry_model)
        trade["management_profile"] = dict(route_meta.get("management_profile") or {"name": "baseline"})
        trade["management_model"] = str(trade["management_profile"].get("name") or "baseline")
        trades.append(trade)
        i = x + 1
    return trades


def _rolling_vwap(bars: List[dict], a: int, b: int) -> float:
    pv=0.0; vol=0.0
    for x in bars[a:b]:
        v=float(x.get("v") or 0); typ=(float(x["h"])+float(x["l"])+float(x["c"]))/3
        pv+=typ*v; vol+=v
    return pv/vol if vol>0 else mean(float(x["c"]) for x in bars[a:b])


def _simulate_vwap_reversion(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        vw=_rolling_vwap(bars,i-w,i); closes=[float(x["c"]) for x in bars[i-w:i]]; sd=pstdev(closes)
        z=(closes[-1]-vw)/sd if sd>0 else 0
        threshold=float(p["z_entry"])
        direction=1 if z<=-threshold else (-1 if z>=threshold else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades

def _simulate_vwap_momentum(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w
    while i < len(bars)-1:
        if not _entry_ok(p, bars, i): i+=1; continue
        vw=_rolling_vwap(bars,i-w,i); px=float(bars[i-1]["c"])
        edge=((px/vw)-1)*10000 if vw>0 else 0.0
        threshold=float(p["buffer_bps"])
        direction=1 if edge>=threshold else (-1 if edge<=-threshold else 0)
        if not direction: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps,direction=direction)); i=x+1
    return trades

def _simulate_asymmetric_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=max(w+1,15)
    while i < len(bars)-2:
        if not _entry_ok(p, bars, i): i+=1; continue
        prior=bars[i-w-1:i-1]
        prev=float(bars[i-1]["c"])
        high_break=max(float(x["h"]) for x in prior)
        low_break=min(float(x["l"]) for x in prior)
        direction=1 if prev>high_break else (-1 if prev<low_break else 0)
        if not direction: i+=1; continue
        entry=float(bars[i]["o"]); atr=_atr(bars,i,14)
        if atr<=0: i+=1; continue
        risk=atr*float(p["stop_atr"])
        target_r=float(p["target_r"])
        stop=_max_loss_stop_price(
            entry=entry,
            risk_distance=risk,
            direction=direction,
            cost_bps=float(cost_bps),
            max_loss_r=1.0,
        )
        target=entry+risk*target_r if direction>0 else entry-risk*target_r
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1); exit_price=float(bars[x]["o"]); r_mult=None
        for j in range(e,x+1):
            lo=float(bars[j]["l"]); hi=float(bars[j]["h"])
            stop_hit=(lo<=stop) if direction>0 else (hi>=stop)
            target_hit=(hi>=target) if direction>0 else (lo<=target)
            if stop_hit:
                exit_price=stop; x=j; r_mult=_net_r_multiple(
                    entry=entry, exit_price=stop, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                ); break
            if target_hit:
                exit_price=target; x=j; r_mult=_net_r_multiple(
                    entry=entry, exit_price=target, risk_distance=risk,
                    direction=direction, cost_bps=float(cost_bps),
                ); break
        if r_mult is None:
            r_mult=_net_r_multiple(
                entry=entry, exit_price=exit_price, risk_distance=risk,
                direction=direction, cost_bps=float(cost_bps),
            )
        trade=_trade(symbol,bars[e],bars[x],entry,exit_price,cost_bps,r_mult,direction=direction)
        trade["risk_distance"]=risk
        trades.append(trade); i=x+1
    return trades

def _trade(
    symbol: str,
    entry_bar: dict,
    exit_bar: dict,
    entry_price: float,
    exit_price: float,
    cost_bps: float,
    r_multiple: Optional[float] = None,
    direction: int = 1,
) -> dict:
    gross = direction * ((exit_price / entry_price) - 1.0) if entry_price else 0.0
    net = gross - (2.0 * cost_bps / 10_000.0)
    return {
        "symbol": symbol,
        "pair": symbol,
        "side": "long" if direction > 0 else "short",
        "direction": direction,
        "entry_time": entry_bar.get("t"),
        "exit_time": exit_bar.get("t"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross,
        "net_return": net,
        "r_multiple": r_multiple,
    }


def metrics(trades: List[dict], trading_days=None) -> dict:
    days = len({str(x)[:10] for x in (trading_days or []) if str(x)})
    if not days and trades:
        days = len({str(t.get("entry_time") or "")[:10] for t in trades if t.get("entry_time")})
    if not trades:
        return {
            "trades": 0,
            "trading_days": days,
            "avg_trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "compounded_trade_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_r": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    ordered = sorted(trades, key=lambda x: str(x.get("exit_time") or ""))
    returns = [float(t["net_return"]) for t in ordered]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= max(0.000001, 1.0 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    avg_win = mean(wins) if wins else 0.0
    avg_loss = abs(mean(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
    r_values = [float(t["r_multiple"]) for t in ordered if t.get("r_multiple") is not None]

    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    for r in returns:
        if r > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif r < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0

    return {
        "trades": len(returns),
        "trading_days": days,
        "avg_trades_per_day": round(len(returns) / days, 3) if days else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / len(returns), 2),
        "expectancy_bps": round(mean(returns) * 10_000.0, 3),
        "profit_factor": round(min(pf, 999.0), 3),
        "compounded_trade_return_pct": round((equity - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
        "avg_win_bps": round(avg_win * 10_000.0, 3),
        "avg_loss_bps": round(avg_loss * 10_000.0, 3),
        "payoff_ratio": round(min(payoff, 999.0), 3),
        "expectancy_r": round(mean(r_values), 3) if r_values else 0.0,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }
