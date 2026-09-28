from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from alpaca_client import AlpacaClient
from config import Settings
from strategy_store import StrategyStore

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
    def __init__(self, settings: Settings, client: AlpacaClient):
        self.settings = settings
        self.client = client
        self.state = LabState()
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._summary: dict = {}
        self._agent_focus: dict = {"families": [], "timeframes": [], "reason": ""}
        self.store = StrategyStore(settings.database_url)
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
        return payload

    def results(self) -> List[dict]:
        return list(self._results)

    def set_agent_focus(self, families: List[str], timeframes: List[int], reason: str = "") -> None:
        self._agent_focus = {
            "families": [str(x) for x in families][:6],
            "timeframes": [int(x) for x in timeframes if int(x) in {1, 3, 5, 15}][:4],
            "reason": str(reason)[:500],
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
            message="Preparing Strategy Lab",
            symbols_total=len(self.settings.lab_symbols),
            target_promoted=self.settings.lab_target_promoted,
        )
        self._results = []
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
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=self.settings.lab_lookback_days)
            bars_by_symbol: Dict[str, List[dict]] = {}
            self.state.stage = "loading_data"
            self.state.message = "Loading historical market data"
            for symbol in self.settings.lab_symbols:
                self.state.message = f"Loading {symbol}"
                bars = await self.client.historical_bars(
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=self.settings.lab_timeframe,
                    max_bars=self.settings.lab_max_bars_per_symbol,
                )
                if len(bars) >= 100:
                    bars_by_symbol[symbol] = bars
                self.state.symbols_loaded += 1
                await asyncio.sleep(0)

            if not bars_by_symbol:
                raise RuntimeError("No usable historical bars returned for Strategy Lab")

            await self.store.init()
            persisted = await self.store.load_results(limit=250)
            persisted_signatures = await self.store.load_signatures()
            persisted_state = await self.store.load_state()

            seen = set(persisted_signatures)
            results: List[dict] = list(persisted)
            promoted: List[dict] = [r for r in results if r.get("promoted")]
            batch_size = max(1, self.settings.lab_batch_size)

            self.state.generation = int(persisted_state.get("generation", 0))
            self.state.tested_total = max(
                int(persisted_state.get("tested_total", len(seen))),
                len(seen),
            )
            self.state.promoted_total = int(persisted_state.get("promoted_total", len(promoted)))
            self._results = list(results)
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
                    await asyncio.sleep(0.05)
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
                        "discovery": min(5000, self.settings.lab_max_bars_per_symbol),
                        "incubator": min(10000, self.settings.lab_max_bars_per_symbol),
                        "deep_search": self.settings.lab_max_bars_per_symbol,
                    }.get(candidate_phase, self.settings.lab_max_bars_per_symbol)
                    tf = int(candidate.params.get("timeframe_min", 1))
                    candidate_bars = {
                        s: aggregate_bars(v[-phase_bar_budget:], tf)
                        for s, v in bars_by_symbol.items()
                    }

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
                        self.settings.lab_cost_bps,
                        self.settings.lab_stress_cost_multiplier,
                        self.settings.lab_min_oos_trades,
                        self.settings.lab_min_profit_factor,
                        self.settings.lab_max_drawdown_pct,
                        self.settings.lab_min_positive_symbol_ratio,
                        symbol_progress,
                    )
                    elapsed = (datetime.now(timezone.utc) - started_candidate).total_seconds()
                    self.state.candidate_seconds = round(elapsed, 2)
                    self.state.last_completed_candidate = candidate.strategy
                    self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                    self.state.last_progress_at = self.state.last_completed_at
                    self.state.current_symbol = ""
                    log.info(
                        "Strategy Lab candidate complete: strategy=%s phase=%s seconds=%.2f funnel_stage=%s score=%s pf=%s exp_bps=%s",
                        candidate.strategy, candidate_phase, elapsed, result.get("funnel_stage"),
                        result.get("funnel_score"), (result.get("oos") or {}).get("profit_factor"),
                        (result.get("oos") or {}).get("expectancy_bps")
                    )
                    results.append(result)
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
                            float(row.get("funnel_score", 0)),
                            row["oos"]["expectancy_bps"],
                        ),
                        reverse=True,
                    )
                    self._results = results[:250]

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
                        "source_timeframe": self.settings.lab_timeframe,
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
                            key=lambda r: float(r.get("funnel_score", 0)),
                            reverse=True,
                        )[:10],
                        "best_candidate": promoted[0] if promoted else (results[0] if results else None),
                        "cost_bps_per_side": self.settings.lab_cost_bps,
                        "stress_cost_multiplier": self.settings.lab_stress_cost_multiplier,
                        "method": (
                            "continuous deterministic candidate search; chronological 70/30 holdout; "
                            "next-bar-open fills; long-only; per-symbol robustness filter"
                        ),
                    }
                    await asyncio.sleep(0)

                    if (
                        self.settings.lab_target_promoted > 0
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
        finally:
            self.state.running = False
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


def candidate_signature(candidate: Candidate) -> str:
    import json
    return candidate.strategy + ":" + json.dumps(candidate.params, sort_keys=True, separators=(",", ":"))


def discovery_candidates() -> List[Candidate]:
    out: List[Candidate] = []
    for tf in (1, 3, 5, 15):
        common = {"timeframe_min": tf, "_phase": "discovery"}
        out.extend([
            Candidate("momentum", {**common, "fast": 4, "slow": 16, "entry_bps": 8.0, "max_hold": 16}),
            Candidate("mean_reversion", {**common, "window": 20, "z_entry": 1.5, "z_exit": 0.25, "max_hold": 20}),
            Candidate("breakout", {**common, "window": 20, "buffer_bps": 3.0, "max_hold": 24}),
            Candidate("extreme_reversal", {**common, "window": 12, "shock_z": 2.0, "max_hold": 12}),
            Candidate("volatility_breakout", {**common, "window": 20, "vol_mult": 1.6, "max_hold": 20}),
            Candidate("trend_pullback", {**common, "fast": 8, "slow": 30, "pullback_z": 1.0, "max_hold": 24}),
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
        and k not in {"timeframe_min", "target_r"}
    ][:4]

    for key in keys:
        for offset in local_offsets:
            q = dict(base)
            v = base[key]
            nv = float(v) * (1.0 + offset)
            q[key] = max(1, int(round(nv))) if isinstance(v, int) else round(max(0.01, nv), 4)
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

def choose_batch(
    results: List[dict],
    seen: set[str],
    generation: int,
    batch_size: int,
    focus: Optional[dict] = None,
) -> List[Candidate]:
    focus = focus or {}
    focus_families = set(focus.get("families") or [])
    focus_timeframes = set(int(x) for x in (focus.get("timeframes") or []))
    pending_discovery = [
        c for c in discovery_candidates()
        if candidate_signature(c) not in seen
    ]
    pending_discovery.sort(
        key=lambda c: (
            c.strategy in focus_families if focus_families else False,
            int(c.params.get("timeframe_min", 0)) in focus_timeframes if focus_timeframes else False,
        ),
        reverse=True,
    )
    if pending_discovery:
        return pending_discovery[:batch_size]

    promising = [
        r for r in results
        if r.get("funnel_stage") in {"incubator", "deep_search", "promoted"}
        and r.get("params", {}).get("_phase") in {"discovery", "incubator", "deep_search"}
    ]
    promising.sort(
        key=lambda r: (
            r.get("strategy") in focus_families if focus_families else False,
            int((r.get("params") or {}).get("timeframe_min", 0)) in focus_timeframes if focus_timeframes else False,
            float(r.get("funnel_score", 0)),
        ),
        reverse=True,
    )

    phase = "deep_search" if any(
        r.get("params", {}).get("_phase") in {"incubator", "deep_search"}
        or r.get("funnel_stage") == "deep_search"
        for r in promising[:12]
    ) else "incubator"
    candidates: List[Candidate] = []
    for row in promising[:12]:
        candidates.extend(parameter_variants(row, phase, generation))

    unique = []
    local = set()
    for cand in candidates:
        sig = candidate_signature(cand)
        if sig in seen or sig in local:
            continue
        local.add(sig)
        unique.append(cand)
        if len(unique) >= batch_size:
            return unique

    # If nothing has shown promise, keep broadening Discovery. The previous
    # version clamped this at generation 6, which eventually exhausted every
    # signature and made the lab stop around ~760 tests.
    g = max(1, generation)
    epoch = max(0, (g - 1) // 6)
    wave = 1 + ((g - 1) % 6)
    for tf in (1, 3, 5, 15):
        broad = [
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
            Candidate("asymmetric_breakout", {"timeframe_min": tf, "_phase": "discovery",
                "window": 15 + 5*wave + 2*epoch, "stop_atr": round(0.4 + 0.05*wave + 0.01*epoch, 2),
                "target_r": float((3, 5, 8, 10, 12, 15)[(wave - 1) % 6]),
                "max_hold": 30 + 5*wave + 2*epoch}),
        ]
        for cand in broad:
            sig = candidate_signature(cand)
            if sig not in seen and sig not in local:
                local.add(sig)
                unique.append(cand)
                if len(unique) >= batch_size:
                    return unique
    return unique


def candidate_stream():
    for c in discovery_candidates():
        yield c


def candidate_grid() -> List[Candidate]:
    return discovery_candidates()


def aggregate_bars(bars: List[dict], minutes: int) -> List[dict]:
    if minutes <= 1:
        return list(bars)
    out: List[dict] = []
    bucket: List[dict] = []
    for bar in bars:
        bucket.append(bar)
        if len(bucket) < minutes:
            continue
        out.append({
            "t": bucket[-1].get("t"),
            "o": float(bucket[0]["o"]),
            "h": max(float(x["h"]) for x in bucket),
            "l": min(float(x["l"]) for x in bucket),
            "c": float(bucket[-1]["c"]),
            "v": sum(float(x.get("v") or 0) for x in bucket),
        })
        bucket = []
    return out


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
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_oos_trades: List[dict] = []
    per_symbol: Dict[str, dict] = {}

    symbol_items = list(bars_by_symbol.items())
    for symbol_index, (symbol, bars) in enumerate(symbol_items, start=1):
        if progress_callback:
            progress_callback(symbol_index, len(symbol_items), symbol)
        split = max(2, int(len(bars) * 0.70))
        train = bars[:split]
        test = bars[split:]

        symbol_train = simulate(candidate, symbol, train, cost_bps)
        symbol_oos = simulate(candidate, symbol, test, cost_bps)
        symbol_stress = simulate(candidate, symbol, test, cost_bps * stress_cost_multiplier)
        train_trades.extend(symbol_train)
        oos_trades.extend(symbol_oos)
        stress_oos_trades.extend(symbol_stress)
        per_symbol[symbol] = {
            "oos": metrics(symbol_oos),
            "stress_oos": metrics(symbol_stress),
        }

    train_metrics = metrics(train_trades)
    oos_metrics = metrics(oos_trades)
    stress_metrics = metrics(stress_oos_trades)
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

    raw_pass = not reasons
    candidate_phase = str(candidate.params.get("_phase", "discovery"))
    promoted = raw_pass and candidate_phase == "deep_search"
    if raw_pass and not promoted:
        reasons.append("passes current filters; requires deep-search full-history confirmation")
    score = funnel_score(oos_metrics, stress_metrics, positive_symbol_ratio, min_oos_trades)
    if promoted:
        funnel_stage = "promoted"
    elif raw_pass or (
        oos_metrics["trades"] >= max(20, min_oos_trades // 2)
        and oos_metrics["expectancy_bps"] > 0
        and oos_metrics["profit_factor"] >= 1.20
        and stress_metrics["expectancy_bps"] > -1.0
        and positive_symbol_ratio >= 0.40
    ):
        funnel_stage = "deep_search"
    elif (
        oos_metrics["trades"] >= 15
        and (oos_metrics["expectancy_bps"] > 0 or oos_metrics["profit_factor"] >= 1.05)
        and oos_metrics["max_drawdown_pct"] <= 12.0
    ):
        funnel_stage = "incubator"
    else:
        funnel_stage = "rejected"

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
    }


def funnel_score(oos: dict, stress: dict, positive_ratio: float, min_trades: int) -> float:
    exp = max(0.0, min(35.0, 17.5 + float(oos.get("expectancy_bps", 0))))
    pf = min(25.0, max(0.0, (float(oos.get("profit_factor", 0)) - 0.8) * 25.0))
    stress_score = min(15.0, max(0.0, 7.5 + float(stress.get("expectancy_bps", 0))))
    robustness = 15.0 * max(0.0, min(1.0, positive_ratio))
    trades = 10.0 * min(1.0, float(oos.get("trades", 0)) / max(1, min_trades))
    return round(exp + pf + stress_score + robustness + trades, 2)


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
        history = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(history[-int(p["fast"]):])
        slow_ma = mean(history)
        edge_bps = ((fast_ma / slow_ma) - 1.0) * 10_000 if slow_ma else 0.0
        if edge_bps < float(p["entry_bps"]):
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
            if s and ((f / s) - 1.0) * 10_000 <= 0:
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(symbol, bars[entry_idx], bars[exit_idx], entry_price, exit_price, cost_bps))
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
        history = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(history)
        sigma = pstdev(history)
        if sigma <= 0:
            i += 1
            continue
        z = (float(bars[i - 1]["c"]) - mu) / sigma
        if z > -float(p["z_entry"]):
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
            if z_j >= -float(p["z_exit"]):
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(symbol, bars[entry_idx], bars[exit_idx], entry_price, exit_price, cost_bps))
        i = exit_idx + 1
    return trades



def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals = []
    for j in range(max(1, i-window), i):
        h, l = float(bars[j]["h"]), float(bars[j]["l"])
        prev = float(bars[j-1]["c"])
        vals.append(max(h-l, abs(h-prev), abs(l-prev)))
    return mean(vals) if vals else 0.0


def _simulate_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w
    while i < len(bars)-1:
        level=max(float(x["h"]) for x in bars[i-w:i])*(1+float(p.get("buffer_bps",0))/10000)
        if float(bars[i-1]["c"]) <= level: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
    return trades


def _simulate_extreme_reversal(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w+1
    while i < len(bars)-1:
        rets=[float(bars[j]["c"])/float(bars[j-1]["c"])-1 for j in range(i-w,i)]
        sd=pstdev(rets) if len(rets)>1 else 0.0
        if sd<=0 or rets[-1] > -float(p["shock_z"])*sd: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
    return trades


def _simulate_volatility_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w+1
    while i < len(bars)-1:
        ranges=[float(x["h"])-float(x["l"]) for x in bars[i-w:i]]
        cur=float(bars[i-1]["h"])-float(bars[i-1]["l"])
        bullish=float(bars[i-1]["c"])>float(bars[i-1]["o"])
        if mean(ranges)<=0 or cur < mean(ranges)*float(p["vol_mult"]) or not bullish: i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
    return trades


def _simulate_trend_pullback(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; slow=int(p["slow"]); fast=int(p["fast"]); trades=[]; i=slow
    while i < len(bars)-1:
        closes=[float(x["c"]) for x in bars[i-slow:i]]
        f=mean(closes[-fast:]); s=mean(closes); sd=pstdev(closes)
        z=(closes[-1]-f)/sd if sd>0 else 0
        if f<=s or z > -float(p["pullback_z"]): i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
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
        vw=_rolling_vwap(bars,i-w,i); closes=[float(x["c"]) for x in bars[i-w:i]]; sd=pstdev(closes)
        z=(closes[-1]-vw)/sd if sd>0 else 0
        if z > -float(p["z_entry"]): i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
    return trades


def _simulate_vwap_momentum(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=w
    while i < len(bars)-1:
        vw=_rolling_vwap(bars,i-w,i); px=float(bars[i-1]["c"])
        if vw<=0 or ((px/vw)-1)*10000 < float(p["buffer_bps"]): i+=1; continue
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1)
        trades.append(_trade(symbol,bars[e],bars[x],float(bars[e]["o"]),float(bars[x]["o"]),cost_bps)); i=x+1
    return trades


def _simulate_asymmetric_breakout(candidate: Candidate, symbol: str, bars: List[dict], cost_bps: float) -> List[dict]:
    p=candidate.params; w=int(p["window"]); trades=[]; i=max(w,15)
    while i < len(bars)-2:
        prior=max(float(x["h"]) for x in bars[i-w:i])
        if float(bars[i-1]["c"]) <= prior: i+=1; continue
        entry=float(bars[i]["o"]); atr=_atr(bars,i,14)
        if atr<=0: i+=1; continue
        risk=atr*float(p["stop_atr"]); stop=entry-risk; target=entry+risk*float(p["target_r"])
        e=i; x=min(e+int(p["max_hold"]),len(bars)-1); exit_price=float(bars[x]["o"]); r_mult=None
        for j in range(e,x+1):
            lo=float(bars[j]["l"]); hi=float(bars[j]["h"])
            if lo<=stop:
                exit_price=stop; x=j; r_mult=-1.0; break
            if hi>=target:
                exit_price=target; x=j; r_mult=float(p["target_r"]); break
        if r_mult is None:
            r_mult=(exit_price-entry)/risk if risk>0 else 0.0
        trades.append(_trade(symbol,bars[e],bars[x],entry,exit_price,cost_bps,r_mult)); i=x+1
    return trades


def _trade(
    symbol: str,
    entry_bar: dict,
    exit_bar: dict,
    entry_price: float,
    exit_price: float,
    cost_bps: float,
    r_multiple: Optional[float] = None,
) -> dict:
    gross = (exit_price / entry_price) - 1.0 if entry_price else 0.0
    net = gross - (2.0 * cost_bps / 10_000.0)
    return {
        "symbol": symbol,
        "entry_time": entry_bar.get("t"),
        "exit_time": exit_bar.get("t"),
        "gross_return": gross,
        "net_return": net,
        "r_multiple": r_multiple,
    }


def metrics(trades: List[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "compounded_trade_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_r": 0.0,
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
    return {
        "trades": len(returns),
        "win_rate_pct": round(100.0 * len(wins) / len(returns), 2),
        "expectancy_bps": round(mean(returns) * 10_000.0, 3),
        "profit_factor": round(min(pf, 999.0), 3),
        "compounded_trade_return_pct": round((equity - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
        "avg_win_bps": round(avg_win * 10_000.0, 3),
        "avg_loss_bps": round(avg_loss * 10_000.0, 3),
        "payoff_ratio": round(min(payoff, 999.0), 3),
        "expectancy_r": round(mean(r_values), 3) if r_values else 0.0,
    }
