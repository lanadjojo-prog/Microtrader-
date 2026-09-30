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
    FOREX_EVALUATION_POLICY_VERSION,
    ForexCandidate,
    candidate_grid,
    candidate_signature,
    evaluate_candidate,
)
from forex_data import ExternalForexData
from forex_store import ForexStrategyStore

log = logging.getLogger("microtrader.forex_lab")


@dataclass
class ForexLabState:
    running: bool = False
    stage: str = "idle"
    message: str = "Ready for read-only forex research data"
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


def _clean_params(params: dict, phase: str) -> dict:
    out = dict(params or {})
    out["_phase"] = phase
    return out


def apply_frequency_gate(results: List[dict], min_trades_per_day: float) -> List[dict]:
    """Reclassify legacy saved rows against the current hard frequency rule."""
    out: List[dict] = []
    threshold = float(min_trades_per_day)
    for row in results:
        item = dict(row)
        params = dict(item.get("params") or {})
        tf = int(item.get("timeframe_min") or params.get("timeframe_min") or 0)
        # Current policy only keeps freshly evaluated 1m/5m strategies with
        # the liquid-session + relative-volume entry filter.
        if tf not in (1, 5):
            continue
        if str(params.get("entry_sessions") or "") != "london_new_york":
            continue
        robustness = dict(item.get("robustness") or {})
        if str(robustness.get("evaluation_policy_version") or "") != FOREX_EVALUATION_POLICY_VERSION:
            continue
        oos = dict(item.get("oos") or {})
        observed = float(oos.get("avg_trades_per_day") or 0.0)
        if observed < threshold:
            status = str(item.get("status") or item.get("funnel_stage") or "")
            if status in {"incubator", "deep_search", "promoted"}:
                item["status"] = "rejected"
                item["funnel_stage"] = "rejected"
                item["promoted"] = False
                reasons = list(item.get("rejection_reasons") or [])
                reason = f"average trades/day below hard minimum {threshold:g}"
                if reason not in reasons:
                    reasons.append(reason)
                item["rejection_reasons"] = reasons
                item["legacy_frequency_reclassified"] = True
        out.append(item)
    return out


def phase_candidates_from_results(results: List[dict]) -> List[ForexCandidate]:
    """Rebuild next-stage work from append-only stored runs.

    Discovery stays baseline-only for CPU efficiency. Once a baseline earns
    Incubator, we fan out a compact trade-management grid. The best Incubator
    configuration is then frozen into Deep Search.
    """
    queued: List[ForexCandidate] = []
    seen_local: set[str] = set()

    def add(strategy: str, params: dict, phase: str) -> None:
        candidate = ForexCandidate(strategy, _clean_params(params, phase))
        key = candidate_signature(candidate)
        if key in seen_local:
            return
        seen_local.add(key)
        queued.append(candidate)

    for row in results:
        params = dict(row.get("params") or {})
        strategy = str(row.get("strategy") or "")
        status = str(row.get("status") or row.get("funnel_stage") or "")
        if not strategy or status in {"rejected", "promoted"}:
            continue
        base_params = {k: v for k, v in params.items() if k != "_phase"}

        if status == "deep_search":
            # Freeze the exact Incubator winner for the final holdout.
            add(strategy, base_params, "deep_search")
            continue

        if status != "incubator":
            continue

        # Baseline always remains a contender.
        baseline = dict(base_params)
        baseline["exit_mode"] = "baseline"
        baseline.pop("management_trigger_r", None)
        baseline.pop("management_lock_net_r", None)
        add(strategy, baseline, "incubator")

        target_r = float(base_params.get("target_r") or 0.0)
        management_grid = (
            (1.0, 0.05, "safe_be_1r"),
            (1.5, 0.05, "safe_be_1_5r"),
            (2.0, 0.05, "safe_be_2r"),
            (1.5, 0.25, "protect_1_5r_025r"),
            (2.0, 0.25, "protect_2r_025r"),
            (2.0, 0.50, "lock_2r_05r"),
        )
        for trigger_r, lock_net_r, label in management_grid:
            # A management trigger at/above the final target is unreachable or
            # ambiguous, so only test it when there is room beyond the trigger.
            if target_r <= trigger_r:
                continue
            managed = dict(base_params)
            managed["exit_mode"] = label
            managed["management_trigger_r"] = trigger_r
            managed["management_lock_net_r"] = lock_net_r
            managed["breakeven_buffer_r"] = 0.05
            add(strategy, managed, "incubator")

    queued.sort(key=lambda c: 0 if c.params.get("_phase") == "deep_search" else 1)
    return queued


class ForexStrategyLab:
    """CPU-conscious forex strategy research.

    Research can run fully independently from cTrader using read-only external
    data. cTrader remains available as a separate Fusion Markets validation
    source when FOREX_DATA_PROVIDER=ctrader is explicitly selected.
    """

    def __init__(
        self,
        settings: Settings,
        client: CTraderClient,
        external_data: Optional[ExternalForexData] = None,
    ):
        self.settings = settings
        self.client = client
        self.external_data = external_data or ExternalForexData(settings)
        self.store = ForexStrategyStore(settings.database_url)
        self.state = ForexLabState(pairs_total=len(settings.forex_pairs))
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._bars_1m: Dict[str, List[dict]] = {}

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["broker"] = self.client.public_state()
        payload["market_data"] = self.external_data.public_state()
        payload["data_provider"] = self.settings.forex_data_provider
        payload["pairs"] = self.settings.forex_pairs
        payload["start_capital_eur"] = self.settings.forex_start_capital
        payload["risk_eur"] = self.settings.forex_risk_eur
        payload["cost_bps_per_side"] = self.settings.forex_cost_bps
        payload["min_trades_per_day"] = self.settings.strategy_min_trades_per_day
        payload["preferred_trades_per_day"] = self.settings.strategy_preferred_trades_per_day
        payload["target_trades_per_day"] = self.settings.strategy_target_trades_per_day
        payload["portfolio_min_trades_per_day"] = self.settings.portfolio_min_trades_per_day
        payload["allowed_timeframes_min"] = [1, 5]
        payload["entry_sessions"] = ["London", "New York"]
        payload["min_volume_ratio"] = self.settings.strategy_min_volume_ratio
        payload["trade_management"] = {
            "search_stage": "incubator",
            "trigger_grid_r": [1.0, 1.5, 2.0],
            "net_lock_grid_r": [0.05, 0.25, 0.50],
            "cost_adjusted": True,
            "next_bar_activation": True,
        }
        payload["results_loaded"] = len(self._results)
        return payload

    def results(self) -> List[dict]:
        return list(self._results)

    async def start(self) -> None:
        if self.state.running:
            return
        if self.settings.forex_data_provider == "ctrader" and not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = (
                "FOREX_DATA_PROVIDER=ctrader requires cTrader credentials/access token."
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

        provider = self.settings.forex_data_provider
        if provider == "ctrader":
            self.state.message = "Loading cTrader 1-minute FX history"
            await self.client.connect_and_authenticate()
        else:
            self.state.message = "Loading external read-only 1-minute FX history"

        bars_by_pair: Dict[str, List[dict]] = {}
        for pair in self.settings.forex_pairs:
            self.state.current_pair = pair
            self.state.message = f"Loading {pair} from {provider}"
            if provider == "ctrader":
                bars = await self.client.historical_bars(
                    pair,
                    timeframe_min=1,
                    max_bars=self.settings.forex_max_bars_per_pair,
                    lookback_days=self.settings.forex_lookback_days,
                )
            else:
                bars = await self.external_data.historical_bars(
                    pair,
                    max_bars=self.settings.forex_max_bars_per_pair,
                    lookback_days=self.settings.forex_lookback_days,
                )
            if len(bars) >= 300:
                bars_by_pair[pair] = bars
            else:
                log.warning("Forex Lab skipped %s: only %s bars", pair, len(bars))
            self.state.pairs_loaded += 1
            await asyncio.sleep(0)
        self.state.current_pair = ""
        if not bars_by_pair:
            raise RuntimeError(f"No usable FX bars were returned by {provider}")
        self._bars_1m = bars_by_pair
        return bars_by_pair

    async def _run(self) -> None:
        try:
            await self.store.init()
            saved = await self.store.load_state()
            self.state.generation = int(saved.get("generation", 0))
            self.state.tested_total = int(saved.get("tested_total", 0))
            self.state.promoted_total = int(saved.get("promoted_total", 0))
            self._results = apply_frequency_gate(
                await self.store.load_results(limit=1000),
                self.settings.strategy_min_trades_per_day,
            )
            self.state.promoted_total = sum(
                1 for row in self._results if bool(row.get("promoted"))
            )
            for row in [x for x in self._results if str(x.get("status") or x.get("funnel_stage") or "") == "promoted"][:5]:
                oos = row.get("oos") or {}
                stress = row.get("stress_oos") or {}
                log.info(
                    "PROMOTED forex strategy=%s tf=%s params=%s score=%s "
                    "oos_trades=%s oos_pf=%s oos_exp_r=%s oos_win=%s "
                    "stress_pf=%s stress_exp_r=%s positive_pairs=%s/%s",
                    row.get("strategy"),
                    row.get("timeframe_min"),
                    row.get("params"),
                    row.get("funnel_score"),
                    oos.get("trades"),
                    oos.get("profit_factor"),
                    oos.get("expectancy_r"),
                    oos.get("win_rate_pct"),
                    stress.get("profit_factor"),
                    stress.get("expectancy_r"),
                    row.get("positive_pairs"),
                    row.get("pair_count"),
                )

            source: Dict[str, List[dict]] = {}
            while self.state.running:
                if not source:
                    try:
                        source = await self._load_source_data()
                        self.state.last_error = None
                    except Exception as exc:
                        self.state.stage = "waiting_data"
                        self.state.last_error = str(exc)
                        self.state.message = f"Data tijdelijk niet beschikbaar: {exc}. Nieuwe poging over 5 min."
                        log.warning("Forex data unavailable; retrying later: %s", exc)
                        await asyncio.sleep(300)
                        self._bars_1m = {}
                        continue

                seen = await self.store.load_signatures()
                self._results = apply_frequency_gate(
                    await self.store.load_results(limit=1000),
                    self.settings.strategy_min_trades_per_day,
                )

                advanced = phase_candidates_from_results(self._results)
                discovery: List[ForexCandidate] = []
                for candidate in candidate_grid():
                    base = {
                        **candidate.params,
                        "entry_sessions": "london_new_york",
                        "min_volume_ratio": self.settings.strategy_min_volume_ratio,
                        "volume_window": self.settings.strategy_volume_window,
                    }
                    for risk_eur in (2.0, 3.0):
                        discovery.append(
                            ForexCandidate(
                                candidate.strategy,
                                {**base, "risk_eur": risk_eur, "_phase": "discovery"},
                            )
                        )

                # Resume Deep Search / Incubator first after a restart.
                ordered = advanced + discovery
                remaining = []
                candidate_seen: set[str] = set()
                for candidate in ordered:
                    sig = evaluation_signature(
                        candidate,
                        source,
                        evaluation_policy_key(self.settings),
                    )
                    local_key = candidate_signature(candidate)
                    if sig in seen or local_key in candidate_seen:
                        continue
                    candidate_seen.add(local_key)
                    remaining.append(candidate)

                if not remaining:
                    self.state.stage = "waiting_new_data"
                    self.state.progress = 0
                    self.state.total = 0
                    self.state.message = (
                        "Alle kandidaten op deze dataset zijn getest. "
                        "Forex Lab blijft actief en controleert over 15 min op nieuwe marktdata."
                    )
                    await asyncio.sleep(900)
                    self._bars_1m = {}
                    source = {}
                    continue

                batch_size = max(1, self.settings.forex_lab_batch_size)
                batch = remaining[:batch_size]
                self.state.generation += 1
                self.state.stage = "testing"
                self.state.total = len(batch)
                self.state.progress = 0

                for idx, candidate in enumerate(batch, start=1):
                    if not self.state.running:
                        break
                    tf = int(candidate.params.get("timeframe_min", 1))
                    phase = str(candidate.params.get("_phase", "discovery"))
                    bars: Dict[str, List[dict]] = {}
                    for pair, pair_bars in source.items():
                        if phase == "deep_search":
                            selected = pair_bars
                        else:
                            holdout_start = max(2, int(len(pair_bars) * 0.80))
                            pre_holdout = pair_bars[:holdout_start]
                            if phase == "discovery":
                                selected = pre_holdout[-min(10000, len(pre_holdout)):]
                            else:
                                selected = pre_holdout
                        bars[pair] = aggregate_bars(selected, tf)
                    self.state.current_candidate = candidate.strategy
                    self.state.current_params = dict(candidate.params)
                    self.state.message = (
                        f"{str(candidate.params.get('_phase', 'discovery')).replace('_', ' ').title()} "
                        f"{idx}/{len(batch)} · {candidate.strategy} · "
                        f"{tf}m · target {candidate.params.get('target_r')}R · "
                        f"risk €{candidate.params.get('risk_eur')} · "
                        f"London/NY · vol ≥{self.settings.strategy_min_volume_ratio:.2f}x · "
                        f"hard min {self.settings.strategy_min_trades_per_day:g}/day · "
                        f"frequency preference {self.settings.strategy_preferred_trades_per_day:g}-"
                        f"{self.settings.strategy_target_trades_per_day:g}/day"
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
                        min_trades_per_day=self.settings.strategy_min_trades_per_day,
                        preferred_trades_per_day=self.settings.strategy_preferred_trades_per_day,
                        target_trades_per_day=self.settings.strategy_target_trades_per_day,
                        start_capital=self.settings.forex_start_capital,
                    )
                    signature = evaluation_signature(
                        candidate,
                        source,
                        evaluation_policy_key(self.settings),
                    )
                    result["dataset"]["version"] = dataset_version(source)
                    result["dataset"]["source"] = self.settings.forex_data_provider
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
                    await asyncio.sleep(0.05)

                if self.state.running:
                    self.state.stage = "continuing"
                    self.state.message = (
                        f"Batch afgerond. {self.state.tested_total} totaal getest; "
                        "automatisch door met de volgende batch."
                    )
                    await asyncio.sleep(1)
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


def evaluation_policy_key(settings: Settings) -> str:
    payload = {
        "version": FOREX_EVALUATION_POLICY_VERSION,
        "cost_bps": settings.forex_cost_bps,
        "stress_multiplier": settings.forex_stress_cost_multiplier,
        "min_oos_trades": settings.forex_min_oos_trades,
        "min_profit_factor": settings.forex_min_profit_factor,
        "min_payoff_ratio": settings.forex_min_payoff_ratio,
        "min_trades_per_day": settings.strategy_min_trades_per_day,
        "preferred_trades_per_day": settings.strategy_preferred_trades_per_day,
        "target_trades_per_day": settings.strategy_target_trades_per_day,
        "min_volume_ratio": settings.strategy_min_volume_ratio,
        "volume_window": settings.strategy_volume_window,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluation_signature(
    candidate: ForexCandidate,
    bars_by_pair: Dict[str, List[dict]],
    policy_key: str = "forex-funnel-v3-default",
) -> str:
    raw = (
        candidate_signature(candidate)
        + ":"
        + dataset_version(bars_by_pair)
        + ":policy="
        + str(policy_key)
    )
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
