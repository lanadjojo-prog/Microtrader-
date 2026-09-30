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
from forex_data import ExternalForexData
from precision_backtest import PrecisionCandidate, QuoteTick, candidate_grid, candidate_signature, evaluate_candidate
from precision_store import PrecisionStrategyStore

log = logging.getLogger("microtrader.precision_lab")


def apply_frequency_gate(results: List[dict], min_trades_per_day: float) -> List[dict]:
    out: List[dict] = []
    threshold = float(min_trades_per_day)
    for row in results:
        item = dict(row)
        params = dict(item.get("params") or {})
        if str(params.get("entry_sessions") or "") != "london_new_york":
            continue
        oos = dict(item.get("oos") or {})
        observed = float(oos.get("avg_trades_per_day") or 0.0)
        status = str(item.get("status") or item.get("funnel_stage") or "")
        if observed < threshold and status in {"precision_incubator", "precision_deep_search"}:
            item["status"] = "rejected"
            item["funnel_stage"] = "rejected"
            reasons = list(item.get("rejection_reasons") or [])
            reason = f"average trades/day below hard minimum {threshold:g}"
            if reason not in reasons:
                reasons.append(reason)
            item["rejection_reasons"] = reasons
            item["legacy_frequency_reclassified"] = True
        out.append(item)
    return out


@dataclass
class PrecisionLabState:
    running: bool = False
    stage: str = "idle"
    message: str = "Ready for research-only precision data"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_error: Optional[str] = None
    generation: int = 0
    tested_total: int = 0
    deep_search_total: int = 0
    progress: int = 0
    total: int = 0
    current_candidate: str = ""
    current_params: dict | None = None
    current_pair: str = ""
    pairs_loaded: int = 0
    pairs_total: int = 0
    bars_loaded: int = 0
    quote_ticks_loaded: int = 0
    last_completed_candidate: str = ""
    last_completed_at: Optional[str] = None


class PrecisionStrategyLab:
    """Persistent tight-stop research worker; never places orders."""

    def __init__(self, settings: Settings, client: CTraderClient, external_data: Optional[ExternalForexData] = None):
        self.settings = settings
        self.client = client
        self.external_data = external_data or ExternalForexData(settings)
        self.store = PrecisionStrategyStore(settings.database_url)
        self.state = PrecisionLabState(pairs_total=len(settings.precision_pairs))
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._bars: Dict[str, List[dict]] = {}
        self._ticks: Dict[str, List[QuoteTick]] = {}

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload.update({
            "broker": self.client.public_state(),
            "market_data": self.external_data.public_state(),
            "data_provider": self.settings.precision_data_provider,
            "pairs": self.settings.precision_pairs,
            "start_capital_eur": self.settings.precision_start_capital,
            "risk_eur": [2.0, 3.0],
            "stop_pips": [2, 3, 4, 5],
            "target_pips": [4, 5, 6, 7, 8, 9, 10],
            "commission_pips_roundtrip": self.settings.precision_commission_pips,
            "min_trades_per_day": self.settings.strategy_min_trades_per_day,
            "preferred_trades_per_day": self.settings.strategy_preferred_trades_per_day,
            "target_trades_per_day": self.settings.strategy_target_trades_per_day,
            "portfolio_min_trades_per_day": self.settings.portfolio_min_trades_per_day,
            "allowed_timeframes_min": [1],
            "entry_sessions": ["London", "New York"],
            "min_volume_ratio": self.settings.strategy_min_volume_ratio,
            "tick_execution": True,
            "results_loaded": len(self._results),
        })
        return payload

    def results(self) -> List[dict]:
        return list(self._results)

    async def start(self) -> None:
        if self.state.running:
            return
        if self.settings.precision_data_provider == "ctrader" and not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = "PRECISION_DATA_PROVIDER=ctrader requires cTrader access."
            return
        self.state = PrecisionLabState(running=True, stage="starting", message="Preparing Precision Lab", started_at=datetime.now(timezone.utc).isoformat(), pairs_total=len(self.settings.precision_pairs))
        self._task = asyncio.create_task(self._run(), name="microtrader-precision-lab")

    async def stop(self) -> None:
        self.state.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _load_data(self) -> tuple[Dict[str, List[dict]], Dict[str, List[QuoteTick]]]:
        if self._bars and self._ticks:
            return self._bars, self._ticks
        self.state.stage = "loading_data"
        self.state.pairs_loaded = 0
        self.state.bars_loaded = 0
        self.state.quote_ticks_loaded = 0
        bars_by_pair: Dict[str, List[dict]] = {}
        ticks_by_pair: Dict[str, List[QuoteTick]] = {}
        use_ctrader = self.settings.precision_data_provider == "ctrader"
        if use_ctrader:
            await self.client.connect_and_authenticate()
        for pair in self.settings.precision_pairs:
            self.state.current_pair = pair
            self.state.message = f"Precision: {pair} 1m-bars laden..."
            if use_ctrader:
                bars = await self.client.historical_bars(pair, timeframe_min=1, max_bars=self.settings.precision_max_bars_per_pair, lookback_days=self.settings.precision_lookback_days)
                self.state.bars_loaded += len(bars)
                self.state.message = f"Precision: {pair} bid/ask ticks laden ({len(bars)} bars klaar)..."
                ticks = await self.client.historical_quote_ticks(pair, lookback_days=self.settings.precision_lookback_days, max_ticks_per_side=self.settings.precision_max_ticks_per_side)
            else:
                bars = await self.external_data.historical_bars(pair, max_bars=self.settings.precision_max_bars_per_pair, lookback_days=self.settings.precision_lookback_days)
                ticks = await self.external_data.historical_quote_ticks(pair, lookback_days=self.settings.precision_lookback_days, max_ticks=self.settings.precision_max_ticks_per_side * 2)
            if len(bars) < 300 or len(ticks) < 1000:
                continue
            tick_start, tick_end = ticks[0][0], ticks[-1][0]
            aligned = []
            for bar in bars:
                try:
                    ms = int(datetime.fromisoformat(str(bar.get("t") or "").replace("Z", "+00:00")).timestamp() * 1000)
                except Exception:
                    continue
                if tick_start <= ms <= tick_end:
                    aligned.append(bar)
            if len(aligned) < 300:
                continue
            bars_by_pair[pair], ticks_by_pair[pair] = aligned, ticks
            self.state.pairs_loaded = len(bars_by_pair)
            if not use_ctrader:
                self.state.bars_loaded += len(aligned)
            self.state.quote_ticks_loaded += len(ticks)
            self.state.message = f"Precision: {pair} data klaar · {len(aligned)} bars · {len(ticks)} quotes"
        self.state.current_pair = ""
        if not bars_by_pair:
            raise RuntimeError("No precision research data available")
        self._bars, self._ticks = bars_by_pair, ticks_by_pair
        return bars_by_pair, ticks_by_pair

    async def _run(self) -> None:
        try:
            await self.store.init()
            saved = await self.store.load_state()
            self.state.generation = int(saved.get("generation", 0))
            self.state.tested_total = int(saved.get("tested_total", 0))
            self.state.deep_search_total = int(saved.get("deep_search_total", 0))
            self._results = apply_frequency_gate(
                await self.store.load_results(limit=250),
                self.settings.strategy_min_trades_per_day,
            )
            self.state.deep_search_total = sum(
                1 for row in self._results
                if str(row.get("status") or row.get("funnel_stage") or "") == "precision_deep_search"
            )

            bars: Dict[str, List[dict]] = {}
            ticks: Dict[str, List[QuoteTick]] = {}
            while self.state.running:
                if not bars or not ticks:
                    try:
                        bars, ticks = await self._load_data()
                        self.state.last_error = None
                    except Exception as exc:
                        self.state.stage = "waiting_data"
                        self.state.last_error = str(exc)
                        self.state.message = f"Tickdata tijdelijk niet beschikbaar: {exc}. Nieuwe poging over 5 min."
                        log.warning("Precision data unavailable; retrying later: %s", exc)
                        await asyncio.sleep(300)
                        self._bars, self._ticks = {}, {}
                        bars, ticks = {}, {}
                        continue

                version = dataset_version(bars, ticks)
                seen = await self.store.load_signatures()
                configured: List[PrecisionCandidate] = []
                templates = candidate_grid()
                by_strategy = {}
                high_frequency_families = {
                    "liquidity_sweep_fvg",
                    "displacement_fvg_retrace",
                }
                for c in templates:
                    if c.strategy in high_frequency_families:
                        by_strategy.setdefault(c.strategy, c)
                for base in by_strategy.values():
                    for stop in (2.0, 3.0, 4.0, 5.0):
                        for target in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
                            for risk in (2.0, 3.0):
                                configured.append(
                                    PrecisionCandidate(
                                        base.strategy,
                                        {
                                            **base.params,
                                            "entry_sessions": "london_new_york",
                                            "min_volume_ratio": self.settings.strategy_min_volume_ratio,
                                            "volume_window": self.settings.strategy_volume_window,
                                            "stop_pips": stop,
                                            "target_pips": target,
                                            "risk_eur": risk,
                                            "_phase": "precision_economics",
                                        },
                                    )
                                )

                remaining = [
                    c for c in configured
                    if evaluation_signature(
                        c, version, self.settings.strategy_min_trades_per_day
                    ) not in seen
                ]
                if not remaining:
                    self.state.stage = "waiting_new_data"
                    self.state.progress = 0
                    self.state.total = 0
                    self.state.message = (
                        "Volledige pip-grid op deze dataset getest. "
                        "Precision Lab blijft actief en controleert over 15 min op nieuwe tickdata."
                    )
                    await asyncio.sleep(900)
                    self._bars, self._ticks = {}, {}
                    bars, ticks = {}, {}
                    continue

                batch_size = max(1, self.settings.precision_batch_size)
                batch = remaining[:batch_size]
                self.state.generation += 1
                self.state.stage = "testing"
                self.state.total = len(batch)
                self.state.progress = 0
                for idx, candidate in enumerate(batch, 1):
                    if not self.state.running:
                        break
                    self.state.progress = idx
                    self.state.current_candidate = candidate.strategy
                    self.state.current_params = dict(candidate.params)
                    self.state.message = (
                        f"Precision {idx}/{len(batch)} · {candidate.strategy} · "
                        f"{candidate.params['stop_pips']}p/{candidate.params['target_pips']}p · "
                        f"EUR {candidate.params['risk_eur']} risk · "
                        f"London/NY · vol ≥{self.settings.strategy_min_volume_ratio:.2f}x · "
                        f"hard min {self.settings.strategy_min_trades_per_day:g}/day · "
                        f"frequency preference {self.settings.strategy_preferred_trades_per_day:g}-"
                        f"{self.settings.strategy_target_trades_per_day:g}/day"
                    )
                    result = await asyncio.to_thread(
                        evaluate_candidate,
                        candidate,
                        bars,
                        ticks,
                        commission_pips_roundtrip=self.settings.precision_commission_pips,
                        stress_multiplier=self.settings.precision_stress_multiplier,
                        min_oos_trades=self.settings.precision_min_oos_trades,
                        min_profit_factor=self.settings.precision_min_profit_factor,
                        min_trades_per_day=self.settings.strategy_min_trades_per_day,
                        preferred_trades_per_day=self.settings.strategy_preferred_trades_per_day,
                        target_trades_per_day=self.settings.strategy_target_trades_per_day,
                        start_capital=self.settings.precision_start_capital,
                    )
                    result["dataset"]["version"] = version
                    result["dataset"]["source"] = self.settings.precision_data_provider
                    signature = evaluation_signature(
                        candidate, version, self.settings.strategy_min_trades_per_day
                    )
                    run_id = await self.store.save_run(signature, result)
                    clean = {k: v for k, v in result.items() if not k.startswith("_")}
                    if run_id:
                        clean["run_id"] = run_id
                    self.state.tested_total += 1
                    if clean.get("funnel_stage") == "precision_deep_search":
                        self.state.deep_search_total += 1
                    self.state.last_completed_candidate = candidate.strategy
                    self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                    self._results.append(clean)
                    self._results.sort(
                        key=lambda r: (
                            float(r.get("funnel_score") or 0),
                            float((r.get("oos") or {}).get("expectancy_r") or 0),
                        ),
                        reverse=True,
                    )
                    self._results = self._results[:250]
                    await self.store.save_state(
                        self.state.generation,
                        self.state.tested_total,
                        self.state.deep_search_total,
                    )
                    await asyncio.sleep(0.05)

                if self.state.running:
                    self.state.stage = "continuing"
                    self.state.message = (
                        f"Pip-grid afgerond. {self.state.tested_total} totaal getest; "
                        "worker blijft actief voor nieuwe marktdata."
                    )
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.stage = "error"
            self.state.last_error = str(exc)
            self.state.message = str(exc)
            log.exception("Precision Lab failed")
        finally:
            self.state.running = False
            self.state.current_candidate = ""
            self.state.current_params = None
            self.state.current_pair = ""
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


def dataset_version(bars: Dict[str, List[dict]], ticks: Dict[str, List[QuoteTick]]) -> str:
    payload = {}
    for pair in sorted(bars):
        pb, pt = bars[pair], ticks.get(pair) or []
        payload[pair] = {"bars": len(pb), "bar_first": str(pb[0].get("t") or "") if pb else "", "bar_last": str(pb[-1].get("t") or "") if pb else "", "ticks": len(pt), "tick_first": pt[0][0] if pt else 0, "tick_last": pt[-1][0] if pt else 0}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def evaluation_signature(
    candidate: PrecisionCandidate,
    data_version: str,
    min_trades_per_day: float = 10.0,
) -> str:
    raw = (
        candidate_signature(candidate)
        + ":"
        + data_version
        + f":hard-min-trades-day={float(min_trades_per_day):g}:v1"
    )
    return hashlib.sha256(raw.encode()).hexdigest()
