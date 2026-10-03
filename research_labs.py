from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import Settings
from ctrader_client import CTraderClient
from forex_research_store import ForexResearchStore
from research_store import RESEARCH_VALIDATION_VERSION, ResearchStore
from strategy_lab import Candidate, aggregate_bars, candidate_signature, metrics, simulate

log = logging.getLogger("microtrader.research_labs")


@dataclass
class ResearchState:
    running: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_lab: str = ""
    completed_labs: int = 0
    total_labs: int = 0
    last_error: Optional[str] = None
    requested_strategy: str = ""
    requested_signature: str = ""


# One Validation phase, three robustness checks, one final decision.
LABS = [
    "walk_forward",
    "parameter_stability",
    "cost_stress",
    "master",
]


class ResearchLabs:
    """Compact robustness validation for a frozen Research candidate.

    This is deliberately not another strategy-search layer. Discovery and
    distillation happen in StrategyLab/ResearchAgent. Validation only asks
    whether the exact frozen rule survives different periods, nearby
    parameters and higher transaction costs.
    """

    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.store = ResearchStore(settings.database_url)
        self.state = ResearchState(total_labs=len(LABS))
        self._task: Optional[asyncio.Task] = None
        self._results: Dict[str, dict] = {}
        self._requested_candidate: Optional[Candidate] = None

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["market"] = "forex"
        payload["data_source"] = "cTrader / Fusion demo"
        payload["pairs"] = list(self.settings.forex_pairs)
        payload["labs"] = {key: value for key, value in self._results.items()}
        payload["validation_version"] = RESEARCH_VALIDATION_VERSION
        return payload

    async def start(self, candidate: Optional[Candidate] = None) -> None:
        if self.state.running:
            return
        self._requested_candidate = candidate
        requested_signature = candidate_signature(candidate) if candidate else ""
        self.state = ResearchState(
            running=True,
            started_at=datetime.now(timezone.utc).isoformat(),
            total_labs=len(LABS),
            requested_strategy=candidate.strategy if candidate else "",
            requested_signature=requested_signature,
        )
        self._results = {}
        log.info(
            "Validation start: strategy=%s signature=%s",
            candidate.strategy if candidate else "auto",
            requested_signature[:12] if requested_signature else "-",
        )
        self._task = asyncio.create_task(
            self._run(), name="microtrader-research-validation"
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self._task = None

    async def _fallback_candidate(self) -> Candidate:
        store = ForexResearchStore(self.settings.database_url)
        await store.init()
        rows = await store.load_research_memory(per_family_stage=20, limit=200)
        eligible = []
        for row in rows:
            try:
                if row.get("funnel_stage") != "promoted" or not row.get("promoted"):
                    continue
                params = dict(row.get("params") or {})
                tf = int(params.get("timeframe_min") or 0)
                avg_tpd = float((row.get("oos") or {}).get("avg_trades_per_day") or 0.0)
                if tf not in (1, 5):
                    continue
                if str(params.get("market") or "") != "forex":
                    continue
                if str(params.get("data_source") or "") != "ctrader":
                    continue
                if str(params.get("direction_mode") or "") != "long_short":
                    continue
                if avg_tpd < float(self.settings.strategy_min_trades_per_day):
                    continue
                eligible.append(row)
            except Exception:
                continue
        if not eligible:
            raise RuntimeError("No promoted Research candidate available for validation")
        eligible.sort(
            key=lambda row: (
                float(row.get("funnel_score") or 0.0),
                float((row.get("oos") or {}).get("profit_factor") or 0.0),
                float((row.get("oos") or {}).get("expectancy_bps") or 0.0),
            ),
            reverse=True,
        )
        best = eligible[0]
        return Candidate(str(best.get("strategy") or ""), dict(best.get("params") or {}))

    async def _run(self) -> None:
        try:
            await self.store.init()
            bars_by_symbol: Dict[str, List[dict]] = {}
            for pair in self.settings.forex_pairs:
                bars = await self.client.historical_bars(
                    pair,
                    timeframe_min=1,
                    max_bars=self.settings.forex_max_bars_per_pair,
                    lookback_days=self.settings.forex_lookback_days,
                )
                if len(bars) >= 300:
                    bars_by_symbol[pair] = bars
            if not bars_by_symbol:
                raise RuntimeError("No usable cTrader forex data available for validation")

            candidate = self._requested_candidate or await self._fallback_candidate()
            candidates = [candidate]
            log.info(
                "Validating frozen candidate: strategy=%s signature=%s",
                candidate.strategy,
                candidate_signature(candidate)[:12],
            )

            for name in LABS[:-1]:
                self.state.current_lab = name
                fn = getattr(self, f"_lab_{name}")
                result = await asyncio.to_thread(fn, candidates, bars_by_symbol)
                self._results[name] = result
                await self.store.save(name, "completed", result)
                self.state.completed_labs += 1

            self.state.current_lab = "master"
            master = self._lab_master(candidates, bars_by_symbol)
            self._results["master"] = master
            await self.store.save("master", "completed", master)
            self.state.completed_labs += 1
            log.info(
                "Validation complete: passed=%s checks=%s/%s",
                bool(master.get("passed")),
                self.state.completed_labs,
                self.state.total_labs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            log.exception("Research validation failed")
        finally:
            self.state.running = False
            self.state.completed_at = datetime.now(timezone.utc).isoformat()

    def _candidate_bars(self, candidate: Candidate, bars: List[dict]) -> List[dict]:
        tf = int(candidate.params.get("timeframe_min") or 1)
        return aggregate_bars(bars, tf)

    def _candidate_trades(
        self,
        candidate: Candidate,
        bars_by_symbol: Dict[str, List[dict]],
        cost: Optional[float] = None,
    ) -> List[dict]:
        cost_bps = self.settings.forex_cost_bps if cost is None else float(cost)
        trades: List[dict] = []
        for symbol, bars in bars_by_symbol.items():
            work = self._candidate_bars(candidate, bars)
            split = max(2, int(len(work) * 0.70))
            trades.extend(simulate(candidate, symbol, work[split:], cost_bps))
        return sorted(trades, key=lambda row: str(row.get("exit_time") or ""))

    def _score_candidates(
        self,
        candidates: List[Candidate],
        bars_by_symbol: Dict[str, List[dict]],
    ) -> List[dict]:
        rows = []
        for candidate in candidates:
            trades = self._candidate_trades(candidate, bars_by_symbol)
            rows.append({
                "strategy": candidate.strategy,
                "params": candidate.params,
                "metrics": metrics(trades),
                "trades": trades,
            })
        rows.sort(
            key=lambda row: (
                float((row.get("metrics") or {}).get("profit_factor") or 0.0),
                float((row.get("metrics") or {}).get("expectancy_bps") or 0.0),
            ),
            reverse=True,
        )
        return rows

    def _lab_walk_forward(self, candidates, bars_by_symbol) -> dict:
        output = []
        for candidate in candidates:
            windows = []
            for window_index in range(4):
                trades = []
                for symbol, bars in bars_by_symbol.items():
                    work = self._candidate_bars(candidate, bars)
                    n = len(work)
                    start = int(n * window_index / 4)
                    end = int(n * (window_index + 1) / 4)
                    chunk = work[start:end]
                    if len(chunk) < 100:
                        continue
                    trades.extend(
                        simulate(
                            candidate,
                            symbol,
                            chunk,
                            self.settings.forex_cost_bps,
                        )
                    )
                windows.append(metrics(trades))
            positive = sum(
                1 for row in windows
                if float(row.get("expectancy_bps") or 0.0) > 0
                and float(row.get("profit_factor") or 0.0) >= 1.0
            )
            output.append({
                "strategy": candidate.strategy,
                "params": candidate.params,
                "positive_windows": positive,
                "windows": windows,
            })
        return {"candidates": output}

    def _neighbors(self, candidate: Candidate) -> List[Candidate]:
        params = dict(candidate.params)
        candidates: List[Candidate] = []
        tunable = [
            key for key, value in params.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {
                "max_hold",
                "timeframe_min",
                "min_volume_ratio",
                "volume_window",
                "risk_eur",
                "start_capital_eur",
            }
        ][:4]
        for key in tunable:
            value = params[key]
            for multiplier in (0.9, 1.1):
                neighbor = dict(params)
                moved = float(value) * multiplier
                neighbor[key] = (
                    max(1, int(round(moved)))
                    if isinstance(value, int)
                    else round(max(0.01, moved), 4)
                )
                if neighbor[key] != value:
                    candidates.append(Candidate(candidate.strategy, neighbor))
        return candidates[:8]

    def _lab_parameter_stability(self, candidates, bars_by_symbol) -> dict:
        rows = []
        for candidate in candidates:
            neighbor_metrics = []
            for neighbor in self._neighbors(candidate):
                try:
                    neighbor_metrics.append(
                        metrics(self._candidate_trades(neighbor, bars_by_symbol))
                    )
                except Exception:
                    continue
            positive = sum(
                1 for row in neighbor_metrics
                if float(row.get("expectancy_bps") or 0.0) > 0
                and float(row.get("profit_factor") or 0.0) >= 1.0
            )
            rows.append({
                "strategy": candidate.strategy,
                "params": candidate.params,
                "neighbors_tested": len(neighbor_metrics),
                "positive_neighbors": positive,
                "stability_ratio": round(
                    positive / max(1, len(neighbor_metrics)), 3
                ),
            })
        return {"candidates": rows}

    def _lab_cost_stress(self, candidates, bars_by_symbol) -> dict:
        best = self._score_candidates(candidates, bars_by_symbol)[0]
        candidate = Candidate(best["strategy"], best["params"])
        return {
            "strategy": candidate.strategy,
            "params": candidate.params,
            "cost_scenarios": {
                str(multiplier): metrics(
                    self._candidate_trades(
                        candidate,
                        bars_by_symbol,
                        self.settings.forex_cost_bps * multiplier,
                    )
                )
                for multiplier in (1.0, 1.5, 2.0)
            },
        }

    def _lab_master(self, candidates, bars_by_symbol) -> dict:
        ranked = self._score_candidates(candidates, bars_by_symbol)
        if not ranked:
            return {
                "validation_version": RESEARCH_VALIDATION_VERSION,
                "passed": False,
                "criteria": {},
                "reason": "No validation trades available.",
            }

        best = ranked[0]
        base = best["metrics"]
        wf = ((self._results.get("walk_forward") or {}).get("candidates") or [{}])[0]
        stability = (
            (self._results.get("parameter_stability") or {}).get("candidates")
            or [{}]
        )[0]
        costs = (self._results.get("cost_stress") or {}).get("cost_scenarios") or {}
        cost_15 = costs.get("1.5") or {}

        criteria = {
            "base_quality": (
                float(base.get("expectancy_bps") or 0.0) > 0
                and float(base.get("profit_factor") or 0.0) >= 1.15
            ),
            "walk_forward": int(wf.get("positive_windows") or 0) >= 3,
            "parameter_stability": (
                int(stability.get("neighbors_tested") or 0) >= 4
                and float(stability.get("stability_ratio") or 0.0) >= 0.50
            ),
            "cost_stress_1_5x": (
                float(cost_15.get("expectancy_bps") or 0.0) > 0
                and float(cost_15.get("profit_factor") or 0.0) >= 1.0
            ),
        }
        passed = all(criteria.values())
        return {
            "validation_version": RESEARCH_VALIDATION_VERSION,
            "passed": passed,
            "market": "forex",
            "data_source": "cTrader / Fusion demo",
            "candidate": {
                "strategy": best["strategy"],
                "params": best["params"],
            },
            "metrics": base,
            "criteria": criteria,
            "checks": {
                "walk_forward_positive_windows": int(
                    wf.get("positive_windows") or 0
                ),
                "parameter_stability_ratio": float(
                    stability.get("stability_ratio") or 0.0
                ),
                "cost_stress_1_5x": cost_15,
            },
            "note": (
                "Single robust validation gate; paper trading is the forward "
                "unseen test."
            ),
        }
