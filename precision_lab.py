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
from precision_backtest import (
    PrecisionCandidate,
    QuoteTick,
    candidate_grid,
    candidate_signature,
    evaluate_candidate,
)
from precision_store import PrecisionStrategyStore

log = logging.getLogger("microtrader.precision_lab")


@dataclass
class PrecisionLabState:
    running: bool = False
    stage: str = "idle"
    message: str = "Waiting for cTrader credentials"
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
    """Separate tight-stop tester.

    Normal Forex Lab = ATR stops / broader targets.
    Precision Lab = 3-4 pip stops / 5-10 pip targets, pending entries and
    historical bid/ask tick execution.
    """

    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.store = PrecisionStrategyStore(settings.database_url)
        self.state = PrecisionLabState(pairs_total=len(settings.precision_pairs))
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._bars: Dict[str, List[dict]] = {}
        self._ticks: Dict[str, List[QuoteTick]] = {}

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["broker"] = self.client.public_state()
        payload["pairs"] = self.settings.precision_pairs
        payload["start_capital_eur"] = self.settings.precision_start_capital
        payload["risk_eur"] = self.settings.precision_risk_eur
        payload["stop_pips"] = [3, 4]
        payload["target_pips"] = [5, 6, 7, 8, 9, 10]
        payload["commission_pips_roundtrip"] = self.settings.precision_commission_pips
        payload["tick_execution"] = True
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
                "Precision Lab is built but cTrader credentials/access token "
                "are not configured yet."
            )
            return

        self.state = PrecisionLabState(
            running=True,
            stage="starting",
            message="Preparing Precision Lab",
            started_at=datetime.now(timezone.utc).isoformat(),
            pairs_total=len(self.settings.precision_pairs),
        )
        self._task = asyncio.create_task(
            self._run(), name="microtrader-precision-lab"
        )

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
        await self.client.connect_and_authenticate()
        bars_by_pair: Dict[str, List[dict]] = {}
        ticks_by_pair: Dict[str, List[QuoteTick]] = {}

        for pair in self.settings.precision_pairs:
            self.state.current_pair = pair
            self.state.message = f"Loading 1m bars for {pair}"
            bars = await self.client.historical_bars(
                pair,
                timeframe_min=1,
                max_bars=self.settings.precision_max_bars_per_pair,
                lookback_days=self.settings.precision_lookback_days,
            )
            if len(bars) < 300:
                log.warning("Precision Lab skipped %s: only %s bars", pair, len(bars))
                continue

            self.state.message = f"Loading bid/ask ticks for {pair}"
            ticks = await self.client.historical_quote_ticks(
                pair,
                lookback_days=self.settings.precision_lookback_days,
                max_ticks_per_side=self.settings.precision_max_ticks_per_side,
            )
            if len(ticks) < 1000:
                log.warning("Precision Lab skipped %s: only %s quote ticks", pair, len(ticks))
                continue

            tick_start = ticks[0][0]
            tick_end = ticks[-1][0]
            aligned_bars = []
            for bar in bars:
                try:
                    dt = datetime.fromisoformat(str(bar.get("t") or "").replace("Z", "+00:00"))
                    ms = int(dt.timestamp() * 1000)
                except Exception:
                    continue
                if tick_start <= ms <= tick_end:
                    aligned_bars.append(bar)
            if len(aligned_bars) < 300:
                log.warning(
                    "Precision Lab skipped %s: only %s bars overlap tick coverage",
                    pair, len(aligned_bars),
                )
                continue

            bars_by_pair[pair] = aligned_bars
            ticks_by_pair[pair] = ticks
            self.state.pairs_loaded += 1
            self.state.bars_loaded += len(bars)
            self.state.quote_ticks_loaded += len(ticks)
            await asyncio.sleep(0)

        self.state.current_pair = ""
        if not bars_by_pair:
            raise RuntimeError("No usable cTrader precision data was returned")

        self._bars = bars_by_pair
        self._ticks = ticks_by_pair
        return bars_by_pair, ticks_by_pair

    async def _run(self) -> None:
        try:
            await self.store.init()
            saved_state = await self.store.load_state()
            self.state.generation = int(saved_state.get("generation", 0))
            self.state.tested_total = int(saved_state.get("tested_total", 0))
            self.state.deep_search_total = int(
                saved_state.get("deep_search_total", 0)
            )
            self._results = await self.store.load_results(limit=250)
            seen = await self.store.load_signatures()

            bars, ticks = await self._load_data()
            data_version = dataset_version(bars, ticks)

            configured: List[PrecisionCandidate] = []
            for candidate in candidate_grid():
                configured.append(PrecisionCandidate(
                    candidate.strategy,
                    {
                        **candidate.params,
                        "risk_eur": self.settings.precision_risk_eur,
                    },
                ))

            remaining = [
                c for c in configured
                if evaluation_signature(c, data_version) not in seen
            ]
            batch_size = max(1, self.settings.precision_batch_size)
            batch = remaining[:batch_size]
            if not batch:
                self.state.stage = "search_exhausted"
                self.state.message = (
                    "All configured precision candidates have been tested "
                    "on this dataset version."
                )
                return

            self.state.generation += 1
            self.state.stage = "testing"
            self.state.total = len(batch)
            self.state.progress = 0

            for idx, candidate in enumerate(batch, start=1):
                if not self.state.running:
                    break
                self.state.current_candidate = candidate.strategy
                self.state.current_params = dict(candidate.params)
                self.state.message = (
                    f"Precision {idx}/{len(batch)} · {candidate.strategy} · "
                    f"{candidate.params['stop_pips']}p stop / "
                    f"{candidate.params['target_pips']}p target"
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
                    start_capital=self.settings.precision_start_capital,
                )
                result["dataset"]["version"] = data_version
                signature = evaluation_signature(candidate, data_version)
                run_id = await self.store.save_run(signature, result)

                # Raw trades are persisted in their own table, not duplicated
                # inside the run-summary JSON or dashboard state.
                clean = {
                    k: v for k, v in result.items()
                    if not k.startswith("_")
                }
                if run_id:
                    clean["run_id"] = run_id

                self.state.tested_total += 1
                if clean.get("funnel_stage") == "precision_deep_search":
                    self.state.deep_search_total += 1
                self.state.progress = idx
                self.state.last_completed_candidate = candidate.strategy
                self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                self._results.append(clean)
                self._results.sort(
                    key=lambda row: (
                        str(row.get("funnel_stage")) == "precision_deep_search",
                        float(row.get("funnel_score") or 0),
                        float((row.get("oos") or {}).get("expectancy_r") or 0),
                    ),
                    reverse=True,
                )
                self._results = self._results[:250]
                await self.store.save_state(
                    self.state.generation,
                    self.state.tested_total,
                    self.state.deep_search_total,
                )

                m = clean.get("oos") or {}
                log.info(
                    "Precision candidate complete: strategy=%s stop=%sp target=%sp "
                    "stage=%s score=%s trades=%s per_day=%s win=%s pf=%s "
                    "payoff=%s exp_r=%s avg_win_r=%s avg_loss_r=%s "
                    "best_r=%s worst_r=%s drawdown=%s fill=%s",
                    candidate.strategy,
                    candidate.params.get("stop_pips"),
                    candidate.params.get("target_pips"),
                    clean.get("funnel_stage"),
                    clean.get("funnel_score"),
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
                    m.get("max_drawdown_pct"),
                    clean.get("avg_fill_rate_pct"),
                )
                await asyncio.sleep(0.05)

            self.state.stage = "batch_complete"
            self.state.message = (
                f"Precision batch complete: {self.state.progress}/"
                f"{self.state.total} tested. Individual trades are stored."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Precision Lab failed")
        finally:
            self.state.running = False
            self.state.current_candidate = ""
            self.state.current_params = None
            self.state.current_pair = ""
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


def dataset_version(
    bars: Dict[str, List[dict]], ticks: Dict[str, List[QuoteTick]]
) -> str:
    payload = {}
    for pair in sorted(bars):
        pb = bars[pair]
        pt = ticks.get(pair) or []
        payload[pair] = {
            "bars": len(pb),
            "bar_first": str(pb[0].get("t") or "") if pb else "",
            "bar_last": str(pb[-1].get("t") or "") if pb else "",
            "ticks": len(pt),
            "tick_first": pt[0][0] if pt else 0,
            "tick_last": pt[-1][0] if pt else 0,
        }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluation_signature(candidate: PrecisionCandidate, data_version: str) -> str:
    raw = candidate_signature(candidate) + ":" + data_version
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
