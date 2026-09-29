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
    """Separate tight-stop research tester using bid/ask tick execution."""

    def __init__(
        self,
        settings: Settings,
        client: CTraderClient,
        external_data: Optional[ExternalForexData] = None,
    ):
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
        payload["broker"] = self.client.public_state()
        payload["market_data"] = self.external_data.public_state()
        payload["data_provider"] = self.settings.precision_data_provider
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
        if self.settings.precision_data_provider == "ctrader" and not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = "PRECISION_DATA_PROVIDER=ctrader requires cTrader access."
            return
        self.state = PrecisionLabState(
            running=True,
            stage="starting",
            message="Preparing Precision Lab",
            started_at=datetime.now(timezone.utc).isoformat(),
            pairs_total=len(self.settings.precision_pairs),
        )
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
        bars_by_pair: Dict[str, List[dict]] = {}
        ticks_by_pair: Dict[str, List[QuoteTick]] = {}
        use_ctrader = self.settings.precision_data_provider == "ctrader"
        if use_ctrader:
            await self.client.connect_and_authenticate()

        for pair in self.settings.precision_pairs:
            self.state.current_pair = pair
            self.state.message = f"Loading 1m bars for {pair}"
            if use_ctrader:
                bars = await self.client.historical_bars(pair, timeframe_min=1, max_bars=self.settings.precision_max_bars_per_pair, lookback_days=self.settings.precision_lookback_days)
            else:
                bars = await self.external_data.historical_bars(pair, max_bars=self.settings.precision_max_bars_per_pair, lookback_days=self.settings.precision_lookback_days)
            if len(bars) < 300:
                log.warning("Precision Lab skipped %s: only %s bars", pair, len(bars))
                continue

            self.state.message = f"Loading bid/ask ticks for {pair}"
            if use_ctrader:
                ticks = await self.client.historical_quote_ticks(pair, lookback_days=self.settings.precision_lookback_days, max_ticks_per_side=self.settings.precision_max_ticks_per_side)
            else:
                ticks = await self.external_data.historical_quote_ticks(pair, lookback_days=self.settings.precision_lookback_days, max_ticks=self.settings.precision_max_ticks_per_side * 2)
            if len(ticks) < 1000:
                log.warning("Precision Lab skipped %s: only %s quote ticks", pair, len(ticks))
                continue

            tick_start, tick_end = ticks[0][0], ticks[-1][0]
            aligned_bars = []
            for bar in bars:
                raw = str(bar.get("t") or "")
                try:
                    ts = int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)
                except ValueError:
                    continue
                if tick_start <= ts <= tick_end:
                    aligned_bars.append(bar)
            if len(aligned_bars) < 300:
                continue
            bars_by_pair[pair] = aligned_bars
            ticks_by_pair[pair] = ticks
            self.state.pairs_loaded = len(bars_by_pair)
            self.state.bars_loaded += len(aligned_bars)
            self.state.quote_ticks_loaded += len(ticks)

        self._bars, self._ticks = bars_by_pair, ticks_by_pair
        return bars_by_pair, ticks_by_pair

    async def _run(self) -> None:
        try:
            bars_by_pair, ticks_by_pair = await self._load_data()
            if not bars_by_pair:
                raise RuntimeError("No precision research data available")
            candidates = candidate_grid(self.settings)
            self.state.total = len(candidates)
            self.state.stage = "testing"
            for idx, candidate in enumerate(candidates, 1):
                if not self.state.running:
                    break
                self.state.progress = idx
                self.state.current_candidate = candidate_signature(candidate)
                self.state.current_params = asdict(candidate)
                result = evaluate_candidate(candidate, bars_by_pair, ticks_by_pair, self.settings)
                self.state.tested_total += 1
                payload = result if isinstance(result, dict) else asdict(result)
                self._results.append(payload)
                try:
                    self.store.save(payload)
                except Exception as exc:
                    log.warning("Unable to persist precision result: %s", exc)
                self.state.last_completed_candidate = self.state.current_candidate
                self.state.last_completed_at = datetime.now(timezone.utc).isoformat()
                await asyncio.sleep(0)
            self.state.stage = "complete"
            self.state.message = "Precision research run complete"
            self.state.completed_at = datetime.now(timezone.utc).isoformat()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Precision Lab failed")
            self.state.stage = "error"
            self.state.last_error = str(exc)
            self.state.message = str(exc)
        finally:
            self.state.running = False
