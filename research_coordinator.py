from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from research_labs import ResearchLabs
from strategy_lab import Candidate, StrategyLab, candidate_signature

log = logging.getLogger("microtrader.research_coordinator")


@dataclass
class CoordinatorState:
    running: bool = False
    mode: str = "idle"
    message: str = ""
    queued_signature: str = ""
    queued_strategy: str = ""
    queued_stage: str = ""
    validations_started: int = 0
    validations_completed: int = 0
    last_validation_signature: str = ""
    last_validation_at: Optional[str] = None
    last_error: Optional[str] = None


class ResearchCoordinator:
    """Serializes CPU-heavy research on small Render instances.

    Strategy discovery owns the CPU by default. Full Research Labs only run
    when the forex funnel produces a genuinely promising candidate, at which
    point discovery pauses between candidates and resumes afterwards.
    """

    def __init__(self, lab: StrategyLab, research: ResearchLabs):
        self.lab = lab
        self.research = research
        self.state = CoordinatorState()
        self._task: Optional[asyncio.Task] = None
        self._seen_validations: set[str] = set()

    def public_state(self) -> dict:
        return asdict(self.state)

    async def start(self):
        if self.state.running:
            return
        self.state = CoordinatorState(
            running=True,
            mode="discovery",
            message="Forex discovery has CPU priority; validation is queued only for promising candidates.",
        )
        self._task = asyncio.create_task(self._run(), name="microtrader-research-coordinator")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self.state.mode = "stopped"
        self._task = None

    def _best_promising(self):
        rows = [
            r for r in self.lab.results()
            if r.get("funnel_stage") in {"deep_search", "promoted", "incubator"}
        ]
        if not rows:
            return None

        # Deep Search and Promoted deserve full validation first. Incubator is
        # validated only if the funnel score is unusually strong.
        stage_rank = {"promoted": 3, "deep_search": 2, "incubator": 1}
        rows.sort(
            key=lambda r: (
                stage_rank.get(r.get("funnel_stage"), 0),
                float(r.get("funnel_score") or 0),
                float((r.get("oos") or {}).get("profit_factor") or 0),
                float((r.get("oos") or {}).get("expectancy_bps") or 0),
            ),
            reverse=True,
        )
        top = rows[0]
        if top.get("funnel_stage") == "incubator" and float(top.get("funnel_score") or 0) < 70:
            return None
        return top

    async def _run_validation(self, row: dict, sig: str):
        self.state.mode = "validation"
        self.state.queued_signature = sig
        self.state.queued_strategy = str(row.get("strategy") or "")
        self.state.queued_stage = str(row.get("funnel_stage") or "")
        self.state.message = (
            f"Pausing forex discovery to run full research validation on "
            f"{self.state.queued_strategy} ({self.state.queued_stage})."
        )
        log.info("Coordinator validation start: strategy=%s stage=%s signature=%s",
                 self.state.queued_strategy, self.state.queued_stage, sig[:12])
        await self.lab.pause(self.state.message)
        self.state.validations_started += 1
        try:
            if self.research.state.running:
                await self.research.stop()
            await self.research.start()

            # Research Labs already run their own 17-stage sequence.
            while self.research.state.running and self.state.running:
                await asyncio.sleep(2)

            if self.research.state.last_error:
                raise RuntimeError(self.research.state.last_error)

            self._seen_validations.add(sig)
            self.state.validations_completed += 1
            self.state.last_validation_signature = sig
            self.state.last_validation_at = datetime.now(timezone.utc).isoformat()
            self.state.message = "Validation completed; forex discovery resumed."
            log.info("Coordinator validation complete: strategy=%s signature=%s",
                     self.state.queued_strategy, sig[:12])
        finally:
            await self.lab.resume()
            log.info("Coordinator resumed Strategy Lab")
            self.state.mode = "discovery"
            self.state.queued_signature = ""
            self.state.queued_strategy = ""
            self.state.queued_stage = ""

    async def _run(self):
        try:
            # If Research Labs were auto-started by an older environment
            # setting, stop them so discovery and validation do not compete.
            if self.research.state.running:
                await self.research.stop()

            while self.state.running:
                row = self._best_promising()
                if row:
                    sig = candidate_signature(
                        Candidate(str(row.get("strategy")), dict(row.get("params") or {}))
                    )
                    if sig not in self._seen_validations:
                        await self._run_validation(row, sig)
                    else:
                        self.state.mode = "discovery"
                        self.state.message = "Discovery active; strongest promising candidate already validated."
                else:
                    self.state.mode = "discovery"
                    self.state.message = "Discovery active; waiting for a candidate strong enough for validation."
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.mode = "error"
            self.state.message = str(exc)
            log.exception("Research coordinator failed")
            try:
                await self.lab.resume()
            except Exception:
                pass
        finally:
            self.state.running = False
