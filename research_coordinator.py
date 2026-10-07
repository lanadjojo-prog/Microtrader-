from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from research_labs import ResearchLabs
from research_store import RESEARCH_VALIDATION_VERSION
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
        try:
            await self.research.store.init()
            self._seen_validations = await self.research.store.load_validated_signatures()
        except Exception as exc:
            log.warning("Could not restore validation history: %s", exc)
        self.state = CoordinatorState(
            running=True,
            mode="discovery",
            message=(
                "Forex discovery has CPU priority; exact frozen candidates are "
                "validated once and validation history survives restarts."
            ),
            validations_completed=len(self._seen_validations),
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
        rows = []
        for row in self.lab.analysis_results():
            if str(row.get("strategy") or "") != "adaptive_router":
                continue
            if row.get("funnel_stage") != "promoted" or not bool(row.get("promoted")):
                continue
            sig = candidate_signature(
                Candidate(
                    str(row.get("strategy") or ""),
                    dict(row.get("params") or {}),
                )
            )
            if sig in self._seen_validations:
                continue
            rows.append(row)
        if not rows:
            return None
        rows.sort(
            key=lambda r: (
                float(r.get("funnel_score") or 0),
                float((r.get("oos") or {}).get("profit_factor") or 0),
                float((r.get("oos") or {}).get("expectancy_bps") or 0),
            ),
            reverse=True,
        )
        return rows[0]

    async def _run_validation(self, row: dict, sig: str):
        self.state.mode = "validation"
        self.state.queued_signature = sig
        self.state.queued_strategy = str(row.get("strategy") or "")
        self.state.queued_stage = str(row.get("funnel_stage") or "")
        self.state.message = (
            f"Pausing forex discovery to validate the exact frozen "
            f"{self.state.queued_strategy} configuration."
        )
        log.info(
            "Coordinator validation start: strategy=%s stage=%s signature=%s",
            self.state.queued_strategy,
            self.state.queued_stage,
            sig[:12],
        )
        await self.lab.pause(self.state.message)
        self.state.validations_started += 1
        candidate = Candidate(
            self.state.queued_strategy,
            dict(row.get("params") or {}),
        )
        try:
            if self.research.state.running:
                await self.research.stop()
            await self.research.start(candidate=candidate)

            while self.research.state.running and self.state.running:
                await asyncio.sleep(2)

            if self.research.state.last_error:
                raise RuntimeError(self.research.state.last_error)

            master = (self.research.public_state().get("labs") or {}).get("master") or {}
            summary = {
                "validation_version": RESEARCH_VALIDATION_VERSION,
                "passed": bool(master.get("passed")),
                "completed_checks": self.research.state.completed_labs,
                "total_checks": self.research.state.total_labs,
                "master": master,
            }
            await self.research.store.save_validation(
                candidate_signature=sig,
                strategy=candidate.strategy,
                params=candidate.params,
                status="completed",
                summary=summary,
            )
            self._seen_validations.add(sig)
            self.state.validations_completed = len(self._seen_validations)
            self.state.last_validation_signature = sig
            self.state.last_validation_at = datetime.now(timezone.utc).isoformat()
            self.state.message = (
                "Validation passed; paper eligibility unlocked."
                if summary["passed"]
                else "Validation rejected; discovery resumed."
            )
            log.info(
                "Coordinator validation complete: strategy=%s passed=%s signature=%s",
                self.state.queued_strategy,
                summary["passed"],
                sig[:12],
            )
        except Exception as exc:
            try:
                await self.research.store.save_validation(
                    candidate_signature=sig,
                    strategy=candidate.strategy,
                    params=candidate.params,
                    status="failed",
                    summary={"error": str(exc)},
                )
            except Exception:
                pass
            # Do not retry the same broken validation every ten seconds in the
            # same process. A restart can retry it after code/data changes.
            self._seen_validations.add(sig)
            raise
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
                        self.state.message = "Discovery active; waiting for the next unvalidated promoted candidate."
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
