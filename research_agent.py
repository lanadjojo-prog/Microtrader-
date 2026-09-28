from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Optional

import psycopg

from config import Settings
from strategy_lab import StrategyLab

log = logging.getLogger("microtrader.research_agent")


@dataclass
class ResearchAgentState:
    running: bool = False
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_error: Optional[str] = None
    cycles: int = 0
    stage: str = "idle"
    message: str = "Ready"
    focus_families: list[str] | None = None
    focus_timeframes: list[int] | None = None
    hypotheses: list[dict] | None = None
    last_decision: dict | None = None


class ResearchAgent:
    """
    Autonomous research director.

    It can:
    - inspect Strategy Lab results
    - rank strategy families and timeframes
    - create research hypotheses
    - steer the next Strategy Lab batches toward promising areas

    It cannot place, modify, or cancel broker orders.
    """

    def __init__(self, settings: Settings, lab: StrategyLab):
        self.settings = settings
        self.lab = lab
        self.state = ResearchAgentState(
            focus_families=[],
            focus_timeframes=[],
            hypotheses=[],
        )
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self.state.running:
            return
        self.state.running = True
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self.state.stage = "starting"
        self.state.message = "Research agent starting"
        try:
            await self._init_store()
        except Exception as exc:
            self.state.last_error = f"Persistence unavailable: {exc}"
            log.warning("Research agent persistence unavailable at startup: %s", exc)
        self._task = asyncio.create_task(self._run(), name="microtrader-research-agent")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self.state.stage = "stopped"
        self.state.message = "Research agent stopped"
        self.state.stopped_at = datetime.now(timezone.utc).isoformat()
        self._task = None

    async def _run(self):
        try:
            while self.state.running:
                await self.cycle()
                await asyncio.sleep(max(20, self.settings.research_agent_interval_seconds))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Research agent failed")
        finally:
            self.state.running = False

    async def cycle(self):
        self.state.stage = "analyzing"
        results = [
            r for r in self.lab.results()
            if (r.get("params") or {}).get("_phase") in {"discovery", "incubator", "deep_search"}
        ]

        if not results:
            decision = {
                "action": "wait_for_discovery",
                "reason": "Strategy Lab has not produced new-funnel results yet.",
                "focus_families": [],
                "focus_timeframes": [],
            }
            self.lab.set_agent_focus([], [], decision["reason"])
            self._commit_decision(decision, [])
            return

        family_rows = defaultdict(list)
        tf_rows = defaultdict(list)
        for r in results:
            family_rows[str(r.get("family") or r.get("strategy") or "unknown")].append(r)
            tf = int((r.get("params") or {}).get("timeframe_min") or 0)
            if tf:
                tf_rows[tf].append(r)

        family_summary = []
        for family, rows in family_rows.items():
            scores = [float(r.get("funnel_score") or 0) for r in rows]
            exps = [float((r.get("oos") or {}).get("expectancy_bps") or 0) for r in rows]
            pfs = [float((r.get("oos") or {}).get("profit_factor") or 0) for r in rows]
            payoffs = [float((r.get("oos") or {}).get("payoff_ratio") or 0) for r in rows]
            deep = sum(1 for r in rows if r.get("funnel_stage") == "deep_search")
            incubator = sum(1 for r in rows if r.get("funnel_stage") == "incubator")
            promoted = sum(1 for r in rows if r.get("promoted"))
            family_summary.append({
                "family": family,
                "tested": len(rows),
                "best_score": round(max(scores or [0]), 2),
                "avg_score": round(mean(scores), 2) if scores else 0.0,
                "best_expectancy_bps": round(max(exps or [0]), 3),
                "best_profit_factor": round(max(pfs or [0]), 3),
                "best_payoff_ratio": round(max(payoffs or [0]), 3),
                "incubator": incubator,
                "deep_search": deep,
                "promoted": promoted,
            })

        family_summary.sort(
            key=lambda x: (
                x["promoted"],
                x["deep_search"],
                x["incubator"],
                x["best_score"],
                x["best_expectancy_bps"],
            ),
            reverse=True,
        )

        # Explore broadly until every family has at least a few observations.
        underexplored = [x for x in family_summary if x["tested"] < 4]
        if underexplored:
            focus = underexplored[: self.settings.research_agent_focus_families]
            mode = "explore"
            reason = "Broad discovery is incomplete; prioritize under-tested families before deeper optimization."
        else:
            focus = family_summary[: self.settings.research_agent_focus_families]
            mode = "exploit"
            reason = "Discovery coverage is sufficient; prioritize families with the strongest robust out-of-sample signals."

        tf_summary = []
        for tf, rows in tf_rows.items():
            scores = [float(r.get("funnel_score") or 0) for r in rows]
            exps = [float((r.get("oos") or {}).get("expectancy_bps") or 0) for r in rows]
            tf_summary.append({
                "timeframe": tf,
                "tested": len(rows),
                "best_score": max(scores or [0]),
                "best_expectancy_bps": max(exps or [0]),
            })
        tf_summary.sort(key=lambda x: (x["best_score"], x["best_expectancy_bps"]), reverse=True)

        focus_families = [x["family"] for x in focus]
        focus_timeframes = [int(x["timeframe"]) for x in tf_summary[:2]] or [1, 3]
        hypotheses = self._make_hypotheses(focus, tf_summary, results)

        decision = {
            "action": "steer_strategy_funnel",
            "mode": mode,
            "reason": reason,
            "focus_families": focus_families,
            "focus_timeframes": focus_timeframes,
            "family_summary": family_summary[:10],
            "timeframe_summary": tf_summary,
        }

        self.lab.set_agent_focus(focus_families, focus_timeframes, reason)
        self._commit_decision(decision, hypotheses)
        await self._persist_decision(decision, hypotheses)

    def _make_hypotheses(self, focus: list[dict], tf_summary: list[dict], results: list[dict]) -> list[dict]:
        out = []
        for row in focus[:5]:
            fam = row["family"]
            fam_rows = [r for r in results if (r.get("family") or r.get("strategy")) == fam]
            fam_rows.sort(key=lambda r: float(r.get("funnel_score") or 0), reverse=True)
            best = fam_rows[0] if fam_rows else {}
            oos = best.get("oos") or {}
            params = best.get("params") or {}
            payoff = float(oos.get("payoff_ratio") or 0)
            winrate = float(oos.get("win_rate_pct") or 0)
            exp = float(oos.get("expectancy_bps") or 0)

            if payoff >= 2.0 and winrate < 50 and exp > 0:
                thesis = f"{fam} may have a positively skewed payoff profile worth deeper R-multiple testing."
            elif exp > 0:
                thesis = f"{fam} shows positive out-of-sample expectancy and should receive local parameter refinement."
            else:
                thesis = f"{fam} remains under-tested or inconclusive; gather broader evidence before rejection."

            out.append({
                "family": fam,
                "timeframe_min": params.get("timeframe_min"),
                "funnel_score": best.get("funnel_score"),
                "expectancy_bps": oos.get("expectancy_bps"),
                "profit_factor": oos.get("profit_factor"),
                "win_rate_pct": oos.get("win_rate_pct"),
                "payoff_ratio": oos.get("payoff_ratio"),
                "thesis": thesis,
            })
        return out

    def _commit_decision(self, decision: dict, hypotheses: list[dict]):
        now = datetime.now(timezone.utc).isoformat()
        self.state.cycles += 1
        self.state.last_cycle_at = now
        self.state.stage = "directing"
        self.state.focus_families = list(decision.get("focus_families") or [])
        self.state.focus_timeframes = list(decision.get("focus_timeframes") or [])
        self.state.hypotheses = hypotheses[:8]
        self.state.last_decision = decision
        self.state.message = decision.get("reason", "Research priorities updated")

    async def _init_store(self):
        if not self.settings.database_url:
            return
        async with await psycopg.AsyncConnection.connect(self.settings.database_url) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_agent_decisions (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    decision JSONB NOT NULL,
                    hypotheses JSONB NOT NULL
                )
            """)
            await conn.commit()

    async def _persist_decision(self, decision: dict, hypotheses: list[dict]):
        if not self.settings.database_url:
            return
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.database_url) as conn:
                await conn.execute(
                    """
                    INSERT INTO microtrader_agent_decisions (decision, hypotheses)
                    VALUES (%s::jsonb, %s::jsonb)
                    """,
                    (json.dumps(decision), json.dumps(hypotheses)),
                )
                await conn.commit()
        except Exception as exc:
            log.warning("Could not persist research-agent decision: %s", exc)

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["permissions"] = {
            "can_read_research": True,
            "can_steer_experiments": True,
            "can_place_orders": False,
            "can_enable_live_trading": False,
        }
        payload["strategy_lab_focus"] = self.lab.agent_focus()
        return payload
