from __future__ import annotations

import json
from typing import Any, Dict, List
import psycopg
from db_connection import connect_db
from psycopg.rows import dict_row

RESEARCH_VALIDATION_VERSION = "robust-v2-walkforward-stability-cost"


class ResearchStore:
    def __init__(self, database_url: str):
        self.database_url = database_url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    async def init(self) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_research_lab_results (
                    lab_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_research_validations (
                    candidate_signature TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    params JSONB NOT NULL,
                    status TEXT NOT NULL,
                    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.commit()

    async def save(self, lab_name: str, status: str, result: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute("""
                INSERT INTO microtrader_research_lab_results (lab_name,status,result,updated_at)
                VALUES (%s,%s,%s::jsonb,NOW())
                ON CONFLICT (lab_name) DO UPDATE SET
                    status=EXCLUDED.status,
                    result=EXCLUDED.result,
                    updated_at=NOW()
            """,(lab_name,status,json.dumps(result)))
            await conn.commit()

    async def save_validation(
        self,
        *,
        candidate_signature: str,
        strategy: str,
        params: Dict[str, Any],
        status: str,
        summary: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_research_validations (
                    candidate_signature, strategy, params, status, summary, updated_at
                ) VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,NOW())
                ON CONFLICT (candidate_signature) DO UPDATE SET
                    strategy=EXCLUDED.strategy,
                    params=EXCLUDED.params,
                    status=EXCLUDED.status,
                    summary=EXCLUDED.summary,
                    updated_at=NOW()
                """,
                (
                    str(candidate_signature),
                    str(strategy),
                    json.dumps(params),
                    str(status),
                    json.dumps(summary),
                ),
            )
            await conn.commit()

    async def load_validated_signatures(self) -> set[str]:
        if not self.enabled:
            return set()
        async with connect_db(self.database_url) as conn:
            cur = await conn.execute(
                """
                SELECT candidate_signature
                FROM microtrader_research_validations
                WHERE status='completed'
                  AND COALESCE(summary->>'validation_version', '') = %s
                """,
                (RESEARCH_VALIDATION_VERSION,),
            )
            return {str(row[0]) for row in await cur.fetchall()}

    async def load_paper_eligible_validations(
        self,
        *,
        research_policy_version: str,
        min_trades_per_day: float,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Frozen Research candidates that passed the simplified Validation gate."""
        if not self.enabled:
            return []
        async with connect_db(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT DISTINCT ON (v.candidate_signature)
                       v.candidate_signature,
                       v.strategy,
                       v.params,
                       v.summary,
                       r.oos,
                       r.stress_oos,
                       r.funnel_score,
                       r.adaptive_diagnostics,
                       r.tested_at
                FROM microtrader_research_validations v
                JOIN microtrader_forex_research_strategy_results r
                  ON r.strategy = v.strategy
                 AND r.params = v.params
                WHERE v.status = 'completed'
                  AND COALESCE(v.summary->>'validation_version', '') = %s
                  AND COALESCE((v.summary->>'passed')::boolean, FALSE) = TRUE
                  AND r.funnel_stage = 'promoted'
                  AND r.promoted = TRUE
                  AND COALESCE(r.params->>'_policy_version', '') = %s
                  AND COALESCE((r.oos->>'avg_trades_per_day')::double precision, 0) >= %s
                ORDER BY v.candidate_signature, r.tested_at DESC
                LIMIT %s
                """,
                (
                    RESEARCH_VALIDATION_VERSION,
                    str(research_policy_version),
                    float(min_trades_per_day),
                    max(1, int(limit)),
                ),
            )
            return [dict(row) for row in await cur.fetchall()]

    async def load_all(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with connect_db(self.database_url, row_factory=dict_row) as conn:
            cur=await conn.execute("""
                SELECT lab_name,status,result,updated_at
                FROM microtrader_research_lab_results
                ORDER BY lab_name
            """)
            return [dict(r) for r in await cur.fetchall()]
