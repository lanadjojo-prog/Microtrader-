from __future__ import annotations

import json
from typing import Any, Dict, List
import psycopg
from psycopg.rows import dict_row


class ResearchStore:
    def __init__(self, database_url: str):
        self.database_url = database_url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    async def init(self) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
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
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
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
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
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
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                """
                SELECT candidate_signature
                FROM microtrader_research_validations
                WHERE status='completed'
                """
            )
            return {str(row[0]) for row in await cur.fetchall()}

    async def load_all(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(self.database_url,row_factory=dict_row) as conn:
            cur=await conn.execute("""
                SELECT lab_name,status,result,updated_at
                FROM microtrader_research_lab_results
                ORDER BY lab_name
            """)
            return [dict(r) for r in await cur.fetchall()]
