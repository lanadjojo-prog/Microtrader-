from __future__ import annotations

import json
from typing import Any, Dict, List
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row


class ForexStrategyStore:
    """Append-only persistence for forex research.

    Every evaluation is stored as its own run. We deliberately do not overwrite
    previous runs so parameter changes, data windows and metric changes remain
    auditable per strategy.
    """

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
                CREATE TABLE IF NOT EXISTS microtrader_forex_strategy_runs (
                    run_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    family TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params JSONB NOT NULL,
                    pairs JSONB NOT NULL,
                    timeframe_min INTEGER,
                    dataset JSONB NOT NULL DEFAULT '{}'::jsonb,
                    risk_model JSONB NOT NULL DEFAULT '{}'::jsonb,
                    train JSONB NOT NULL DEFAULT '{}'::jsonb,
                    oos JSONB NOT NULL DEFAULT '{}'::jsonb,
                    stress_oos JSONB NOT NULL DEFAULT '{}'::jsonb,
                    per_pair JSONB NOT NULL DEFAULT '{}'::jsonb,
                    robustness JSONB NOT NULL DEFAULT '{}'::jsonb,
                    funnel_score DOUBLE PRECISION,
                    promoted BOOLEAN NOT NULL DEFAULT FALSE,
                    rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
                    tested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_forex_runs_signature
                ON microtrader_forex_strategy_runs(signature)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_forex_runs_score
                ON microtrader_forex_strategy_runs(funnel_score DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_forex_runs_tested_at
                ON microtrader_forex_strategy_runs(tested_at DESC)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_forex_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    generation INTEGER NOT NULL DEFAULT 0,
                    tested_total INTEGER NOT NULL DEFAULT 0,
                    promoted_total INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                INSERT INTO microtrader_forex_state (id)
                VALUES (1)
                ON CONFLICT (id) DO NOTHING
            """)
            await conn.commit()

    async def save_run(self, signature: str, result: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        run_id = str(uuid4())
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_forex_strategy_runs (
                    run_id, signature, strategy, family, phase, status,
                    params, pairs, timeframe_min, dataset, risk_model,
                    train, oos, stress_oos, per_pair, robustness, funnel_score,
                    promoted, rejection_reasons, tested_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    %s, %s::jsonb, NOW()
                )
                """,
                (
                    run_id,
                    signature,
                    result.get("strategy", ""),
                    result.get("family", result.get("strategy", "")),
                    result.get("phase", "discovery"),
                    result.get("status", result.get("funnel_stage", "tested")),
                    json.dumps(result.get("params", {})),
                    json.dumps(result.get("pairs", [])),
                    result.get("timeframe_min"),
                    json.dumps(result.get("dataset", {})),
                    json.dumps(result.get("risk_model", {})),
                    json.dumps(result.get("train", {})),
                    json.dumps(result.get("oos", {})),
                    json.dumps(result.get("stress_oos", {})),
                    json.dumps(result.get("per_pair", {})),
                    json.dumps(result.get("robustness", {})),
                    result.get("funnel_score"),
                    bool(result.get("promoted", False)),
                    json.dumps(result.get("rejection_reasons", [])),
                ),
            )
            await conn.commit()
        return run_id

    async def save_state(self, generation: int, tested_total: int, promoted_total: int) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_forex_state
                SET generation=%s, tested_total=%s, promoted_total=%s, updated_at=NOW()
                WHERE id=1
                """,
                (generation, tested_total, promoted_total),
            )
            await conn.commit()

    async def load_state(self) -> Dict[str, int]:
        if not self.enabled:
            return {"generation": 0, "tested_total": 0, "promoted_total": 0}
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT generation, tested_total, promoted_total "
                "FROM microtrader_forex_state WHERE id=1"
            )
            row = await cur.fetchone()
            return dict(row) if row else {
                "generation": 0,
                "tested_total": 0,
                "promoted_total": 0,
            }

    async def load_results(self, limit: int = 250) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT run_id, signature, strategy, family, phase, status,
                       params, pairs, timeframe_min, dataset, risk_model,
                       train, oos, stress_oos, per_pair, robustness, funnel_score,
                       promoted, rejection_reasons, tested_at
                FROM microtrader_forex_strategy_runs
                ORDER BY promoted DESC,
                         funnel_score DESC NULLS LAST,
                         (oos->>'expectancy_r')::double precision DESC NULLS LAST,
                         tested_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def load_signatures(self) -> set[str]:
        if not self.enabled:
            return set()
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT DISTINCT signature FROM microtrader_forex_strategy_runs"
            )
            rows = await cur.fetchall()
            return {str(row[0]) for row in rows}


    async def load_promoted(
        self, limit: int = 100, min_trades_per_day: float = 0.0
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT run_id, signature, strategy, family, phase, status,
                       params, pairs, timeframe_min, dataset, risk_model,
                       train, oos, stress_oos, per_pair, robustness, funnel_score,
                       promoted, rejection_reasons, tested_at
                FROM microtrader_forex_strategy_runs
                WHERE (promoted = TRUE OR status = 'promoted')
                  AND timeframe_min IN (1, 5)
                  AND COALESCE(params->>'entry_sessions', '') = 'london_new_york'
                  AND COALESCE(robustness->>'evaluation_policy_version', '') = 'forex-funnel-v3-frozen-holdout20'
                  AND COALESCE((oos->>'avg_trades_per_day')::double precision, 0) >= %s
                ORDER BY tested_at ASC
                LIMIT %s
                """,
                (float(min_trades_per_day), limit),
            )
            return [dict(row) for row in await cur.fetchall()]
