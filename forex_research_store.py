from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row


class ForexResearchStore:
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
                CREATE TABLE IF NOT EXISTS microtrader_forex_research_strategy_results (
                    signature TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    params JSONB NOT NULL,
                    promoted BOOLEAN NOT NULL,
                    rejection_reasons JSONB NOT NULL,
                    train JSONB NOT NULL,
                    oos JSONB NOT NULL,
                    stress_oos JSONB NOT NULL,
                    positive_symbol_ratio DOUBLE PRECISION,
                    positive_symbols INTEGER,
                    symbol_count INTEGER,
                    per_symbol JSONB NOT NULL DEFAULT '{}'::jsonb,
                    tested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                ALTER TABLE microtrader_forex_research_strategy_results
                ADD COLUMN IF NOT EXISTS family TEXT,
                ADD COLUMN IF NOT EXISTS funnel_stage TEXT,
                ADD COLUMN IF NOT EXISTS funnel_score DOUBLE PRECISION
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_forex_research_strategy_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    generation INTEGER NOT NULL DEFAULT 0,
                    tested_total INTEGER NOT NULL DEFAULT 0,
                    promoted_total INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                INSERT INTO microtrader_forex_research_strategy_state (id)
                VALUES (1)
                ON CONFLICT (id) DO NOTHING
            """)
            await conn.commit()

    async def save_result(self, signature: str, result: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_forex_research_strategy_results (
                    signature, strategy, params, promoted, rejection_reasons,
                    train, oos, stress_oos, positive_symbol_ratio,
                    positive_symbols, symbol_count, per_symbol,
                    family, funnel_stage, funnel_score, tested_at
                ) VALUES (
                    %s, %s, %s::jsonb, %s, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s::jsonb,
                    %s, %s, %s, NOW()
                )
                ON CONFLICT (signature) DO UPDATE SET
                    promoted = EXCLUDED.promoted,
                    rejection_reasons = EXCLUDED.rejection_reasons,
                    train = EXCLUDED.train,
                    oos = EXCLUDED.oos,
                    stress_oos = EXCLUDED.stress_oos,
                    positive_symbol_ratio = EXCLUDED.positive_symbol_ratio,
                    positive_symbols = EXCLUDED.positive_symbols,
                    symbol_count = EXCLUDED.symbol_count,
                    per_symbol = EXCLUDED.per_symbol,
                    family = EXCLUDED.family,
                    funnel_stage = EXCLUDED.funnel_stage,
                    funnel_score = EXCLUDED.funnel_score,
                    tested_at = NOW()
                """,
                (
                    signature,
                    result["strategy"],
                    json.dumps(result["params"]),
                    bool(result["promoted"]),
                    json.dumps(result.get("rejection_reasons", [])),
                    json.dumps(result.get("train", {})),
                    json.dumps(result.get("oos", {})),
                    json.dumps(result.get("stress_oos", {})),
                    result.get("positive_symbol_ratio"),
                    result.get("positive_symbols"),
                    result.get("symbol_count"),
                    json.dumps(result.get("per_symbol", {})),
                    result.get("family"),
                    result.get("funnel_stage"),
                    result.get("funnel_score"),
                ),
            )
            await conn.commit()

    def save_checkpoint_sync(
        self,
        signature: str,
        result: Dict[str, Any],
        generation: int,
        tested_total: int,
        promoted_total: int,
    ) -> None:
        """Synchronous checkpoint used from a worker thread.

        Keeping network/DNS/database work off the asyncio event loop prevents
        dashboard and strategy-search stalls when the remote pooler is slow.
        """
        if not self.enabled:
            return
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            conn.execute(
                """
                INSERT INTO microtrader_forex_research_strategy_results (
                    signature, strategy, params, promoted, rejection_reasons,
                    train, oos, stress_oos, positive_symbol_ratio,
                    positive_symbols, symbol_count, per_symbol,
                    family, funnel_stage, funnel_score, tested_at
                ) VALUES (
                    %s, %s, %s::jsonb, %s, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s::jsonb,
                    %s, %s, %s, NOW()
                )
                ON CONFLICT (signature) DO UPDATE SET
                    promoted = EXCLUDED.promoted,
                    rejection_reasons = EXCLUDED.rejection_reasons,
                    train = EXCLUDED.train,
                    oos = EXCLUDED.oos,
                    stress_oos = EXCLUDED.stress_oos,
                    positive_symbol_ratio = EXCLUDED.positive_symbol_ratio,
                    positive_symbols = EXCLUDED.positive_symbols,
                    symbol_count = EXCLUDED.symbol_count,
                    per_symbol = EXCLUDED.per_symbol,
                    family = EXCLUDED.family,
                    funnel_stage = EXCLUDED.funnel_stage,
                    funnel_score = EXCLUDED.funnel_score,
                    tested_at = NOW()
                """,
                (
                    signature,
                    result["strategy"],
                    json.dumps(result["params"]),
                    bool(result["promoted"]),
                    json.dumps(result.get("rejection_reasons", [])),
                    json.dumps(result.get("train", {})),
                    json.dumps(result.get("oos", {})),
                    json.dumps(result.get("stress_oos", {})),
                    result.get("positive_symbol_ratio"),
                    result.get("positive_symbols"),
                    result.get("symbol_count"),
                    json.dumps(result.get("per_symbol", {})),
                    result.get("family"),
                    result.get("funnel_stage"),
                    result.get("funnel_score"),
                ),
            )
            conn.execute(
                """
                UPDATE microtrader_forex_research_strategy_state
                SET generation=%s, tested_total=%s, promoted_total=%s, updated_at=NOW()
                WHERE id=1
                """,
                (generation, tested_total, promoted_total),
            )
            conn.commit()

    async def save_checkpoint(
        self,
        signature: str,
        result: Dict[str, Any],
        generation: int,
        tested_total: int,
        promoted_total: int,
    ) -> None:
        """Persist one evaluated candidate and the matching counter atomically.

        A short connect timeout prevents a slow database/pooler from freezing the
        entire research loop.
        """
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(
            self.database_url, connect_timeout=5
        ) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_forex_research_strategy_results (
                    signature, strategy, params, promoted, rejection_reasons,
                    train, oos, stress_oos, positive_symbol_ratio,
                    positive_symbols, symbol_count, per_symbol,
                    family, funnel_stage, funnel_score, tested_at
                ) VALUES (
                    %s, %s, %s::jsonb, %s, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s::jsonb,
                    %s, %s, %s, NOW()
                )
                ON CONFLICT (signature) DO UPDATE SET
                    promoted = EXCLUDED.promoted,
                    rejection_reasons = EXCLUDED.rejection_reasons,
                    train = EXCLUDED.train,
                    oos = EXCLUDED.oos,
                    stress_oos = EXCLUDED.stress_oos,
                    positive_symbol_ratio = EXCLUDED.positive_symbol_ratio,
                    positive_symbols = EXCLUDED.positive_symbols,
                    symbol_count = EXCLUDED.symbol_count,
                    per_symbol = EXCLUDED.per_symbol,
                    family = EXCLUDED.family,
                    funnel_stage = EXCLUDED.funnel_stage,
                    funnel_score = EXCLUDED.funnel_score,
                    tested_at = NOW()
                """,
                (
                    signature,
                    result["strategy"],
                    json.dumps(result["params"]),
                    bool(result["promoted"]),
                    json.dumps(result.get("rejection_reasons", [])),
                    json.dumps(result.get("train", {})),
                    json.dumps(result.get("oos", {})),
                    json.dumps(result.get("stress_oos", {})),
                    result.get("positive_symbol_ratio"),
                    result.get("positive_symbols"),
                    result.get("symbol_count"),
                    json.dumps(result.get("per_symbol", {})),
                    result.get("family"),
                    result.get("funnel_stage"),
                    result.get("funnel_score"),
                ),
            )
            await conn.execute(
                """
                UPDATE microtrader_forex_research_strategy_state
                SET generation=%s, tested_total=%s, promoted_total=%s, updated_at=NOW()
                WHERE id=1
                """,
                (generation, tested_total, promoted_total),
            )
            await conn.commit()

    async def save_state(self, generation: int, tested_total: int, promoted_total: int) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_forex_research_strategy_state
                SET generation=%s, tested_total=%s, promoted_total=%s, updated_at=NOW()
                WHERE id=1
                """,
                (generation, tested_total, promoted_total),
            )
            await conn.commit()

    async def load_state(self) -> Dict[str, int]:
        if not self.enabled:
            return {"generation": 0, "tested_total": 0, "promoted_total": 0}
        async with await psycopg.AsyncConnection.connect(self.database_url, row_factory=dict_row) as conn:
            cur = await conn.execute(
                "SELECT generation, tested_total, promoted_total FROM microtrader_forex_research_strategy_state WHERE id=1"
            )
            row = await cur.fetchone()
            return dict(row) if row else {"generation": 0, "tested_total": 0, "promoted_total": 0}

    async def load_results(self, limit: int = 250) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(self.database_url, row_factory=dict_row) as conn:
            cur = await conn.execute(
                """
                SELECT signature, strategy, params, promoted, rejection_reasons,
                       train, oos, stress_oos, positive_symbol_ratio,
                       positive_symbols, symbol_count, per_symbol,
                       family, funnel_stage, funnel_score, tested_at
                FROM microtrader_forex_research_strategy_results
                ORDER BY promoted DESC,
                         funnel_score DESC NULLS LAST,
                         (oos->>'expectancy_bps')::double precision DESC NULLS LAST,
                         tested_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def load_research_memory(
        self,
        *,
        per_family_stage: int = 40,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load a diverse, bounded research memory after restarts.

        Keeping only the global top-N can let one strategy family crowd every
        other family out of the next search generation. This query keeps the
        strongest rows per family/stage while preserving every promoted row.
        """
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                WITH ranked AS (
                    SELECT signature, strategy, params, promoted, rejection_reasons,
                           train, oos, stress_oos, positive_symbol_ratio,
                           positive_symbols, symbol_count, per_symbol,
                           family, funnel_stage, funnel_score, tested_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY COALESCE(family, strategy),
                                            COALESCE(funnel_stage, 'rejected')
                               ORDER BY promoted DESC,
                                        funnel_score DESC NULLS LAST,
                                        (oos->>'expectancy_bps')::double precision DESC NULLS LAST,
                                        tested_at DESC
                           ) AS family_stage_rank
                    FROM microtrader_forex_research_strategy_results
                )
                SELECT signature, strategy, params, promoted, rejection_reasons,
                       train, oos, stress_oos, positive_symbol_ratio,
                       positive_symbols, symbol_count, per_symbol,
                       family, funnel_stage, funnel_score, tested_at
                FROM ranked
                WHERE promoted = TRUE OR family_stage_rank <= %s
                ORDER BY promoted DESC,
                         funnel_score DESC NULLS LAST,
                         (oos->>'expectancy_bps')::double precision DESC NULLS LAST,
                         tested_at DESC
                LIMIT %s
                """,
                (max(1, int(per_family_stage)), max(1, int(limit))),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def funnel_summary(self, policy_version: str) -> List[Dict[str, Any]]:
        """Return active candidate counts, not cumulative phase history.

        A configuration can move through several phases. Counting every stored
        evaluation makes one configuration appear multiple times. Keep only the
        most recent row for each strategy + phase-independent parameter set.
        """
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                WITH active AS (
                    SELECT strategy, family, funnel_stage, tested_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY strategy,
                                            (params - '_phase' - '_policy_version')
                               ORDER BY tested_at DESC
                           ) AS rn
                    FROM microtrader_forex_research_strategy_results
                    WHERE COALESCE(params->>'_policy_version', '') = %s
                )
                SELECT COALESCE(family, strategy) AS family,
                       COALESCE(funnel_stage, 'rejected') AS funnel_stage,
                       COUNT(*)::integer AS candidates
                FROM active
                WHERE rn = 1
                GROUP BY COALESCE(family, strategy),
                         COALESCE(funnel_stage, 'rejected')
                ORDER BY funnel_stage, candidates DESC, family
                """,
                (str(policy_version),),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def load_signatures(self) -> set[str]:
        if not self.enabled:
            return set()
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT signature FROM microtrader_forex_research_strategy_results"
            )
            rows = await cur.fetchall()
            return {str(r[0]) for r in rows}
