from __future__ import annotations

import json
from typing import Any, Dict, List
from uuid import uuid4

import psycopg
from db_connection import connect_db
from psycopg.rows import dict_row


class PrecisionStrategyStore:
    """Append-only precision-lab storage, including individual trades."""

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
                CREATE TABLE IF NOT EXISTS microtrader_precision_runs (
                    run_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    family TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params JSONB NOT NULL,
                    pairs JSONB NOT NULL,
                    dataset JSONB NOT NULL DEFAULT '{}'::jsonb,
                    risk_model JSONB NOT NULL DEFAULT '{}'::jsonb,
                    execution_model JSONB NOT NULL DEFAULT '{}'::jsonb,
                    train JSONB NOT NULL DEFAULT '{}'::jsonb,
                    oos JSONB NOT NULL DEFAULT '{}'::jsonb,
                    stress_oos JSONB NOT NULL DEFAULT '{}'::jsonb,
                    per_pair JSONB NOT NULL DEFAULT '{}'::jsonb,
                    funnel_score DOUBLE PRECISION,
                    positive_pair_ratio DOUBLE PRECISION,
                    avg_fill_rate_pct DOUBLE PRECISION,
                    rejection_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
                    tested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_precision_runs_signature
                ON microtrader_precision_runs(signature)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_precision_runs_score
                ON microtrader_precision_runs(funnel_score DESC)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_precision_trades (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES microtrader_precision_runs(run_id)
                        ON DELETE CASCADE,
                    sample TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    strategy_tag TEXT,
                    order_type TEXT,
                    side TEXT,
                    entry_time TIMESTAMPTZ,
                    exit_time TIMESTAMPTZ,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    stop_pips DOUBLE PRECISION,
                    target_pips DOUBLE PRECISION,
                    gross_pips DOUBLE PRECISION,
                    net_pips DOUBLE PRECISION,
                    entry_spread_pips DOUBLE PRECISION,
                    commission_pips DOUBLE PRECISION,
                    r_multiple DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    risk_eur DOUBLE PRECISION,
                    exit_reason TEXT
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_precision_trades_run
                ON microtrader_precision_trades(run_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_precision_trades_pair_time
                ON microtrader_precision_trades(pair, entry_time)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_precision_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    generation INTEGER NOT NULL DEFAULT 0,
                    tested_total INTEGER NOT NULL DEFAULT 0,
                    deep_search_total INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                INSERT INTO microtrader_precision_state (id)
                VALUES (1)
                ON CONFLICT (id) DO NOTHING
            """)
            await conn.commit()

    async def save_run(self, signature: str, result: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        run_id = str(uuid4())
        train_trades = list(result.get("_train_trades") or [])
        oos_trades = list(result.get("_oos_trades") or [])
        stress_trades = list(result.get("_stress_trades") or [])

        async with connect_db(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_precision_runs (
                    run_id, signature, strategy, family, phase, status,
                    params, pairs, dataset, risk_model, execution_model,
                    train, oos, stress_oos, per_pair, funnel_score,
                    positive_pair_ratio, avg_fill_rate_pct,
                    rejection_reasons, tested_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s::jsonb, NOW()
                )
                """,
                (
                    run_id,
                    signature,
                    result.get("strategy", ""),
                    result.get("family", result.get("strategy", "")),
                    result.get("phase", "precision_discovery"),
                    result.get("status", "tested"),
                    json.dumps(result.get("params", {})),
                    json.dumps(result.get("pairs", [])),
                    json.dumps(result.get("dataset", {})),
                    json.dumps(result.get("risk_model", {})),
                    json.dumps(result.get("execution_model", {})),
                    json.dumps(result.get("train", {})),
                    json.dumps(result.get("oos", {})),
                    json.dumps(result.get("stress_oos", {})),
                    json.dumps(result.get("per_pair", {})),
                    result.get("funnel_score"),
                    result.get("positive_pair_ratio"),
                    result.get("avg_fill_rate_pct"),
                    json.dumps(result.get("rejection_reasons", [])),
                ),
            )

            rows = []
            for sample, trades in (
                ("train", train_trades),
                ("oos", oos_trades),
                ("stress_oos", stress_trades),
            ):
                for t in trades:
                    rows.append((
                        run_id, sample, t.get("pair") or t.get("symbol") or "",
                        t.get("strategy_tag"), t.get("order_type"), t.get("side"),
                        t.get("entry_time"), t.get("exit_time"),
                        t.get("entry_price"), t.get("exit_price"),
                        t.get("stop_pips"), t.get("target_pips"),
                        t.get("gross_pips"), t.get("net_pips"),
                        t.get("entry_spread_pips"), t.get("commission_pips"),
                        t.get("r_multiple"), t.get("pnl"), t.get("risk_eur"),
                        t.get("exit_reason"),
                    ))
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO microtrader_precision_trades (
                        run_id, sample, pair, strategy_tag, order_type, side,
                        entry_time, exit_time, entry_price, exit_price,
                        stop_pips, target_pips, gross_pips, net_pips,
                        entry_spread_pips, commission_pips, r_multiple,
                        pnl, risk_eur, exit_reason
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    rows,
                )
            await conn.commit()
        return run_id

    async def load_results(self, limit: int = 250) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with connect_db(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT run_id, signature, strategy, family, phase, status,
                       params, pairs, dataset, risk_model, execution_model,
                       train, oos, stress_oos, per_pair, funnel_score,
                       positive_pair_ratio, avg_fill_rate_pct,
                       rejection_reasons, tested_at
                FROM microtrader_precision_runs
                ORDER BY funnel_score DESC NULLS LAST,
                         (oos->>'expectancy_r')::double precision DESC NULLS LAST,
                         tested_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in await cur.fetchall()]

    async def load_signatures(self) -> set[str]:
        if not self.enabled:
            return set()
        async with connect_db(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT DISTINCT signature FROM microtrader_precision_runs"
            )
            return {str(row[0]) for row in await cur.fetchall()}

    async def load_state(self) -> Dict[str, int]:
        if not self.enabled:
            return {"generation": 0, "tested_total": 0, "deep_search_total": 0}
        async with connect_db(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT generation, tested_total, deep_search_total "
                "FROM microtrader_precision_state WHERE id=1"
            )
            row = await cur.fetchone()
            return dict(row) if row else {
                "generation": 0, "tested_total": 0, "deep_search_total": 0
            }

    async def save_state(
        self, generation: int, tested_total: int, deep_search_total: int
    ) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_precision_state
                SET generation=%s, tested_total=%s, deep_search_total=%s,
                    updated_at=NOW()
                WHERE id=1
                """,
                (generation, tested_total, deep_search_total),
            )
            await conn.commit()


    async def funnel_summary(
        self, evaluation_policy_version: str
    ) -> List[Dict[str, Any]]:
        """Return uncapped current-policy Precision counts."""
        if not self.enabled:
            return []
        async with connect_db(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                WITH active AS (
                    SELECT strategy, family, status, tested_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY strategy, (params - '_phase')
                               ORDER BY tested_at DESC
                           ) AS rn
                    FROM microtrader_precision_runs
                    WHERE COALESCE(params->>'entry_sessions', '') = 'london_new_york'
                      AND COALESCE(execution_model->>'evaluation_policy_version', '') = %s
                )
                SELECT COALESCE(family, strategy) AS family,
                       COALESCE(status, 'rejected') AS funnel_stage,
                       COUNT(*)::integer AS candidates
                FROM active
                WHERE rn = 1
                GROUP BY COALESCE(family, strategy), COALESCE(status, 'rejected')
                ORDER BY funnel_stage, candidates DESC, family
                """,
                (str(evaluation_policy_version),),
            )
            return [dict(row) for row in await cur.fetchall()]
