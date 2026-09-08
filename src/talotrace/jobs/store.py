"""PostgreSQL is the durable queue; provider work never holds a DB transaction open."""

from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class IdempotencyConflict(ValueError):
    pass


class JobStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    async def connect(self):
        return await psycopg.AsyncConnection.connect(
            self.dsn, autocommit=True, row_factory=dict_row, connect_timeout=5
        )

    async def migrate(self) -> None:
        migration = Path("infra/postgres/004-jobs.sql")
        async with await self.connect() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(472930401)")
                await conn.execute(migration.read_text("utf-8"))

    async def submit(self, key: str, digest: str, request: dict, settings: dict) -> dict:
        async with await self.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """INSERT INTO pipeline.jobs
                    (id,idempotency_key,request_sha256,request,settings) VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (idempotency_key) DO NOTHING""",
                    (uuid4(), key, digest, Jsonb(request), Jsonb(settings)),
                )
                row = await (
                    await conn.execute(
                        "SELECT * FROM pipeline.jobs WHERE idempotency_key=%s", (key,)
                    )
                ).fetchone()
                if row["request_sha256"] != digest:
                    raise IdempotencyConflict(
                        "Idempotency key was already used for another request"
                    )
                return row

    async def get(self, job_id: str) -> dict | None:
        async with await self.connect() as conn:
            return await (
                await conn.execute("SELECT * FROM pipeline.jobs WHERE id=%s", (job_id,))
            ).fetchone()

    async def active_corpus_summary(self) -> dict | None:
        async with await self.connect() as conn:
            row = await (
                await conn.execute("SELECT summary FROM rag.corpora WHERE active AND state='ready'")
            ).fetchone()
            return row["summary"] if row else None

    async def list(self, limit: int, offset: int) -> list[dict]:
        async with await self.connect() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM pipeline.jobs ORDER BY created_at DESC,id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            ).fetchall()

    async def update(self, job_id: str, **values) -> None:
        allowed = {"state", "context", "direction", "frames", "error"}
        if not values or set(values) - allowed:
            raise ValueError("Unsupported job update")
        assignments = psycopg.sql.SQL(",").join(
            psycopg.sql.SQL("{}=%s").format(psycopg.sql.Identifier(name)) for name in values
        )
        params = [
            Jsonb(value) if name != "state" and value is not None else value
            for name, value in values.items()
        ]
        async with await self.connect() as conn:
            await conn.execute(
                psycopg.sql.SQL("UPDATE pipeline.jobs SET {},updated_at=now() WHERE id=%s").format(
                    assignments
                ),
                (*params, job_id),
            )

    async def resume(self, job_id: str) -> dict | None:
        async with await self.connect() as conn:
            return await (
                await conn.execute(
                    """UPDATE pipeline.jobs SET state='queued',error=NULL,updated_at=now()
                WHERE id=%s AND state IN ('failed','interrupted') AND attempts<3 RETURNING *""",
                    (job_id,),
                )
            ).fetchone()

    async def claim(self, conn) -> dict | None:
        async with conn.transaction():
            return await (
                await conn.execute(
                    """UPDATE pipeline.jobs
                SET state='retrieving',attempts=attempts+1,updated_at=now()
                WHERE id=(SELECT id FROM pipeline.jobs WHERE state='queued' AND attempts<3
                  ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *"""
                )
            ).fetchone()

    async def mark_interrupted(self, conn) -> None:
        await conn.execute(
            """UPDATE pipeline.jobs SET state='interrupted',updated_at=now(),
            error=%s || jsonb_build_object('failed_stage',state)
            WHERE state IN ('retrieving','directing','generating_keyframes')""",
            (
                Jsonb(
                    {
                        "category": "WorkerInterrupted",
                        "detail": "Saved outputs are retained; unfinished calls may bill on resume",
                    }
                ),
            ),
        )
