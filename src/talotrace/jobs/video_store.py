"""Video queue and upstream keyframe creation share a transaction and idempotency identity."""

from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from talotrace.jobs.store import IdempotencyConflict, JobStore


class VideoJobStore(JobStore):
    async def migrate(self):
        async with await self.connect() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(472930403)")
                await conn.execute(Path("infra/postgres/005-video-jobs.sql").read_text("utf-8"))

    async def submit_video(
        self,
        key: str,
        digest: str,
        body: dict,
        settings: dict,
        upstream: dict | None,
        upstream_settings: dict,
    ) -> dict:
        async with await self.connect() as conn:
            async with conn.transaction():
                # A per-key transaction lock prevents concurrent duplicate upstream creation.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (f"video-submit:{key}",)
                )
                saved = await (
                    await conn.execute(
                        "SELECT * FROM pipeline.video_jobs WHERE idempotency_key=%s", (key,)
                    )
                ).fetchone()
                if saved:
                    if saved["request_sha256"] != digest:
                        raise IdempotencyConflict("Video idempotency input changed")
                    return saved
                if upstream is None:
                    from talotrace.artifacts.store import json_bytes, sha256

                    request = {
                        "question": body["question"],
                        "keyframe_count": body["keyframe_count"],
                    }
                    upstream_id = uuid4()
                    await conn.execute(
                        """INSERT INTO pipeline.jobs
                        (id,idempotency_key,request_sha256,request,settings)
                        VALUES (%s,%s,%s,%s,%s)""",
                        (
                            upstream_id,
                            f"video:{uuid4()}",
                            sha256(json_bytes({"request": request, "settings": upstream_settings})),
                            Jsonb(request),
                            Jsonb(upstream_settings),
                        ),
                    )
                else:
                    upstream_id = upstream["id"]
                return await (
                    await conn.execute(
                        """INSERT INTO pipeline.video_jobs
                    (id,keyframe_job_id,idempotency_key,request_sha256,request,settings)
                    VALUES (%s,%s,%s,%s,%s,%s) RETURNING *""",
                        (uuid4(), upstream_id, key, digest, Jsonb(body), Jsonb(settings)),
                    )
                ).fetchone()

    async def get(self, job_id: str) -> dict | None:
        async with await self.connect() as conn:
            return await (
                await conn.execute("SELECT * FROM pipeline.video_jobs WHERE id=%s", (job_id,))
            ).fetchone()

    async def list(self, limit: int, offset: int) -> list[dict]:
        async with await self.connect() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM pipeline.video_jobs "
                    "ORDER BY created_at DESC,id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            ).fetchall()

    async def update(self, job_id: str, **values):
        if not values or set(values) - {"state", "timeline", "progress", "result", "error"}:
            raise ValueError("Unsupported video job update")
        assignments = psycopg.sql.SQL(",").join(
            psycopg.sql.SQL("{}=%s").format(psycopg.sql.Identifier(name)) for name in values
        )
        params = [
            Jsonb(value) if key != "state" and value is not None else value
            for key, value in values.items()
        ]
        async with await self.connect() as conn:
            await conn.execute(
                psycopg.sql.SQL(
                    "UPDATE pipeline.video_jobs SET {},updated_at=now() WHERE id=%s"
                ).format(assignments),
                (*params, job_id),
            )

    async def claim(self, conn):
        async with conn.transaction():
            return await (
                await conn.execute(
                    """UPDATE pipeline.video_jobs SET state='synthesizing',attempts=attempts+1,
                updated_at=now() WHERE id=(
                  SELECT v.id FROM pipeline.video_jobs v JOIN pipeline.jobs k
                    ON k.id=v.keyframe_job_id
                  WHERE v.state IN ('queued','waiting_for_keyframes') AND v.attempts<3
                    AND k.state IN ('keyframes_completed','failed','interrupted')
                  ORDER BY v.created_at,v.id FOR UPDATE OF v SKIP LOCKED LIMIT 1
                ) RETURNING *"""
                )
            ).fetchone()

    async def resume(self, job_id: str) -> dict | None:
        async with await self.connect() as conn:
            return await (
                await conn.execute(
                    """UPDATE pipeline.video_jobs SET state='queued',error=NULL,updated_at=now()
                WHERE id=%s AND state IN ('failed','interrupted') AND attempts<3 RETURNING *""",
                    (job_id,),
                )
            ).fetchone()

    async def mark_interrupted(self, conn):
        await conn.execute(
            """UPDATE pipeline.video_jobs SET state='interrupted',updated_at=now(),
            error=jsonb_build_object('category','WorkerInterrupted','failed_stage',state)
            WHERE state IN ('synthesizing','rendering','assembling')"""
        )
