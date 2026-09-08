"""Async chemistry keyframes and whiteboard videos with durable, separate workers."""

import asyncio
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from talotrace.api.keyframes import router as keyframe_router
from talotrace.api.videos import router as video_router
from talotrace.artifacts.store import ArtifactStore
from talotrace.config import Settings
from talotrace.jobs.store import JobStore
from talotrace.jobs.video_store import VideoJobStore
from talotrace.jobs.video_worker import VideoPipeline, VideoWorker
from talotrace.jobs.worker import JobWorker, KeyframePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.artifacts = ArtifactStore(app.state.settings.artifacts_dir)
    app.state.job_store = JobStore(app.state.settings.database_url.get_secret_value())
    await app.state.job_store.migrate()
    app.state.video_store = VideoJobStore(app.state.settings.database_url.get_secret_value())
    await app.state.video_store.migrate()
    worker = JobWorker(
        KeyframePipeline(
            app.state.settings,
            app.state.job_store,
            app.state.artifacts,
        )
    )
    app.state.worker = worker
    video_worker = VideoWorker(
        VideoPipeline(
            app.state.settings, app.state.video_store, app.state.job_store, app.state.artifacts
        )
    )
    app.state.video_worker = video_worker
    worker.start()
    if app.state.settings.video_worker_enabled:
        video_worker.start()
    try:
        yield
    finally:
        await video_worker.stop()
        await worker.stop()


app = FastAPI(
    title="Talotrace Chemistry Video Service",
    version="0.1.0",
    description="Grounded chemistry keyframes and narrated whiteboard drawing/erasing videos.",
    lifespan=lifespan,
)
app.include_router(keyframe_router)
app.include_router(video_router)


@app.get("/health/live", tags=["infrastructure"])
async def live():
    return {
        "status": "ok",
        "keyframe_pipeline_implemented": True,
        "video_pipeline_implemented": True,
        "video_worker_enabled": app.state.settings.video_worker_enabled,
        "video_renderer": app.state.settings.video_renderer,
    }


@app.get("/health/ready", tags=["infrastructure"])
async def ready():
    """Control-plane dependencies only; does not claim model or pipeline readiness."""
    settings = app.state.settings

    async def check_postgres():
        try:
            async with await psycopg.AsyncConnection.connect(
                settings.database_url.get_secret_value(), connect_timeout=3
            ) as connection:
                await connection.execute("SELECT 1")
            return True
        except (psycopg.Error, OSError, ValueError):
            return False

    async def check_redis():
        try:
            async with Redis.from_url(
                settings.redis_url.get_secret_value(),
                socket_connect_timeout=3,
                socket_timeout=3,
            ) as client:
                return bool(await client.ping())
        except Exception:
            return False

    postgres_ok, redis_ok = await asyncio.gather(check_postgres(), check_redis())
    ok = postgres_ok and redis_ok
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "scope": "control_plane_dependencies",
            "postgres": postgres_ok,
            "redis": redis_ok,
            "keyframe_pipeline_implemented": True,
            "video_pipeline_implemented": True,
            "video_worker_enabled": settings.video_worker_enabled,
            "local_video_worker_active": app.state.video_worker.active,
            "local_job_worker_active": app.state.worker.active,
        },
    )
