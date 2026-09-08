"""Durable question-to-keyframes requests, polling, and live status snapshots."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from talotrace.api.status import (
    JobRoute,
    accepted,
    configured,
    lifecycle,
    snapshot_stream,
    verified_artifact,
)
from talotrace.artifacts.store import json_bytes, sha256
from talotrace.jobs.preconditions import CorpusNotReady, require_corrected_corpus
from talotrace.jobs.schemas import KeyframeRequest
from talotrace.jobs.store import IdempotencyConflict
from talotrace.jobs.worker import pipeline_settings

router = APIRouter(prefix="/v1/keyframe-jobs", tags=["question-to-keyframes"], route_class=JobRoute)

STAGE_LABELS = {
    "queued": "Waiting for the keyframe worker",
    "retrieving": "Embedding the question and retrieving chemistry sources",
    "directing": "Writing the grounded script and scene directions",
    "generating_keyframes": "Generating scene keyframes",
    "keyframes_completed": "Keyframes ready",
    "failed": "Keyframe generation failed",
    "interrupted": "Keyframe generation was interrupted",
}


def summary(job: dict) -> dict:
    job_id = str(job["id"])
    return {
        "id": job_id,
        "question": job["request"]["question"],
        "state": job["state"],
        **lifecycle(job),
        "stage": {"code": job["state"], "label": STAGE_LABELS.get(job["state"], job["state"])},
        "progress": {
            "unit": "keyframes",
            "completed": len(job["frames"]),
            "total": job["request"]["keyframe_count"],
        },
        "completed_keyframes": len(job["frames"]),
        "requested_keyframes": job["request"]["keyframe_count"],
        "attempts": job["attempts"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job["error"],
        "video_generated": False,
        "timing_basis": "estimated_until_tts",
        "status_url": f"/v1/keyframe-jobs/{job_id}",
        "events_url": f"/v1/keyframe-jobs/{job_id}/events",
        "result_url": f"/v1/keyframe-jobs/{job_id}/result",
    }


async def require_job(request: Request, job_id: UUID) -> dict:
    job = await request.app.state.job_store.get(str(job_id))
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.post("", status_code=202)
async def submit(
    body: KeyframeRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(min_length=8, max_length=128)],
):
    try:
        require_corrected_corpus(await request.app.state.job_store.active_corpus_summary())
    except CorpusNotReady as error:
        raise HTTPException(
            503, "The verified PDF-inspector plus Luna-vision corpus is not active"
        ) from error
    settings = await configured(pipeline_settings, request.app.state.settings)
    digest = sha256(json_bytes({"request": body.model_dump(), "settings": settings}))
    try:
        job = await request.app.state.job_store.submit(
            idempotency_key,
            digest,
            body.model_dump(),
            settings,
        )
    except IdempotencyConflict as error:
        raise HTTPException(409, "Idempotency key was used for different inputs") from error
    return accepted(summary(job), response)


@router.get("")
async def list_jobs(
    request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    return {"jobs": [summary(job) for job in await request.app.state.job_store.list(limit, offset)]}


@router.get("/{job_id}")
async def status(job_id: UUID, request: Request):
    return summary(await require_job(request, job_id))


@router.get(
    "/{job_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def events(job_id: UUID, request: Request):
    """Current snapshot, changed snapshots, heartbeats, then a terminal event; no replay."""
    initial = await status(job_id, request)  # Validate before sending streaming HTTP200 headers.

    async def load():
        return await status(job_id, request)

    return snapshot_stream(initial, load)


@router.get("/{job_id}/result")
async def result(job_id: UUID, request: Request):
    job = await require_job(request, job_id)
    return summary(job) | {
        "retrieval": job["context"],
        "direction": job["direction"],
        "keyframes": job["frames"],
        "visual_review": "not_automatically_certified",
    }


@router.post("/{job_id}/resume", status_code=202)
async def resume(job_id: UUID, request: Request, response: Response):
    job = await require_job(request, job_id)
    if job["state"] == "keyframes_completed":
        return accepted(summary(job), response)
    resumed = await request.app.state.job_store.resume(str(job_id))
    if resumed is None:
        raise HTTPException(409, "Only failed/interrupted jobs below three attempts can resume")
    return accepted(summary(resumed), response)


@router.get("/{job_id}/images/{order}")
async def image(job_id: UUID, order: Annotated[int, Path(ge=1, le=10)], request: Request):
    job = await require_job(request, job_id)
    receipt = next((frame for frame in job["frames"] if frame["order"] == order), None)
    if receipt is None:
        raise HTTPException(404, "Keyframe not available yet")
    path = await verified_artifact(
        request.app.state.artifacts,
        str(job_id),
        f"scene-{order:02d}.png",
        receipt["image"]["sha256"],
    )
    return FileResponse(path, media_type="image/png", filename=f"scene-{order:02d}.png")
