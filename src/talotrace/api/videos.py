"""Question-to-narrated-video API; keyframe-only routes remain independently usable."""

import asyncio
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import Field, field_validator, model_validator

from talotrace.api.keyframes import summary as keyframe_summary
from talotrace.api.status import (
    JobRoute,
    accepted,
    configured,
    lifecycle,
    snapshot_stream,
    verified_artifact,
)
from talotrace.artifacts.store import json_bytes, sha256
from talotrace.jobs.narration_review import effective_direction, selected_review
from talotrace.jobs.preconditions import CorpusNotReady, require_corrected_corpus
from talotrace.jobs.schemas import KeyframeRequest, StrictModel
from talotrace.jobs.store import IdempotencyConflict
from talotrace.jobs.video_worker import video_settings
from talotrace.jobs.worker import pipeline_settings

router = APIRouter(prefix="/v1/video-jobs", tags=["question-to-video"], route_class=JobRoute)

STAGE_LABELS = {
    "queued": "Waiting for video generation",
    "waiting_for_keyframes": "Waiting for the script and keyframes",
    "synthesizing": "Speaking the script with Kokoro",
    "rendering": "Rendering the whiteboard video",
    "assembling": "Combining video with the complete narration",
    "video_completed": "Narrated video ready",
    "failed": "Video generation failed",
    "interrupted": "Video generation was interrupted",
}


class VideoRequest(StrictModel):
    question: str | None = Field(default=None, min_length=5, max_length=2000)
    keyframe_job_id: UUID | None = None
    keyframe_count: int = Field(default=5, ge=5, le=10)
    mode: Literal["whiteboard_loop"] = Field(
        default="whiteboard_loop",
        description="Blank board, visible hand drawing, completed keyframe, erasing, then repeat.",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def reject_retired_mode(cls, value):
        if value == "references":
            raise ValueError(
                "Reference generation is retired for new requests; use whiteboard_loop"
            )
        return value

    @model_validator(mode="after")
    def input_choice(self):
        if (self.question is None) == (self.keyframe_job_id is None):
            raise ValueError("Supply a chemistry question or an existing keyframe_job_id")
        if self.question is not None:
            self.question = KeyframeRequest(question=self.question).question
        return self


def summary(job: dict) -> dict:
    job_id = str(job["id"])
    job_lifecycle = lifecycle(job)
    if job["request"]["mode"] != "whiteboard_loop":
        job_lifecycle["can_resume"] = False
    # Old persisted jobs predate the explicit renderer field and used LTX.
    renderer = job.get("settings", {}).get("renderer", {"name": "ltx"})
    stage_label = STAGE_LABELS.get(job["state"], job["state"])
    if job["state"] == "rendering":
        if renderer["name"] == "keyframe-composite-v1":
            stage_label = "Animating the keyframes and supplied hand locally"
        elif renderer["name"] == "ltx":
            stage_label = "Rendering the whiteboard video with LTX"
    return {
        "id": job_id,
        "keyframe_job_id": str(job["keyframe_job_id"]),
        "question": job["request"]["question"],
        "mode": job["request"]["mode"],
        "renderer": renderer,
        "state": job["state"],
        **job_lifecycle,
        "stage": {"code": job["state"], "label": stage_label},
        "progress": job["progress"],
        "attempts": job["attempts"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "video_generated": job["state"] == "video_completed",
        "status_url": f"/v1/video-jobs/{job_id}",
        "events_url": f"/v1/video-jobs/{job_id}/events",
        "result_url": f"/v1/video-jobs/{job_id}/result",
        "video_url": f"/v1/video-jobs/{job_id}/video" if job["result"] else None,
        "keyframes_status_url": f"/v1/keyframe-jobs/{job['keyframe_job_id']}",
        "keyframes_events_url": f"/v1/keyframe-jobs/{job['keyframe_job_id']}/events",
    }


async def require_job(request: Request, job_id: UUID) -> dict:
    job = await request.app.state.video_store.get(str(job_id))
    if job is None:
        raise HTTPException(404, "Video job not found")
    return job


@router.post("", status_code=202)
async def submit(
    body: VideoRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(min_length=8, max_length=128)],
):
    try:
        require_corrected_corpus(await request.app.state.job_store.active_corpus_summary())
    except CorpusNotReady as error:
        raise HTTPException(
            503, "Verified PDF-inspector plus Luna-vision corpus is not active"
        ) from error
    settings = request.app.state.settings
    upstream_settings = await configured(pipeline_settings, settings)
    upstream = None
    if body.keyframe_job_id:
        upstream = await request.app.state.job_store.get(str(body.keyframe_job_id))
        if upstream is None:
            raise HTTPException(404, "Upstream keyframe job not found")
        if upstream["settings"] != upstream_settings:
            raise HTTPException(409, "Upstream keyframe settings differ from this pipeline")
    review = None
    if upstream:
        try:
            review = await asyncio.to_thread(
                selected_review, request.app.state.artifacts, str(upstream["id"])
            )
            if review:
                effective_direction(upstream["direction"], str(upstream["id"]), review)
        except (OSError, ValueError) as error:
            raise HTTPException(
                409, "Narration review is unapproved or does not match the source"
            ) from error
    identity = (await configured(video_settings, settings)) | {
        "keyframe_settings": upstream_settings,
        "narration_review": review,
    }
    value = body.model_dump(mode="json")
    if upstream:
        value["question"] = upstream["request"]["question"]
        value["keyframe_count"] = upstream["request"]["keyframe_count"]
    digest = sha256(json_bytes({"request": value, "settings": identity}))
    try:
        job = await request.app.state.video_store.submit_video(
            idempotency_key, digest, value, identity, upstream, upstream_settings
        )
    except IdempotencyConflict as error:
        raise HTTPException(409, "Idempotency key was used for different video inputs") from error
    return accepted(summary(job), response)


@router.get("")
async def list_jobs(
    request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    return {
        "jobs": [summary(job) for job in await request.app.state.video_store.list(limit, offset)]
    }


@router.get("/{job_id}")
async def status(job_id: UUID, request: Request):
    job = await require_job(request, job_id)
    upstream = await request.app.state.job_store.get(str(job["keyframe_job_id"]))
    value = summary(job)
    worker_enabled = request.app.state.settings.video_worker_enabled
    upstream_value = keyframe_summary(upstream) if upstream else None
    blocking_reason = None
    effective_stage = value["stage"]
    if job["state"] in {"queued", "waiting_for_keyframes"}:
        if upstream is None:
            blocking_reason = "upstream_missing"
        elif upstream["state"] in {"failed", "interrupted"}:
            blocking_reason = "upstream_requires_resume"
            effective_stage = upstream_value["stage"]
        elif upstream["state"] != "keyframes_completed":
            effective_stage = upstream_value["stage"]
        elif not worker_enabled:
            blocking_reason = "video_worker_disabled"
            effective_stage = {"code": "video_worker_disabled", "label": "Video worker is disabled"}
    if value["can_resume"] and (upstream is None or upstream["state"] in {"failed", "interrupted"}):
        value["can_resume"] = False
        blocking_reason = "upstream_missing" if upstream is None else "upstream_requires_resume"
    return value | {
        "upstream_state": upstream["state"] if upstream else None,
        "upstream": upstream_value,
        "completed_keyframes": len(upstream["frames"]) if upstream else 0,
        "requested_keyframes": job["request"]["keyframe_count"],
        "video_worker_enabled": worker_enabled,
        "effective_stage": effective_stage,
        "blocking_reason": blocking_reason,
    }


@router.get(
    "/{job_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def events(job_id: UUID, request: Request):
    """Observe current video and upstream progress; disconnecting leaves generation running."""
    initial = await status(job_id, request)

    async def load():
        return await status(job_id, request)

    return snapshot_stream(initial, load)


@router.get("/{job_id}/result")
async def result(job_id: UUID, request: Request):
    job = await require_job(request, job_id)
    return summary(job) | {
        "timeline": job["timeline"],
        "result": job["result"],
        "visual_review": "not_automatically_certified",
    }


@router.post("/{job_id}/resume", status_code=202)
async def resume(job_id: UUID, request: Request, response: Response):
    job = await require_job(request, job_id)
    if job["state"] == "video_completed":
        return accepted(summary(job), response)
    if job["request"]["mode"] != "whiteboard_loop":
        raise HTTPException(
            409, "Reference generation is retired; create a new whiteboard_loop video job"
        )
    upstream = await request.app.state.job_store.get(str(job["keyframe_job_id"]))
    if upstream is None:
        raise HTTPException(409, "Upstream keyframe job is unavailable")
    if upstream["state"] in {"failed", "interrupted"}:
        raise HTTPException(409, "Resume the upstream keyframe job first")
    resumed = await request.app.state.video_store.resume(str(job_id))
    if resumed is None:
        raise HTTPException(
            409, "Only failed/interrupted video jobs below three attempts can resume"
        )
    return accepted(summary(resumed), response)


@router.get("/{job_id}/video")
async def video(job_id: UUID, request: Request):
    job = await require_job(request, job_id)
    if job["state"] != "video_completed" or not job["result"]:
        raise HTTPException(409, "Final video is not available yet")
    path = await verified_artifact(
        request.app.state.artifacts, str(job_id), "final.mp4", job["result"]["media"]["sha256"]
    )
    return FileResponse(path, media_type="video/mp4", filename=f"chemistry-{job_id}.mp4")
