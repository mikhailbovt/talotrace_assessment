"""Current-state observers; streams never own or cancel durable generation work."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import psycopg
from fastapi import HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

log = logging.getLogger(__name__)
POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0
READ_TIMEOUT_SECONDS = 10.0
TERMINAL_STATES = {"keyframes_completed", "video_completed", "failed", "interrupted"}


class JobRoute(APIRoute):
    """Return a safe retryable response if the durable store cannot be reached."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request):
            try:
                return await handler(request)
            except psycopg.Error as error:
                raise HTTPException(
                    503, "Job storage is temporarily unavailable", headers={"Retry-After": "2"}
                ) from error

        return handle


def lifecycle(job: dict) -> dict:
    return {
        "terminal": job["state"] in TERMINAL_STATES,
        "can_resume": job["state"] in {"failed", "interrupted"} and job["attempts"] < 3,
    }


def accepted(value: dict, response: Response) -> dict:
    response.headers["Location"] = value["status_url"]
    response.headers["Retry-After"] = "2"
    return value


async def configured(call: Callable, *args):
    """Reference hashing and local manifest reads must not block status requests."""
    try:
        return await asyncio.to_thread(call, *args)
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Pipeline configuration is unavailable or invalid") from error


async def verified_artifact(artifacts, job_id: str, filename: str, expected_sha: str) -> Path:
    def validate():
        path = artifacts.path(job_id, filename)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected_sha:
            raise ValueError("Checksum mismatch")
        return path

    try:
        return await asyncio.to_thread(validate)
    except (OSError, ValueError) as error:
        raise HTTPException(409, "Artifact is missing or failed integrity validation") from error


def snapshot_stream(initial: dict, load: Callable[[], Awaitable[dict]]) -> StreamingResponse:
    """Emit the latest snapshot and changes, not an invented historical event log.

    Each read completes before sleeping. There are no retained DB connections,
    producer tasks or generation handles. StreamingResponse cancels this observer
    on disconnect; CancelledError deliberately propagates through the generator.
    """

    def encode(value: dict) -> str:
        return json.dumps(jsonable_encoder(value), separators=(",", ":"), ensure_ascii=False)

    async def events():
        snapshot = initial
        previous = None
        last_sent = asyncio.get_running_loop().time()
        first = True
        while True:
            encoded = encode(snapshot)
            if encoded != previous:
                retry = "retry: 2000\n" if first else ""
                yield f"{retry}event: status\ndata: {encoded}\n\n"
                previous = encoded
                first = False
                last_sent = asyncio.get_running_loop().time()
            if snapshot["state"] in TERMINAL_STATES:
                event = snapshot["state"]
                if event in {"keyframes_completed", "video_completed"}:
                    event = "completed"
                yield f"event: {event}\ndata: {encoded}\n\n"
                return
            await asyncio.sleep(POLL_SECONDS)
            try:
                async with asyncio.timeout(READ_TIMEOUT_SECONDS):
                    snapshot = await load()
            except Exception as error:
                # Once SSE headers are sent, report observer failure inside the stream.
                # Do not serialize exception messages (they can contain connection secrets).
                missing = isinstance(error, HTTPException) and error.status_code == 404
                log.warning("Job status stream read failed (%s)", type(error).__name__)
                details = {
                    "code": "job_not_found" if missing else "status_unavailable",
                    "detail": "The status stream closed; poll or reconnect for the latest state",
                    "retryable": not missing,
                    "generation_cancelled": False,
                }
                yield f"event: stream_error\ndata: {encode(details)}\n\n"
                return
            if (
                encode(snapshot) == previous
                and asyncio.get_running_loop().time() - last_sent >= HEARTBEAT_SECONDS
            ):
                yield ": heartbeat\n\n"
                last_sent = asyncio.get_running_loop().time()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Event-Semantics": "current-snapshot",
        },
    )
