"""Real HTTP/SSE transport, with isolated stores and no paid provider requests."""

import asyncio
import copy
import importlib.util
import json
import socket
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import psycopg
import pytest
import uvicorn
from fastapi import FastAPI

from talotrace.api import keyframes, videos
from talotrace.artifacts.store import ArtifactStore, sha256
from talotrace.config import Settings
from talotrace.jobs.preconditions import EXTRACTION_POLICY
from talotrace.jobs.store import IdempotencyConflict


class MemoryStore:
    """Only the persistence boundary is substituted; requests use real HTTP sockets."""

    def __init__(self):
        self.rows = {}
        self.keys = {}
        self.reads = 0
        self.read_error = None
        self.read_gate = None
        self.read_started = asyncio.Event()
        self.cancelled_reads = 0
        self.submit_gate = None
        self.persisted = asyncio.Event()

    async def active_corpus_summary(self):
        return {
            "extraction_policy": EXTRACTION_POLICY,
            "extraction_verified": True,
            "vision_required_pages": 1,
            "vision_completed_pages": 1,
        }

    async def submit(self, key, digest, body, settings):
        if key in self.keys:
            saved_digest, job_id = self.keys[key]
            if saved_digest != digest:
                raise IdempotencyConflict()
            return copy.deepcopy(self.rows[job_id])
        job_id = str(uuid4())
        value = {
            "id": job_id,
            "request": body,
            "settings": settings,
            "state": "queued",
            "frames": [],
            "context": None,
            "direction": None,
            "error": None,
            "attempts": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        self.rows[job_id] = value
        self.keys[key] = (digest, job_id)
        self.persisted.set()
        if self.submit_gate:
            await self.submit_gate.wait()
        return copy.deepcopy(value)

    async def get(self, job_id):
        self.reads += 1
        if self.read_error:
            raise self.read_error
        if self.read_gate:
            self.read_started.set()
            try:
                await self.read_gate.wait()
            except asyncio.CancelledError:
                self.cancelled_reads += 1
                raise
        return copy.deepcopy(self.rows.get(job_id))

    async def list(self, limit, offset):
        return copy.deepcopy(list(self.rows.values())[offset : offset + limit])

    async def resume(self, job_id):
        job = self.rows[job_id]
        if job["state"] not in {"failed", "interrupted"} or job["attempts"] >= 3:
            return None
        job.update(state="queued", error=None)
        return copy.deepcopy(job)

    def change(self, job_id, **values):
        self.rows[job_id].update(values, updated_at=datetime.now(UTC))


class VideoStore(MemoryStore):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream

    async def submit_video(self, key, digest, body, settings, upstream, upstream_settings):
        if upstream is None:
            upstream = await self.upstream.submit(
                "upstream:" + key,
                digest,
                {"question": body["question"], "keyframe_count": body["keyframe_count"]},
                upstream_settings,
            )
        result = await self.submit(key, digest, body, settings)
        job = self.rows[result["id"]]
        job.setdefault("keyframe_job_id", upstream["id"])
        job.setdefault("progress", {})
        job.setdefault("timeline", None)
        job.setdefault("result", None)
        return copy.deepcopy(job)


@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(keyframes.router)
    app.include_router(videos.router)
    app.state.settings = Settings(
        openai_api_key="test-unused", video_worker_enabled=False, video_renderer="ltx"
    )
    app.state.artifacts = ArtifactStore(Path("data/tests/unit-api") / str(uuid4()))
    app.state.job_store = MemoryStore()
    app.state.video_store = VideoStore(app.state.job_store)
    monkeypatch.setattr("talotrace.api.status.POLL_SECONDS", 0.02)
    monkeypatch.setattr("talotrace.api.status.HEARTBEAT_SECONDS", 0.08)
    return app


@asynccontextmanager
async def serve(app):
    # Ephemeral listener cannot collide with or restart the assessment API on port8000.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            yield client
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        listener.close()


async def next_event(lines):
    name, data = None, None
    async with asyncio.timeout(2):
        async for line in lines:
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif not line and name:
                return name, data
    raise AssertionError("Stream ended without the expected event")


async def submit(client, kind="keyframe", key="async-test-key"):
    body = {"question": "How does the pH scale work?", "keyframe_count": 5}
    return await client.post(f"/v1/{kind}-jobs", headers={"Idempotency-Key": key}, json=body)


@pytest.mark.parametrize("kind", ["keyframe", "video"])
async def test_http202_idempotency_and_polling_do_not_wait_for_generation(api, kind):
    async with serve(api) as client:
        # No worker/provider is running; acknowledgement must finish despite that.
        response = await asyncio.wait_for(submit(client, kind), 1)
        assert response.status_code == 202
        data = response.json()
        assert response.headers["location"] == data["status_url"]
        assert response.headers["retry-after"] == "2"
        assert data["state"] == "queued" and not data["terminal"]
        reads = await asyncio.gather(*(client.get(data["status_url"]) for _ in range(5)))
        assert all(item.status_code == 200 for item in reads)
        assert (await submit(client, kind)).json()["id"] == data["id"]
        body = {"question": "Explain covalent bonding", "keyframe_count": 5}
        if kind == "video":
            body["mode"] = "whiteboard_loop"
        changed = await client.post(
            f"/v1/{kind}-jobs", headers={"Idempotency-Key": "async-test-key"}, json=body
        )
        assert changed.status_code == 409
        assert len(api.state.job_store.rows) == 1
        if kind == "video":
            assert len(api.state.video_store.rows) == 1
            assert data["mode"] == "whiteboard_loop"


async def test_new_video_defaults_to_whiteboard_and_rejects_references_without_persisting(api):
    async with serve(api) as client:
        rejected = await client.post(
            "/v1/video-jobs",
            headers={"Idempotency-Key": "retired-mode-test"},
            json={"question": "How does the pH scale work?", "mode": "references"},
        )
        assert rejected.status_code == 422
        assert "retired for new requests; use whiteboard_loop" in rejected.text
        assert not api.state.video_store.rows and not api.state.job_store.rows
        default = (await submit(client, "video")).json()
        assert default["mode"] == "whiteboard_loop"
        explicit = await client.post(
            "/v1/video-jobs",
            headers={"Idempotency-Key": "async-test-key"},
            json={
                "question": "How does the pH scale work?",
                "keyframe_count": 5,
                "mode": "whiteboard_loop",
            },
        )
        assert explicit.status_code == 202 and explicit.json()["id"] == default["id"]
        schema = (await client.get("/openapi.json")).json()
        mode = schema["components"]["schemas"]["VideoRequest"]["properties"]["mode"]
        assert mode["const"] == "whiteboard_loop" and mode["default"] == "whiteboard_loop"


@pytest.mark.parametrize("renderer", ["ltx", "keyframe-composite-v1"])
async def test_renderer_provenance_and_rendering_label_follow_persisted_job(api, renderer):
    async with serve(api) as client:
        job = (await submit(client, "video")).json()
        api.state.video_store.rows[job["id"]]["settings"]["renderer"] = {"name": renderer}
        api.state.video_store.change(job["id"], state="rendering")
        status = (await client.get(job["status_url"])).json()
        assert status["renderer"]["name"] == renderer
        assert ("LTX" in status["stage"]["label"]) is (renderer == "ltx")
        if renderer == "keyframe-composite-v1":
            assert "locally" in status["stage"]["label"]


async def test_health_exposes_only_safe_renderer_and_worker_configuration(monkeypatch):
    from talotrace.main import app

    settings = Settings(
        openai_api_key="private-test-value",
        video_worker_enabled=True,
        video_renderer="keyframe-composite-v1",
    )
    monkeypatch.setattr(app.state, "settings", settings, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health/live")
    assert response.json()["video_renderer"] == "keyframe-composite-v1"
    assert response.json()["video_worker_enabled"] is True
    assert "private-test-value" not in response.text


@pytest.mark.parametrize(
    "deployed",
    [
        {"video_renderer": "ltx", "video_worker_enabled": True},
        {"video_renderer": "keyframe-composite-v1", "video_worker_enabled": False},
    ],
)
async def test_acceptance_client_refuses_wrong_renderer_or_disabled_worker_before_post(
    monkeypatch, deployed
):
    specification = importlib.util.spec_from_file_location(
        "smoke_videos", "scripts/smoke_videos.py"
    )
    smoke = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(smoke)
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json=deployed)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://test")
    monkeypatch.setattr(smoke.httpx, "AsyncClient", lambda **kwargs: client)
    args = SimpleNamespace(
        prefix=f"unit-api/acceptance-{uuid4()}",
        mode="whiteboard_loop",
        api_url="http://test",
        expected_renderer="keyframe-composite-v1",
    )
    with pytest.raises(RuntimeError, match="no job submitted"):
        await smoke.main(args)
    assert calls == [("GET", "/health/live")]


async def test_historical_reference_job_remains_listable_observable_and_downloadable(api):
    async with serve(api) as client:
        job = (await submit(client, "video")).json()
        stored = api.state.video_store.rows[job["id"]]
        stored["request"]["mode"] = "references"  # Simulated record created by the previous API.
        content = b"historical transport fixture; not a generated video"
        api.state.artifacts.write(job["id"], "final.mp4", content)
        api.state.video_store.change(
            job["id"], state="video_completed", result={"media": {"sha256": sha256(content)}}
        )
        status = (await client.get(job["status_url"])).json()
        assert status["mode"] == "references" and status["video_generated"]
        assert (await client.get("/v1/video-jobs")).json()["jobs"][0]["mode"] == "references"
        result = (await client.get(job["result_url"])).json()
        assert result["mode"] == "references" and result["result"]["media"]["sha256"] == sha256(
            content
        )
        async with client.stream("GET", job["events_url"]) as response:
            lines = response.aiter_lines()
            assert (await next_event(lines))[1]["mode"] == "references"
            assert (await next_event(lines))[0] == "completed"
        downloaded = await client.get(status["video_url"])
        assert downloaded.status_code == 200 and downloaded.content == content
        replay = await client.post(job["status_url"] + "/resume")
        assert replay.status_code == 202 and replay.json()["state"] == "video_completed"
        assert replay.json()["id"] == job["id"]
        assert api.state.video_store.rows[job["id"]]["attempts"] == 0


@pytest.mark.parametrize("state", ["queued", "rendering", "failed", "interrupted"])
async def test_unfinished_historical_reference_jobs_cannot_resume_generation(api, state):
    async with serve(api) as client:
        job = (await submit(client, "video")).json()
        stored = api.state.video_store.rows[job["id"]]
        stored["request"]["mode"] = "references"
        api.state.video_store.change(job["id"], state=state, attempts=1)
        before = copy.deepcopy(api.state.video_store.rows[job["id"]])
        assert (await client.get(job["status_url"])).json()["can_resume"] is False
        response = await client.post(job["status_url"] + "/resume")
        assert response.status_code == 409
        assert "Reference generation is retired" in response.text
        assert api.state.video_store.rows[job["id"]] == before


async def test_submitter_disconnect_does_not_remove_persisted_job(api):
    api.state.job_store.submit_gate = asyncio.Event()
    async with serve(api) as client:
        sending = asyncio.create_task(submit(client))
        await asyncio.wait_for(api.state.job_store.persisted.wait(), 1)
        sending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sending
        job_id = next(iter(api.state.job_store.rows))
        api.state.job_store.submit_gate.set()
        # The separately owned worker can still claim and complete this persisted job.
        api.state.job_store.change(job_id, state="keyframes_completed")
        response = await client.get(f"/v1/keyframe-jobs/{job_id}")
        assert response.status_code == 200 and response.json()["state"] == "keyframes_completed"
        assert (await submit(client)).json()["id"] == job_id


@pytest.mark.parametrize(
    "kind,final", [("keyframe", "keyframes_completed"), ("video", "video_completed")]
)
async def test_live_sse_flushes_changes_with_concurrent_polling_and_terminal_event(
    api, kind, final
):
    store = api.state.job_store if kind == "keyframe" else api.state.video_store
    async with serve(api) as client:
        job = (await submit(client, kind)).json()
        async with client.stream("GET", job["events_url"]) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["x-event-semantics"] == "current-snapshot"
            assert response.headers["x-accel-buffering"] == "no"
            lines = response.aiter_lines()
            name, initial = await next_event(lines)
            assert name == "status" and initial["state"] == "queued"
            assert (await client.get(job["status_url"])).status_code == 200
            if kind == "video":
                # Updating only the upstream must still push useful video status.
                api.state.job_store.change(
                    job["keyframe_job_id"], state="generating_keyframes", frames=[{"order": 1}]
                )
                _, update = await next_event(lines)
                assert update["effective_stage"]["code"] == "generating_keyframes"
                assert update["completed_keyframes"] == 1
                api.state.job_store.change(job["keyframe_job_id"], state="keyframes_completed")
                _, update = await next_event(lines)
                assert update["blocking_reason"] == "video_worker_disabled"
                store.change(job["id"], state="rendering", progress={"scene_order": 2})
            else:
                store.change(job["id"], state="generating_keyframes", frames=[{"order": 1}])
            _, update = await next_event(lines)
            assert update["state"] in {"rendering", "generating_keyframes"}
            store.change(job["id"], state=final)
            _, update = await next_event(lines)
            assert update["state"] == final and update["terminal"]
            name, update = await next_event(lines)
            assert name == "completed" and update["state"] == final
            assert [line async for line in lines] == []


@pytest.mark.parametrize("state", ["failed", "interrupted"])
async def test_terminal_sse_and_resume_semantics(api, state):
    async with serve(api) as client:
        job = (await submit(client)).json()
        api.state.job_store.change(
            job["id"],
            state=state,
            attempts=1,
            error={"category": "TimeoutError", "failed_stage": "directing"},
        )
        async with client.stream(
            "GET", job["events_url"], headers={"Last-Event-ID": "not-a-stored-cursor"}
        ) as response:
            lines = response.aiter_lines()
            name, initial = await next_event(lines)
            assert name == "status" and initial["can_resume"]
            name, terminal = await next_event(lines)
            assert name == state and terminal["error"]["failed_stage"] == "directing"
        resumed = await client.post(job["status_url"] + "/resume")
        assert resumed.status_code == 202 and resumed.json()["state"] == "queued"
        assert resumed.headers["location"] == job["status_url"]
        api.state.job_store.change(job["id"], state=state, attempts=3)
        assert (await client.post(job["status_url"] + "/resume")).status_code == 409


async def test_stream_heartbeat_disconnect_and_read_cancellation_leave_job_running(api):
    async with serve(api) as client:
        job = (await submit(client)).json()
        api.state.job_store.change(job["id"], state="generating_keyframes")
        async with client.stream("GET", job["events_url"]) as response:
            lines = response.aiter_lines()
            await next_event(lines)
            async with asyncio.timeout(1):
                async for line in lines:
                    assert not line.startswith("event: status")  # No duplicate snapshots.
                    if line == ": heartbeat":
                        break
            # A disconnect must cancel a pending observer read, not await it indefinitely.
            api.state.job_store.read_gate = asyncio.Event()
            await asyncio.wait_for(api.state.job_store.read_started.wait(), 1)
        async with asyncio.timeout(1):
            while not api.state.job_store.cancelled_reads:
                await asyncio.sleep(0.01)
        count = api.state.job_store.reads
        await asyncio.sleep(0.1)
        assert api.state.job_store.reads == count
        assert api.state.job_store.rows[job["id"]]["state"] == "generating_keyframes"
        api.state.job_store.read_gate = None
        api.state.job_store.change(job["id"], state="keyframes_completed")
        assert (await client.get(job["status_url"])).json()["state"] == "keyframes_completed"


async def test_invalid_missing_and_storage_error_streams_have_safe_http_errors(api):
    async with serve(api) as client:
        for path, code in [("invalid", 422), (str(uuid4()), 404)]:
            response = await client.get(f"/v1/keyframe-jobs/{path}/events")
            assert response.status_code == code
            assert "text/event-stream" not in response.headers["content-type"]
        job = (await submit(client)).json()
        api.state.job_store.read_error = psycopg.OperationalError("secret DSN must stay private")
        response = await client.get(job["events_url"])
        assert response.status_code == 503 and "secret" not in response.text


@pytest.mark.parametrize("failure", ["exception", "timeout"])
async def test_storage_failure_after_stream_started_is_safe_and_does_not_fail_job(
    api, monkeypatch, failure
):
    async with serve(api) as client:
        job = (await submit(client)).json()
        async with client.stream("GET", job["events_url"]) as response:
            lines = response.aiter_lines()
            await next_event(lines)
            if failure == "exception":
                api.state.job_store.read_error = psycopg.OperationalError("private credentials")
            else:
                monkeypatch.setattr("talotrace.api.status.READ_TIMEOUT_SECONDS", 0.05)
                api.state.job_store.read_gate = asyncio.Event()
            name, error = await next_event(lines)
            assert name == "stream_error" and error["retryable"]
            assert error["generation_cancelled"] is False and "private" not in str(error)
            assert [line async for line in lines] == []
        assert api.state.job_store.rows[job["id"]]["state"] == "queued"
        if failure == "timeout":
            assert api.state.job_store.cancelled_reads == 1


async def test_video_resume_reports_upstream_block_and_preserves_render_progress(api):
    async with serve(api) as client:
        job = (await submit(client, "video")).json()
        api.state.job_store.change(job["keyframe_job_id"], state="failed", attempts=1)
        api.state.video_store.change(
            job["id"], state="failed", attempts=1, progress={"scene_order": 3, "completed_clips": 2}
        )
        response = (await client.get(job["status_url"])).json()
        assert response["blocking_reason"] == "upstream_requires_resume"
        assert response["can_resume"] is False
        assert response["progress"] == {"scene_order": 3, "completed_clips": 2}
        assert (await client.post(job["status_url"] + "/resume")).status_code == 409
        api.state.job_store.change(job["keyframe_job_id"], state="keyframes_completed")
        assert (await client.get(job["status_url"])).json()["can_resume"] is True
        response = await client.post(job["status_url"] + "/resume")
        assert response.status_code == 202
        assert response.json()["progress"] == {"scene_order": 3, "completed_clips": 2}


async def test_video_download_hashing_does_not_block_status_and_supports_byte_ranges(
    api, monkeypatch
):
    import talotrace.api.status as status_helpers

    started, release = threading.Event(), threading.Event()
    original = status_helpers.hashlib.file_digest

    def slow_digest(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    async with serve(api) as client:
        job = (await submit(client, "video")).json()
        content = b"synthetic transport-only file; not a generated or validated video"
        api.state.artifacts.write(job["id"], "final.mp4", content)
        api.state.video_store.change(
            job["id"], state="video_completed", result={"media": {"sha256": sha256(content)}}
        )
        monkeypatch.setattr(status_helpers.hashlib, "file_digest", slow_digest)
        download = asyncio.create_task(
            client.get(job["status_url"] + "/video", headers={"Range": "bytes=0-8"})
        )
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(0.01)
            response = await asyncio.wait_for(client.get(job["status_url"]), 0.5)
            assert response.status_code == 200 and not download.done()
        finally:
            release.set()
        response = await download
        assert response.status_code == 206 and response.content == content[:9]
        api.state.artifacts.write(job["id"], "final.mp4", b"corrupt")
        response = await client.get(job["status_url"] + "/video")
        assert response.status_code == 409
