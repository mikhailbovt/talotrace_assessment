"""Contract and recovery tests. Generated test artifacts stay under data/tests."""

import asyncio
import base64
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

from talotrace.api.keyframes import router
from talotrace.artifacts.store import ArtifactStore, inspect_png, sha256
from talotrace.config import Settings
from talotrace.jobs.preconditions import (
    EXTRACTION_POLICY,
    CorpusNotReady,
    validate_extracted_sources,
)
from talotrace.jobs.schemas import Direction, KeyframeRequest, validate_grounding
from talotrace.jobs.worker import KeyframePipeline, pipeline_settings
from talotrace.providers.openai_media import direct, generate_frame, make_client, safe_error


@pytest.fixture
def plan():
    return Direction.model_validate(
        {
            "title": "Sharing electrons",
            "answer_summary": "Shared electron density can stabilize atoms.",
            "timing_basis": "estimated_until_tts",
            "grounding_limitations": [],
            "scenes": [
                {
                    "order": i,
                    "title": f"Scene {i}",
                    "educational_objective": "Explain electron sharing",
                    "narration": "Hydrogen atoms can share electrons.",
                    "start_seconds": (i - 1) * 8,
                    "end_seconds": i * 8,
                    "visible_labels": ["H", "Shared electrons"],
                    "scene_direction": "Draw two hydrogen nuclei and shared electron density.",
                    "image_prompt": "A simple covalent bonding schematic.",
                    "needs_hand": False,
                    "source_ids": ["source-a"],
                    "limitations": [],
                }
                for i in range(1, 6)
            ],
        }
    )


@pytest.fixture
def png():
    output = io.BytesIO()
    Image.new("RGB", (2048, 1152), "white").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def artifacts():
    return ArtifactStore(Path("data/tests/unit") / str(uuid4()))


@pytest.fixture
def job(plan):
    settings = Settings()
    return {
        "id": uuid4(),
        "request": {"question": "Why do atoms share electrons?", "keyframe_count": 5},
        "settings": pipeline_settings(settings),
        "state": "generating_keyframes",
        "frames": [],
        "context": {
            "question": "Why do atoms share electrons?",
            "corpus_id": "test-only",
            "corpus_summary": {
                "extraction_policy": EXTRACTION_POLICY,
                "extraction_verified": True,
                "vision_required_pages": 1,
                "vision_completed_pages": 1,
            },
            "results": [{"id": "source-a", "quality_flags": [], "extractor": "pdf-inspector"}],
        },
        "direction": {"plan": plan.model_dump()},
        "error": None,
        "attempts": 1,
        "created_at": "2026-09-08T00:00:00Z",
        "updated_at": "2026-09-08T00:00:00Z",
    }


def test_director_rejects_wrong_count_citations_and_timeline(plan, job):
    validate_grounding(plan, job["context"], 5)
    with pytest.raises(ValueError, match="scene count"):
        validate_grounding(plan, job["context"], 6)
    plan.scenes[0].source_ids = ["fabricated"]
    with pytest.raises(ValueError, match="unknown citation"):
        validate_grounding(plan, job["context"], 5)
    raw = plan.model_dump()
    raw["scenes"][0]["end_seconds"] = 0
    with pytest.raises(ValidationError):
        Direction.model_validate(raw)


def test_flagged_source_requires_limitations(plan, job):
    job["context"]["results"][0]["quality_flags"] = ["source_formula_rendering_error"]
    with pytest.raises(ValueError, match="omitted extraction limitations"):
        validate_grounding(plan, job["context"], 5)
    plan.grounding_limitations = ["Use prose; source formulas are damaged."]
    validate_grounding(plan, job["context"], 5)


def test_old_or_fallback_corpus_cannot_reach_media(job):
    validate_extracted_sources(job["context"])
    job["context"]["results"][0]["extractor"] = "fallback_pypdf"
    with pytest.raises(CorpusNotReady):
        validate_extracted_sources(job["context"])
    job["context"]["results"][0]["extractor"] = "gpt-5.6-luna-vision"
    job["context"]["corpus_summary"] = {}
    with pytest.raises(CorpusNotReady):
        validate_extracted_sources(job["context"])


def test_question_and_image_validation(png):
    with pytest.raises(ValidationError):
        KeyframeRequest(question="     ")
    assert inspect_png(png)["width"] == 2048
    with pytest.raises(ValueError, match="dimensions"):
        inspect_png(png, expected=(1024, 1024))
    with pytest.raises(OSError):
        inspect_png(png[:100])


async def test_images_attach_exact_anchor_and_only_requested_hand(plan, png):
    captured = []

    async def edit(**kwargs):
        captured.append({**kwargs, "image": [handle.read() for handle in kwargs["image"]]})
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), revised_prompt=None)],
            usage=None,
            _request_id="test-request",
        )

    client = SimpleNamespace(images=SimpleNamespace(edit=edit))
    settings = Settings()
    scene = plan.scenes[0]
    data, receipt = await generate_frame(
        client, "gpt-image-2", scene, settings.style_anchor_path, settings.hand_reference_path
    )
    assert data == png
    assert captured[0]["image"] == [settings.style_anchor_path.read_bytes()]
    assert receipt["references"][0]["sha256"] == sha256(settings.style_anchor_path.read_bytes())
    assert "input_fidelity" not in captured[0]
    scene.needs_hand = True
    await generate_frame(
        client, "gpt-image-2", scene, settings.style_anchor_path, settings.hand_reference_path
    )
    assert captured[1]["image"][1] == settings.hand_reference_path.read_bytes()


async def test_incomplete_director_and_invalid_image_are_rejected(plan, job):
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=AsyncMock(return_value=SimpleNamespace(status="incomplete", output_parsed=None))
        )
    )
    with pytest.raises(ValueError, match="complete structured plan"):
        await direct(client, "gpt-5.6-luna", job["context"], 5)
    client = SimpleNamespace(
        images=SimpleNamespace(edit=AsyncMock(return_value=SimpleNamespace(data=[])))
    )
    settings = Settings()
    with pytest.raises(ValueError, match="exactly one"):
        await generate_frame(
            client,
            "gpt-image-2",
            plan.scenes[0],
            settings.style_anchor_path,
            settings.hand_reference_path,
        )


async def test_rejected_corpus_stops_before_director_or_image(monkeypatch, artifacts, job):
    provider = AsyncMock()
    monkeypatch.setattr("talotrace.jobs.worker.make_client", lambda settings: FakeClient())
    monkeypatch.setattr("talotrace.jobs.worker.direct", provider)
    monkeypatch.setattr("talotrace.jobs.worker.generate_frame", provider)
    job["context"]["corpus_summary"] = {}
    await KeyframePipeline(Settings(), FakeStore(job), artifacts).run(job)
    assert job["state"] == "failed" and job["error"]["category"] == "CorpusNotReady"
    assert provider.await_count == 0


def test_secret_safe_errors_and_explicit_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-global-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    settings = Settings(openai_api_key="test-session-key")
    client = make_client(settings)
    assert client.api_key == "test-session-key"
    assert client.organization == "" and client.project == ""
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/images"),
        headers={"x-request-id": "safe-request"},
    )
    error = BadRequestError(
        "secret-request-text", response=response, body={"code": "invalid_parameter"}
    )
    details = safe_error(error)
    assert details["http_status"] == 400
    assert "secret-request-text" not in str(details)


class FakeStore:
    def __init__(self, job):
        self.job = job

    async def update(self, job_id, **values):
        self.job.update(values)

    async def get(self, job_id):
        return self.job if job_id == str(self.job["id"]) else None


class FakeClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


async def test_failed_job_resume_retains_successful_images(monkeypatch, artifacts, job, png):
    calls = []

    async def generate(client, model, scene, style, hand):
        calls.append(scene.order)
        if calls == [1, 2]:
            raise TimeoutError("Sensitive upstream exception text")
        return png, {"order": scene.order, "image": inspect_png(png), "request_id": "test"}

    monkeypatch.setattr("talotrace.jobs.worker.make_client", lambda settings: FakeClient())
    monkeypatch.setattr("talotrace.jobs.worker.generate_frame", generate)
    store = FakeStore(job)
    pipeline = KeyframePipeline(Settings(), store, artifacts)
    await pipeline.run(job)
    assert job["state"] == "failed" and len(job["frames"]) == 1
    assert job["error"] == {
        "category": "TimeoutError",
        "failed_stage": "generating_keyframes",
        "scene_order": 2,
    }
    await pipeline.run(job)
    assert job["state"] == "keyframes_completed" and len(job["frames"]) == 5
    assert calls == [1, 2, 2, 3, 4, 5]
    await pipeline.run(job)
    assert calls == [1, 2, 2, 3, 4, 5]  # All receipts reused, zero additional provider calls.
    assert artifacts.read_json(str(job["id"]), "manifest.json")["video_generated"] is False


async def test_interruption_marks_job_and_preserves_receipt(monkeypatch, artifacts, job, png):
    async def generate(client, model, scene, style, hand):
        if scene.order == 2:
            raise asyncio.CancelledError()
        return png, {"order": scene.order, "image": inspect_png(png)}

    monkeypatch.setattr("talotrace.jobs.worker.make_client", lambda settings: FakeClient())
    monkeypatch.setattr("talotrace.jobs.worker.generate_frame", generate)
    with pytest.raises(asyncio.CancelledError):
        await KeyframePipeline(Settings(), FakeStore(job), artifacts).run(job)
    assert job["state"] == "interrupted" and len(job["frames"]) == 1
    assert job["error"]["failed_stage"] == "generating_keyframes"
    assert job["error"]["scene_order"] == 2
    assert artifacts.path(str(job["id"]), "scene-01.json").exists()


async def test_artifact_receipt_recovers_db_update_gap(monkeypatch, artifacts, job, png):
    generate = AsyncMock(return_value=(png, {"order": 1, "image": inspect_png(png)}))
    monkeypatch.setattr("talotrace.jobs.worker.make_client", lambda settings: FakeClient())

    async def scene_image(client, model, scene, style, hand):
        data, receipt = await generate()
        return data, receipt | {"order": scene.order}

    monkeypatch.setattr("talotrace.jobs.worker.generate_frame", scene_image)
    pipeline = KeyframePipeline(Settings(), FakeStore(job), artifacts)
    await pipeline.run(job)
    job["frames"] = []  # Simulate a DB rollback after files were durably written.
    await pipeline.run(job)
    assert generate.await_count == 5
    assert len(job["frames"]) == 5
    artifacts.path(str(job["id"]), "scene-01.png").write_bytes(b"damaged")
    await pipeline.run(job)
    assert job["state"] == "failed" and generate.await_count == 5


async def test_result_and_safe_artifact_http(artifacts, job, png):
    app = FastAPI()
    app.include_router(router)
    app.state.job_store = FakeStore(job)
    app.state.artifacts = artifacts
    job_id = str(job["id"])
    job["frames"] = [{"order": 1, "image": inspect_png(png)}]
    artifacts.write(job_id, "scene-01.png", png)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get(f"/v1/keyframe-jobs/{job_id}/images/1")).content == png
        assert (await c.get(f"/v1/keyframe-jobs/{job_id}/images/2")).status_code == 404
        assert (await c.get(f"/v1/keyframe-jobs/{job_id}/images/99")).status_code == 422
        assert (await c.get("/v1/keyframe-jobs/not-a-uuid/images/1")).status_code == 422
        result = (await c.get(f"/v1/keyframe-jobs/{job_id}/result")).json()
        assert (
            result["video_generated"] is False and result["retrieval"]["corpus_id"] == "test-only"
        )
    with pytest.raises(ValueError, match="escapes"):
        artifacts.path(job_id, "../private.json")
