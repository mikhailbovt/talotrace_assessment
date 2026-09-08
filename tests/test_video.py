"""Meaningful graph, recovery, narration-preservation, and actual ffmpeg contract checks."""

import io
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError

from talotrace.api.videos import VideoRequest, router
from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.artifacts.video import (
    assemble_final,
    assemble_scene,
    assembly_timing_policy,
    command,
    make_timeline,
    narration_track,
    validate_final,
)
from talotrace.config import Settings
from talotrace.jobs.narration_review import effective_direction, selected_review
from talotrace.jobs.preconditions import EXTRACTION_POLICY
from talotrace.jobs.video_worker import VideoPipeline, video_settings
from talotrace.jobs.worker import pipeline_settings
from talotrace.providers.kokoro import inspect_wav
from talotrace.providers.ltx import AmbiguousSubmission, ComfyClient, ComfyExecutionError
from talotrace.providers.ltx_graph import build_graph, render_identity_settings, render_settings


def speech(seconds=1.25):
    samples = np.sin(np.arange(int(seconds * 24000)) * 2 * np.pi * 440 / 24000) * 10000
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(samples.astype("<i2").tobytes())
    return output.getvalue()


def scene(order=1):
    return {
        "order": order,
        "educational_objective": "Shared electron pair",
        "narration": "Two hydrogen atoms share an electron pair.",
        "visible_labels": ["H", "H"],
        "scene_direction": "Draw a shared electron pair.",
    }


def receipt(text, data):
    return {"text": text, "text_sha256": sha256(text.encode()), "audio": inspect_wav(data)}


def test_public_request_requires_one_input_and_known_mode():
    with pytest.raises(ValidationError):
        VideoRequest()
    with pytest.raises(ValidationError):
        VideoRequest(question="Explain bonds", keyframe_job_id=uuid4())
    with pytest.raises(ValidationError):
        VideoRequest(question="Explain bonds", mode="fake_loop")
    with pytest.raises(ValidationError, match="retired for new requests"):
        VideoRequest(question="Explain bonds", mode="references")
    assert VideoRequest(question="Explain bonds").mode == "whiteboard_loop"
    assert (
        VideoRequest(question=" Explain   bonds ", mode="whiteboard_loop").question
        == "Explain bonds"
    )


def review_fixture():
    job_id = str(uuid4())
    direction = {
        "request_id": "original",
        "plan": {
            "title": "pH",
            "answer_summary": "Explain acidity",
            "timing_basis": "estimated_until_tts",
            "grounding_limitations": [],
            "scenes": [
                {
                    **scene(i),
                    "title": "pH",
                    "start_seconds": (i - 1) * 5,
                    "end_seconds": i * 5,
                    "image_prompt": "pH diagram",
                    "needs_hand": False,
                    "source_ids": ["source1"],
                    "limitations": [],
                }
                for i in range(1, 6)
            ],
        },
    }
    review = {
        "schema_version": "narration-review-v1",
        "keyframe_job_id": job_id,
        "original_direction_sha256": sha256(json_bytes(direction)),
        "approval_status": "approved",
        "approved_by": "root",
        "scenes": [
            {
                "order": 2,
                "original_narration": scene()["narration"],
                "replacement_narration": "Below neutral pH, a solution is acidic.",
                "source_ids": ["source1"],
            }
        ],
    }
    return job_id, direction, review


def test_approved_narration_is_exactly_bound_and_only_changes_spoken_text():
    job_id, direction, review = review_fixture()
    selected = {"sha256": sha256(json_bytes(review)), "review": review}
    effective, evidence = effective_direction(direction, job_id, selected)
    assert effective.scenes[1].narration == review["scenes"][0]["replacement_narration"]
    original = direction["plan"]["scenes"][1]
    assert effective.scenes[1].model_dump(exclude={"narration"}) == {
        key: value for key, value in original.items() if key != "narration"
    }
    assert evidence["original_scene_scripts"][1]["narration"] == original["narration"]
    assert evidence["review_sha256"] == selected["sha256"]
    direction["request_id"] = "different director receipt"
    with pytest.raises(ValueError, match="stale"):
        effective_direction(direction, job_id, selected)


def test_wrong_original_citations_and_unapproved_narration_are_rejected():
    job_id, direction, review = review_fixture()
    review["scenes"][0]["source_ids"] = ["invented"]
    with pytest.raises(ValueError, match="citations"):
        effective_direction(
            direction, job_id, {"sha256": sha256(json_bytes(review)), "review": review}
        )
    artifacts = ArtifactStore(Path("data/tests/unit-video") / str(uuid4()))
    review["approval_status"] = "proposed"
    artifacts.write_json(job_id, "narration-review.json", review)
    with pytest.raises(ValueError):
        selected_review(artifacts, job_id)


@pytest.mark.parametrize(
    "mode,phase,guides",
    [
        ("references", "draw", []),
        ("whiteboard_loop", "draw", [176]),
        ("whiteboard_loop", "erase", [72]),
    ],
)
def test_both_passes_use_actual_msr_and_distinct_temporal_guides(mode, phase, guides):
    graph, report = build_graph(
        mode=mode,
        phase=phase,
        scene=scene(),
        images={"keyframe": "talotrace/keyframe.png", "hand_reference": "talotrace/hand.png"},
        profile=render_settings()["profile"],
        seed=42,
        output_prefix="tests/check",
    )
    assert graph["model"]["inputs"]["weight_dtype"] == "default"
    assert "bf16" in graph["model"]["inputs"]["unet_name"]
    for stage in ("base", "refine"):
        node = graph[f"{stage}_msr"]
        if mode == "whiteboard_loop":
            assert node["inputs"]["pic1"] == ["hand", 0]
            assert "pic2" not in node["inputs"]
            pin = graph[f"{stage}_start_pin"]
            assert pin["class_type"] == "LTXVImgToVideoInplace"
            assert pin["inputs"]["image"] == ["blank" if phase == "draw" else "keyframe", 0]
            assert pin["inputs"]["strength"] == 1.0
            assert pin["inputs"]["bypass"] is False
            assert pin["inputs"]["latent"] == ["empty_video" if stage == "base" else "upscale", 0]
            end = f"{stage}_{'finished' if phase == 'draw' else 'blank'}_end"
            assert graph[end]["inputs"]["latent"] == [f"{stage}_start_pin", 0]
            assert node["inputs"]["latent"] == [end, 2]
        else:
            assert node["inputs"]["pic1"] == ["keyframe", 0]
            assert node["inputs"]["pic2"] == ["hand", 0]
        assert node["inputs"]["msr_parameters"] == ["msr_loader", 1]
        actual_guides = [
            v["inputs"]["frame_idx"]
            for key, v in graph.items()
            if key.startswith(stage) and v["class_type"] == "LTXVAddGuide"
        ]
        assert actual_guides == guides
        assert graph[f"{stage}_crop"]["class_type"] == "LTXVCropGuides"
    assert graph["upscale"]["inputs"]["samples"] == ["base_crop", 2]
    first_refinement = (
        "refine_msr"
        if not guides
        else next(
            key
            for key, value in graph.items()
            if key.startswith("refine_") and value["class_type"] == "LTXVAddGuide"
        )
    )
    assert graph[first_refinement]["inputs"]["positive"] == ["conditioning", 0]
    assert graph["decode"]["inputs"]["samples"] == ["refine_crop", 2]
    assert "audio" not in graph["video"]["inputs"]
    assert not any(v["class_type"] == "LTXVAudioVAEDecode" for v in graph.values())
    assert report["audio_decoded"] is False
    if mode == "whiteboard_loop":
        assert report["references"] == {"pic1": "hand_reference"}
        assert "Keep all chemical symbols and labels legible and unchanged" not in report["prompt"]
        if phase == "erase":
            assert "eraser instead of a marker" in report["prompt"]
            assert "all untouched marks remain visible until wiped" in report["prompt"]
        else:
            assert "7.375 seconds" in report["prompt"]
            assert report["frames"] == 177


async def test_retired_queued_mode_fails_before_any_provider_work(monkeypatch):
    job_id = uuid4()
    artifacts = ArtifactStore(Path("data/tests/unit-video") / str(uuid4()))
    store, upstream = SimpleNamespace(update=AsyncMock()), SimpleNamespace(get=AsyncMock())
    tts, render = AsyncMock(), AsyncMock()
    monkeypatch.setattr("talotrace.jobs.video_worker.speak", tts)
    monkeypatch.setattr("talotrace.jobs.video_worker.ComfyClient.render", render)
    pipeline = VideoPipeline(SimpleNamespace(), store, upstream, artifacts)
    await pipeline.run({"id": job_id, "request": {"mode": "references"}})
    tts.assert_not_awaited()
    render.assert_not_awaited()
    upstream.get.assert_not_awaited()
    assert store.update.await_args.kwargs["state"] == "failed"
    assert store.update.await_args.kwargs["error"]["failed_stage"] == "preflight"


def test_long_narration_is_never_truncated_and_holds_are_explicit():
    text = scene()["narration"]
    data = speech(16.135)
    profile = render_settings()["profile"]
    timeline = make_timeline([scene()], [receipt(text, data)], "whiteboard_loop", profile)
    item = timeline["scenes"][0]
    assert item["reading_hold_frames"] > 0
    assert item["speech_end_seconds"] == 16.135
    assert item["end_seconds"] >= 16.135 + 73 / 24
    assert item["erase_frames"] == 73
    assert item["reading_hold_source"] == "exact_generated_keyframe"
    broken = receipt("truncated script", data)
    with pytest.raises(ValueError, match="exact complete"):
        make_timeline([scene()], [broken], "whiteboard_loop", profile)


def test_narration_track_preserves_every_original_pcm_sample():
    root = Path("data/tests/unit-video") / str(uuid4())
    root.mkdir(parents=True)
    scenes = [scene(1), scene(2)]
    audio = [speech(7.4), speech(1.3)]
    receipts = [receipt(s["narration"], a) for s, a in zip(scenes, audio, strict=True)]
    timeline = make_timeline(
        scenes, receipts, "whiteboard_loop", render_settings()["profile"], assembly_timing_policy()
    )
    paths = []
    for i, data in enumerate(audio):
        path = root / f"speech-{i}.wav"
        path.write_bytes(data)
        paths.append(path)
    combined = narration_track(paths, timeline)
    with wave.open(io.BytesIO(combined), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    for data, item in zip(audio, timeline["scenes"], strict=True):
        with wave.open(io.BytesIO(data), "rb") as wav:
            original = wav.readframes(wav.getnframes())
        start = item["start_frame"] * 24000 // 24 * 2
        assert pcm[start : start + len(original)] == original
        end = int(item["end_seconds"] * 24000) * 2
        assert pcm[start + len(original) : end] == bytes(end - start - len(original))
        assert item["reading_hold_frames"] >= 48
        assert item["speech_end_seconds"] <= item["erase_start_seconds"]
    assert timeline["scenes"][1]["speech_start_seconds"] == timeline["scenes"][0]["end_seconds"]


@pytest.mark.parametrize("seconds", [1.3, 7.4, 20.9])
def test_showing_policy_guarantees_two_seconds_and_preserves_long_speech_timeline(seconds):
    data = speech(seconds)
    profile = render_settings()["profile"]
    receipts = [receipt(scene()["narration"], data)]
    old = make_timeline([scene()], receipts, "whiteboard_loop", profile)
    new = make_timeline([scene()], receipts, "whiteboard_loop", profile, assembly_timing_policy())
    item = new["scenes"][0]
    assert item["reading_hold_frames"] == max(48, old["scenes"][0]["reading_hold_frames"])
    assert item["showing_end_seconds"] - item["showing_start_seconds"] >= 2
    assert item["speech_end_seconds"] == receipts[0]["audio"]["duration_seconds"]
    assert item["speech_end_seconds"] == pytest.approx(seconds, abs=1 / 24000)
    assert item["speech_end_seconds"] <= item["erase_start_seconds"]
    assert new["speech_samples"] == old["speech_samples"]
    if seconds == 20.9:
        assert new["frames"] == old["frames"] == 575


def test_assembly_policy_changes_job_identity_without_changing_render_seed():
    current = video_settings(Settings())
    legacy = {key: value for key, value in current.items() if key != "timing_policy"}
    assert sha256(json_bytes(current)) != sha256(json_bytes(legacy))
    assert render_identity_settings(current) == legacy
    assert render_identity_settings(legacy) == legacy
    # Removing only the new outer policy keeps all model/profile/narration bindings intact.
    changed_render = current | {"profile": current["profile"] | {"draw_frames": 97}}
    assert render_identity_settings(changed_render) != render_identity_settings(current)
    old_probe = {"scene": scene(), "settings": legacy}
    new_probe_render = {"scene": scene(), "settings": render_identity_settings(current)}
    assert sha256(json_bytes(old_probe)) == sha256(json_bytes(new_probe_render))


def fake_comfy(handler):
    artifacts = ArtifactStore(Path("data/tests/unit-video") / str(uuid4()))
    job_id = str(uuid4())
    client = ComfyClient("http://test", artifacts, job_id, poll_seconds=0.001)
    client.client = lambda: httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    return client, artifacts, job_id


def success_history(prompt_id="new"):
    return {
        prompt_id: {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "save": {
                    "images": [{"filename": "test.mp4", "subfolder": "tests", "type": "output"}]
                }
            },
        }
    }


async def test_confirmed_failure_requires_explicit_resume_and_preserves_attempts():
    posts = []

    def handle(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        if request.url.path == "/prompt":
            posts.append(request)
            return httpx.Response(200, json={"prompt_id": "new"})
        if request.url.path == "/history/new":
            return httpx.Response(200, json=success_history())
        return httpx.Response(200, content=b"video bytes")

    client, artifacts, job = fake_comfy(handle)
    graph = {"save": {"class_type": "SaveVideo"}}
    artifacts.write_json(
        job,
        "draw-submission.json",
        {
            "graph_sha256": sha256(json_bytes(graph)),
            "dispatch_id": "old",
            "prompt_id": "old",
            "state": "execution_failed",
            "job_attempt": 1,
        },
    )
    with pytest.raises(ComfyExecutionError, match="explicit"):
        await client.render("draw", graph, attempt=1)
    assert not posts
    data, record = await client.render("draw", graph, attempt=2)
    assert data == b"video bytes" and len(posts) == 1
    assert record["previous_dispatches"][0]["prompt_id"] == "old"


async def test_unresolved_prompt_reuses_saved_id_without_new_charge():
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/history/existing":
            return httpx.Response(200, json=success_history("existing"))
        if request.url.path == "/view":
            return httpx.Response(200, content=b"saved video")
        raise AssertionError("Resume must not submit or inspect an unrelated queue")

    client, artifacts, job = fake_comfy(handle)
    graph = {"save": {}}
    artifacts.write_json(
        job,
        "draw-submission.json",
        {
            "graph_sha256": sha256(json_bytes(graph)),
            "dispatch_id": "existing",
            "prompt_id": "existing",
            "state": "submitted",
            "job_attempt": 1,
        },
    )
    await client.render("draw", graph, attempt=2)
    assert paths == ["/history/existing", "/view"]


async def test_reference_upload_verifies_actual_bytes_and_reuses_provider_identity():
    data = b"actual reference PNG bytes for transport contract"
    posts = []

    def handle(request):
        if request.method == "POST":
            assert data in request.content
            posts.append(request)
            return httpx.Response(
                200, json={"name": "fixed.png", "subfolder": "talotrace", "type": "input"}
            )
        return httpx.Response(200, content=data)

    client, _, _ = fake_comfy(handle)
    assert await client.upload("hand.png", data) == "talotrace/fixed.png"
    assert await client.upload("hand.png", data) == "talotrace/fixed.png"
    assert len(posts) == 1


async def test_changed_unresolved_graph_does_not_overwrite_original_evidence():
    def handle(request):
        raise AssertionError("Must reject before provider work")

    client, artifacts, job = fake_comfy(handle)
    original = {"save": {"class_type": "SaveVideo", "inputs": {"video": ["old", 0]}}}
    artifacts.write_json(job, "draw-graph.json", original)
    artifacts.write_json(
        job,
        "draw-submission.json",
        {
            "graph_sha256": sha256(json_bytes(original)),
            "dispatch_id": "old",
            "prompt_id": "old",
            "state": "submitted",
            "job_attempt": 1,
        },
    )
    with pytest.raises(ValueError, match="graph changed"):
        await client.render("draw", {"save": {}}, attempt=2)
    assert artifacts.read_json(job, "draw-graph.json") == original


async def test_ambiguous_send_is_not_blindly_resubmitted():
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={})

    client, artifacts, job = fake_comfy(handle)
    graph = {"save": {}}
    artifacts.write_json(
        job,
        "draw-submission.json",
        {
            "graph_sha256": sha256(json_bytes(graph)),
            "dispatch_id": "unknown",
            "state": "dispatching",
            "job_attempt": 1,
        },
    )
    with pytest.raises(AmbiguousSubmission):
        await client.render("draw", graph, attempt=2)


async def test_validation_rejection_has_distinct_record_and_can_retry_next_attempt():
    posts = []

    def handle(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={})
        posts.append(request)
        return httpx.Response(400, json={"node_errors": {"node": {"errors": []}}})

    client, artifacts, job = fake_comfy(handle)
    with pytest.raises(httpx.HTTPStatusError):
        await client.render("draw", {"save": {}}, attempt=1)
    assert artifacts.read_json(job, "draw-submission.json")["state"] == "validation_rejected"
    with pytest.raises(ComfyExecutionError):
        await client.render("draw", {"save": {}}, attempt=1)
    assert len(posts) == 1


async def test_http_video_submission_reuses_upstream_and_rejects_legacy_corpus():
    settings = Settings()
    upstream_id, job_id = uuid4(), uuid4()
    upstream = {
        "id": upstream_id,
        "settings": pipeline_settings(settings),
        "request": {"question": "How does pH work?", "keyframe_count": 5},
    }
    value = {
        "id": job_id,
        "keyframe_job_id": upstream_id,
        "request": {"question": "How does pH work?", "mode": "whiteboard_loop"},
        "state": "waiting_for_keyframes",
        "progress": {},
        "result": None,
        "attempts": 0,
        "error": None,
        "created_at": "2026-09-08T00:00:00Z",
        "updated_at": "2026-09-08T00:00:00Z",
    }
    corpus = {
        "extraction_policy": EXTRACTION_POLICY,
        "extraction_verified": True,
        "vision_required_pages": 185,
        "vision_completed_pages": 185,
    }
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.artifacts = ArtifactStore(Path("data/tests/unit-video") / str(uuid4()))
    app.state.job_store = SimpleNamespace(
        get=AsyncMock(return_value=upstream), active_corpus_summary=AsyncMock(return_value=corpus)
    )
    app.state.video_store = SimpleNamespace(submit_video=AsyncMock(return_value=value))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/video-jobs",
            headers={"Idempotency-Key": "video-test-key"},
            json={"keyframe_job_id": str(upstream_id)},
        )
        assert response.status_code == 202
        assert response.json()["keyframe_job_id"] == str(upstream_id)
        assert response.json()["video_url"] is None
        args = app.state.video_store.submit_video.call_args.args
        assert args[4] is upstream
        assert args[3]["narration_review"] is None
        app.state.job_store.active_corpus_summary.return_value = {"extraction_policy": "pypdf"}
        response = await client.post(
            "/v1/video-jobs",
            headers={"Idempotency-Key": "video-test-key"},
            json={"question": "How does pH work?", "mode": "whiteboard_loop"},
        )
        assert response.status_code == 503
        assert app.state.video_store.submit_video.await_count == 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Actual ffmpeg required")
@pytest.mark.parametrize("mode", ["references", "whiteboard_loop"])
async def test_real_mux_keeps_complete_audio_and_exact_measured_duration(mode):
    root = Path("data/tests/unit-video") / str(uuid4())
    root.mkdir(parents=True)
    profile = render_settings()["profile"] | {
        "width": 160,
        "height": 96,
        "draw_frames": 9,
        "erase_frames": 9,
    }
    image = root / "keyframe.png"
    Image.new("RGB", (320, 192), "white").save(image)
    draw = root / "draw.mp4"
    await command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x192:rate=24",
            "-frames:v",
            "9",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(draw),
        ]
    )
    data = speech(1.275)
    audio = root / "speech.wav"
    audio.write_bytes(data)
    timeline = make_timeline(
        [scene()], [receipt(scene()["narration"], data)], mode, profile, assembly_timing_policy()
    )
    combined = root / "complete-kokoro-narration.wav"
    combined.write_bytes(narration_track([audio], timeline))
    segment = root / "scene-01.mp4"
    await assemble_scene(
        draw=draw,
        keyframe=image,
        erase=draw if mode == "whiteboard_loop" else None,
        output=segment,
        timeline=timeline["scenes"][0],
        profile=profile,
    )
    final = root / "final.mp4"
    await assemble_final(scenes=[segment], narration=combined, output=final)
    report = await validate_final(final, combined, timeline, profile)
    assert report["full_decode"] == "passed"
    assert report["decoded_audio_similarity"] > 0.97
    assert report["native_ltx_audio_used"] is False
    assert report["background_music"] is False
    assert report["video_duration_seconds"] == pytest.approx(timeline["duration_seconds"], abs=1e-6)
    assert report["audio_duration_seconds"] == pytest.approx(timeline["duration_seconds"], abs=0.06)
    assert abs(report["audio_video_duration_delta_seconds"]) < 0.06
    # Correct frame counts alone cannot prove timestamps cover the expected speech timeline.
    wrong_timeline = timeline | {"duration_seconds": timeline["duration_seconds"] + 0.5}
    with pytest.raises(ValueError, match="video presentation duration"):
        await validate_final(final, combined, wrong_timeline, profile)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Actual ffmpeg required")
async def test_five_segment_mux_uses_exact_timeline_not_rounded_container_durations():
    root = Path("data/tests/unit-video") / str(uuid4())
    root.mkdir(parents=True)
    segment = root / "seventeen-frames.mp4"
    await command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x192:rate=24",
            "-frames:v",
            "17",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(segment),
        ]
    )
    scenes = [scene(i) for i in range(1, 6)]
    data = speech(0.7)
    audio = root / "speech.wav"
    audio.write_bytes(data)
    profile = {"width": 160, "height": 96, "fps": 24, "draw_frames": 17}
    timeline = make_timeline(
        scenes, [receipt(s["narration"], data) for s in scenes], "references", profile
    )
    combined = root / "narration.wav"
    combined.write_bytes(narration_track([audio] * 5, timeline))
    final = root / "final.mp4"
    await assemble_final(scenes=[segment] * 5, narration=combined, output=final, timeline=timeline)
    report = await validate_final(final, combined, timeline, profile)
    assert report["video_frames"] == 85
    assert abs(report["video_duration_seconds"] - 85 / 24) < 0.001
    assert (root / "scene-concat.txt").read_text("utf-8").count("duration 0.708333333333") == 5
