"""Reviewed revisions reject stale evidence and reuse only unchanged outputs."""

import copy
import io
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from scripts.revise_keyframes import (
    RevisionRefused,
    digest,
    frame_identity,
    seed_artifacts,
    validate_revision,
)
from talotrace.artifacts.store import ArtifactStore, inspect_png, sha256
from talotrace.config import Settings
from talotrace.jobs.preconditions import EXTRACTION_POLICY
from talotrace.jobs.worker import KeyframePipeline, pipeline_settings


@pytest.fixture
def case():
    artifacts = ArtifactStore(Path("data/tests/unit") / str(uuid4()))
    settings = pipeline_settings(Settings())
    scene = {
        "order": 1,
        "title": "Hydrogen",
        "educational_objective": "Explain shared electrons",
        "narration": "The two hydrogen atoms share a pair of electrons.",
        "start_seconds": 0.0,
        "end_seconds": 8.0,
        "visible_labels": ["H2"],
        "scene_direction": "Draw the shared pair",
        "image_prompt": "Draw two shared dots",
        "needs_hand": False,
        "source_ids": ["source"],
        "limitations": [],
    }
    plan = {
        "title": "Covalent bonding",
        "answer_summary": "Electrons are shared.",
        "timing_basis": "estimated_until_tts",
        "grounding_limitations": [],
        "scenes": [
            scene | {"order": i, "start_seconds": (i - 1) * 8.0, "end_seconds": i * 8.0}
            for i in range(1, 6)
        ],
    }
    summary = {
        "extraction_policy": EXTRACTION_POLICY,
        "extraction_verified": True,
        "vision_required_pages": 1,
        "vision_completed_pages": 1,
    }
    hit = {
        "id": "source",
        "content": "Hydrogen shares two electrons.",
        "file_sha256": "pdf-sha",
        "path": "chemistry.pdf",
        "page_number": 1,
        "quality_flags": [],
        "extractor": "gpt-5.6-luna-vision",
    }
    hit["content_sha256"] = sha256(hit["content"].encode())
    original = {
        "id": str(uuid4()),
        "state": "keyframes_completed",
        "settings": settings,
        "request": {"question": "Why do atoms form covalent bonds?", "keyframe_count": 5},
        "context": {"corpus_id": "final", "corpus_summary": summary, "results": [hit]},
        "direction": {"plan": plan, "model": "gpt-5.6-luna", "request_id": "original-request"},
        "frames": [],
    }
    output = io.BytesIO()
    Image.new("RGB", (2048, 1152), "white").save(output, format="PNG")
    png = output.getvalue()
    for scene in plan["scenes"]:
        receipt = {
            "order": scene["order"],
            "input_sha256": frame_identity(scene, settings),
            "source_ids": ["source"],
            "references": [settings["style_anchor"]],
            "image": inspect_png(png),
            "prompt": "Original image request",
            "artifact_url": f"/original/{scene['order']}",
        }
        original["frames"].append(receipt)
        artifacts.write(original["id"], f"scene-{scene['order']:02d}.png", png)
        artifacts.write_json(original["id"], f"scene-{scene['order']:02d}.json", receipt)
    for name in ("context", "direction"):
        artifacts.write_json(original["id"], f"{name}.json", original[name])
    direction_hash = digest(original["direction"])
    proposal = {
        "schema_version": "image-correction-proposal-v1",
        "keyframe_job_id": original["id"],
        "preserve_originals": True,
        "original_direction_sha256": direction_hash,
        "generation_requirements": {
            "model": "gpt-image-2",
            "quality": "high",
            "size": "2048x1152",
            "attach_original_style_anchor": True,
        },
        "corrections": [
            {
                "scene_order": i,
                "original_direction_sha256": direction_hash,
                "original_image_sha256": inspect_png(png)["sha256"],
                "original_scene": copy.deepcopy(plan["scenes"][i - 1]),
                "original_image_prompt": "Original image request",
                "reason": "Correct the dots",
                "narration_affected": False,
                "narration_review": "Unchanged and supported",
                "source_ids": ["source"],
                "replacement_scene_direction": "Exactly two central dots",
                "replacement_image_prompt": "Exactly two dots total; no outer dots",
                "replacement_visible_labels": ["H2", "Two electrons"],
            }
            for i in (3, 4)
        ],
    }
    return {
        "original": original,
        "proposal": proposal,
        "approved_hash": digest(proposal),
        "settings": settings,
        "artifacts": artifacts,
        "active": {"id": "final", "summary": summary},
        "sources": [copy.deepcopy(hit)],
    }


def test_revision_reuses_three_frames_and_preserves_originals(case):
    artifacts = case["artifacts"]
    original = case["original"]
    before = {p.name: sha256(p.read_bytes()) for p in artifacts.directory(original["id"]).iterdir()}
    direction, lineage, receipts = validate_revision(**case)
    assert lineage["reused_scene_orders"] == [1, 2, 5]
    assert lineage["generated_scene_orders"] == [3, 4]
    assert direction["original_provider_output"] == original["direction"]
    new_id = str(uuid4())
    for _ in range(2):  # Unpublished transaction recovery reuses the same reviewed files.
        seed_artifacts(artifacts, new_id, original, case["proposal"], direction, lineage, receipts)
    assert len(list(artifacts.directory(new_id).glob("*.png"))) == 3
    for scene in direction["plan"]["scenes"]:
        order = scene["order"]
        receipt = artifacts.frame_receipt(new_id, order, frame_identity(scene, case["settings"]))
        assert (receipt is not None) == (order in {1, 2, 5})
        assert scene["narration"] == original["direction"]["plan"]["scenes"][order - 1]["narration"]
        if receipt:
            assert new_id in receipt["artifact_url"]
            assert receipt["reused_from"]["receipt_sha256"] == digest(receipts[order])
    assert before == {
        p.name: sha256(p.read_bytes()) for p in artifacts.directory(original["id"]).iterdir()
    }


@pytest.mark.parametrize("changed_orders", [[3, 4], [4]])
async def test_existing_worker_generates_only_changed_scenes_and_replays_without_calls(
    case, monkeypatch, changed_orders
):
    case["proposal"]["corrections"] = [
        change
        for change in case["proposal"]["corrections"]
        if change["scene_order"] in changed_orders
    ]
    case["approved_hash"] = digest(case["proposal"])
    direction, lineage, receipts = validate_revision(**case)
    new_id = str(uuid4())
    original, artifacts = case["original"], case["artifacts"]
    seed_artifacts(artifacts, new_id, original, case["proposal"], direction, lineage, receipts)
    revised = original | {"id": new_id, "direction": direction, "frames": [], "state": "queued"}
    calls = []

    @asynccontextmanager
    async def client(settings):
        yield None

    class Store:
        async def update(self, job_id, **values):
            revised.update(values)

    async def generate(client, model, scene, style, hand):
        calls.append(scene.order)
        png = artifacts.path(original["id"], "scene-01.png").read_bytes()
        return png, {"order": scene.order, "image": inspect_png(png)}

    monkeypatch.setattr("talotrace.jobs.worker.make_client", client)
    monkeypatch.setattr("talotrace.jobs.worker.generate_frame", generate)
    pipeline = KeyframePipeline(Settings(), Store(), artifacts)
    await pipeline.run(revised)
    assert revised["state"] == "keyframes_completed"
    assert calls == changed_orders
    assert len(revised["frames"]) == 5
    await pipeline.run(revised)
    assert calls == changed_orders


@pytest.mark.parametrize(
    "change,expected",
    [
        ("corpus", "no longer active"),
        ("source", "content changed"),
        ("direction", "differs from DB"),
        ("image", "integrity"),
        ("patch", "approved proposal hash"),
        ("narration", "unsupported fields"),
        ("citation", "citations changed"),
        ("incomplete", "not completed"),
    ],
)
def test_revision_refuses_stale_or_unreviewed_inputs(case, change, expected):
    if change == "corpus":
        case["active"]["id"] = "replacement-corpus"
    elif change == "source":
        case["sources"][0]["content"] = "Different source"
    elif change == "direction":
        case["original"]["direction"]["request_id"] = "changed"
    elif change == "image":
        case["original"]["frames"][0]["image"]["sha256"] = "changed"
        case["artifacts"].write_json(
            case["original"]["id"], "scene-01.json", case["original"]["frames"][0]
        )
    elif change == "patch":
        case["proposal"]["corrections"][0]["replacement_image_prompt"] = "Unapproved image"
    elif change == "narration":
        case["proposal"]["corrections"][0]["replacement_narration"] = "Unreviewed speech"
        case["approved_hash"] = digest(case["proposal"])
    elif change == "citation":
        case["proposal"]["corrections"][0]["source_ids"] = ["fabricated"]
        case["approved_hash"] = digest(case["proposal"])
    elif change == "incomplete":
        case["original"]["state"] = "generating_keyframes"
    with pytest.raises((RevisionRefused, ValueError), match=expected):
        validate_revision(**case)
