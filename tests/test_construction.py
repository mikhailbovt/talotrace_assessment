"""Reviewed stage guides must remain bound to source, image and provider evidence."""

from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import ValidationError

from talotrace.artifacts.construction import load_reviewed_stage
from talotrace.artifacts.store import json_bytes, sha256
from talotrace.providers.ltx_graph import build_graph, render_settings


def guide_fixture():
    root = Path("data/tests/unit-construction") / str(uuid4())
    folder = root / "guides"
    folder.mkdir(parents=True)
    source_id = str(uuid4())
    source = root / source_id / "scene-01.png"
    source.parent.mkdir()
    Image.new("RGB", (2048, 1152), "white").save(source)
    images = {}
    for role in ("blank", "outline"):
        image = folder / f"{role}.png"
        Image.new("RGB", (2048, 1152), "white").save(image)
        receipt = folder / f"{role}.receipt.json"
        receipt.write_bytes(json_bytes({"scope": "synthetic_unit_fixture_not_provider_evidence"}))
        images[role] = {
            "file": image.name,
            "sha256": sha256(image.read_bytes()),
            "provider_receipt_file": receipt.name,
            "provider_receipt_sha256": sha256(receipt.read_bytes()),
        }
    manifest = {
        "schema_version": "construction-stage-v1",
        "approval_status": "approved",
        "approved_by": "root",
        "source_keyframe_job_id": source_id,
        "source_scene_order": 1,
        "source_keyframe_sha256": sha256(source.read_bytes()),
        "images": images,
        "stage": {
            "id": "beaker-outline-v1",
            "frames": 97,
            "prompt": "Blank → rim → walls → base. "
            "Draw only an empty black beaker contour with the reference marker hand, "
            "without liquid, color, lettering, symbols or other objects.",
        },
    }
    path = folder / "manifest.json"
    path.write_bytes(json_bytes(manifest))
    return root, path, manifest


def test_reviewed_utf8_guide_manifest_and_all_provenance_hashes():
    root, path, raw = guide_fixture()
    stage, files = load_reviewed_stage(path, sha256(json_bytes(raw)), root)
    assert "→" in stage.stage.prompt
    assert stage.stage.frames == 97
    assert files["outline"].read_bytes() == (path.parent / "outline.png").read_bytes()
    with pytest.raises(ValueError, match="approved hash"):
        load_reviewed_stage(path, "0" * 64, root)
    files["outline_receipt"].write_bytes(b"{}")
    with pytest.raises(ValueError, match="provider receipt"):
        load_reviewed_stage(path, sha256(json_bytes(raw)), root)


@pytest.mark.parametrize(
    "field,value", [("approval_status", "pending"), ("approved_by", "someone_else")]
)
def test_unapproved_guides_cannot_reach_provider_work(field, value):
    root, path, raw = guide_fixture()
    raw[field] = value
    path.write_bytes(json_bytes(raw))
    with pytest.raises(ValidationError):
        load_reviewed_stage(path, sha256(json_bytes(raw)), root)


def test_source_and_image_changes_and_path_escape_are_rejected():
    root, path, raw = guide_fixture()
    source = root / raw["source_keyframe_job_id"] / "scene-01.png"
    original = source.read_bytes()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source keyframe changed"):
        load_reviewed_stage(path, sha256(json_bytes(raw)), root)
    source.write_bytes(original)
    raw["images"]["blank"]["file"] = "../scene-01.png"
    path.write_bytes(json_bytes(raw))
    with pytest.raises(ValueError, match="basenames"):
        load_reviewed_stage(path, sha256(json_bytes(raw)), root)
    raw["images"]["blank"]["file"] = "blank.png"
    path.write_bytes(json_bytes(raw))
    (path.parent / "blank.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image failed"):
        load_reviewed_stage(path, sha256(json_bytes(raw)), root)


def test_primitive_graph_uses_only_approved_prompt_and_matched_plates():
    _, _, manifest = guide_fixture()
    prompt = manifest["stage"]["prompt"]
    graph, report = build_graph(
        mode="whiteboard_loop",
        phase="draw",
        scene={},  # Full chemistry scene must not enter this isolated prompt.
        images={"blank": "blank.png", "keyframe": "outline.png", "hand_reference": "hand.png"},
        profile=render_settings()["profile"] | {"draw_frames": 97},
        seed=42,
        output_prefix="tests/primitive",
        prompt_override=prompt,
    )
    assert graph["positive"]["inputs"]["text"] == prompt
    assert graph["blank"] == {"class_type": "LoadImage", "inputs": {"image": "blank.png"}}
    assert graph["keyframe"]["inputs"]["image"] == "outline.png"
    for stage in ("base", "refine"):
        assert graph[f"{stage}_start_pin"]["inputs"]["image"] == ["blank", 0]
        assert graph[f"{stage}_finished_end"]["inputs"]["frame_idx"] == 96
        assert graph[f"{stage}_msr"]["inputs"]["pic1"] == ["hand", 0]
        assert "pic2" not in graph[f"{stage}_msr"]["inputs"]
    assert report["frames"] == 97
