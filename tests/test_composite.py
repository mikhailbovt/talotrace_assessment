"""Real local composition checks: hand is visible, exact science image returns, no audio."""

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image, ImageDraw

from talotrace.artifacts.composite import composite_settings, render_composite_clip
from talotrace.artifacts.video import command, inspect_clip
from talotrace.config import Settings
from talotrace.jobs.video_worker import video_settings


def test_renderer_selection_has_truthful_distinct_identity():
    ltx = video_settings(Settings(video_renderer="ltx"))
    composed = video_settings(Settings(video_renderer="keyframe-composite-v1"))
    assert ltx["renderer"]["new_ltx_generation"] is True
    assert composed["renderer"]["new_ltx_generation"] is False
    assert composed["renderer"]["name"] == "keyframe-composite-v1"
    assert composed["renderer"]["blank"]["sha256"]
    assert composed["hand_reference"] == ltx["hand_reference"]
    assert composed["assembly_profile"]["draw_frames"] == 97
    assert composed["assembly_profile"]["erase_frames"] == 25


def test_portable_blank_needs_no_diagnostic_corpus_and_checks_receipt(monkeypatch):
    root = (Path("data/tests/unit-composite") / str(uuid4())).resolve()
    for relative in (
        "configs/blank_whiteboard.json",
        "configs/blank_whiteboard.receipt.json",
        "assets/references/blank_whiteboard.png",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, target)
    monkeypatch.chdir(root)
    policy, image = composite_settings()
    assert image.is_file() and policy["blank"]["sha256"]
    assert not (root / "data/tests").exists()
    receipt = root / "configs/blank_whiteboard.receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt checksum"):
        composite_settings()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real ffmpeg required")
async def test_composite_has_visible_hand_correct_endpoints_and_no_native_audio():
    root = Path("data/tests/unit-composite") / str(uuid4())
    root.mkdir(parents=True)
    blank, diagram = root / "blank.png", root / "diagram.png"
    picture = Image.new("RGB", (320, 192), "white")
    picture.save(blank)
    ImageDraw.Draw(picture).rectangle((25, 30, 130, 160), outline="black", width=4)
    picture.save(diagram)
    settings = Settings()
    policy, _ = composite_settings()
    profile = {"width": 160, "height": 96, "fps": 24}
    for phase in ("draw", "erase"):
        output = root / f"{phase}.mp4"
        receipt = await render_composite_clip(
            phase=phase,
            keyframe=diagram,
            blank=blank,
            hand=settings.hand_reference_path,
            output=output,
            profile=profile,
            policy=policy,
        )
        media = await inspect_clip(output, policy[f"{phase}_frames"], 320, 192, 24)
        assert receipt["new_ltx_generation"] is False and media["audio_streams"] == 0
        raw = await command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(output),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 192, 320, 3)
        end = np.asarray(picture if phase == "draw" else Image.open(blank)).astype(np.int16)
        assert np.abs(frames[-1].astype(np.int16) - end).mean() < 3
        if phase == "draw":
            assert frames[0].mean() > 250
            mid = frames[48].astype(np.int16)
            skin = (mid[:, :, 0] > mid[:, :, 1] + 20) & (mid[:, :, 1] > mid[:, :, 2] + 15)
            assert skin.sum() > 1000
