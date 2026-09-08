"""Only an explicitly approved, hash-bound narration-only correction may reach TTS."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.jobs.schemas import Direction, StrictModel


class NarrationCorrection(StrictModel):
    order: int = Field(ge=1, le=10)
    original_narration: str = Field(min_length=1, max_length=1800)
    replacement_narration: str = Field(min_length=1, max_length=1800)
    source_ids: list[str] = Field(min_length=1, max_length=12)


class NarrationReview(StrictModel):
    schema_version: Literal["narration-review-v1"]
    keyframe_job_id: UUID
    original_direction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    approval_status: Literal["approved"]
    approved_by: Literal["root"]
    scenes: list[NarrationCorrection] = Field(min_length=1, max_length=10)


def selected_review(artifacts: ArtifactStore, upstream_id: str) -> dict | None:
    value = artifacts.read_json(upstream_id, "narration-review.json")
    if value is None:
        return None
    review = NarrationReview.model_validate(value)
    if str(review.keyframe_job_id) != str(UUID(upstream_id)):
        raise ValueError("Narration review belongs to another keyframe job")
    normalized = review.model_dump(mode="json")
    return {"sha256": sha256(json_bytes(normalized)), "review": normalized}


def effective_direction(
    direction_receipt: dict, upstream_id: str, selected: dict | None
) -> tuple[Direction, dict]:
    original = Direction.model_validate(direction_receipt["plan"])
    effective = original.model_copy(deep=True)
    changes = []
    if selected:
        if selected["sha256"] != sha256(json_bytes(selected["review"])):
            raise ValueError("Narration review checksum changed")
        review = NarrationReview.model_validate(selected["review"])
        if str(review.keyframe_job_id) != upstream_id:
            raise ValueError("Narration review belongs to another job")
        if review.original_direction_sha256 != sha256(json_bytes(direction_receipt)):
            raise ValueError("Narration review targets a stale direction receipt")
        orders = [item.order for item in review.scenes]
        if len(orders) != len(set(orders)):
            raise ValueError("Duplicate narration correction")
        for item in review.scenes:
            if item.order > len(original.scenes):
                raise ValueError("Narration correction refers to an absent scene")
            scene = original.scenes[item.order - 1]
            if item.original_narration != scene.narration or item.source_ids != scene.source_ids:
                raise ValueError("Narration review original text or citations changed")
            effective.scenes[item.order - 1].narration = item.replacement_narration
            changes.append(item.model_dump())
    return effective, {
        "review_sha256": selected["sha256"] if selected else None,
        "original_direction_sha256": sha256(json_bytes(direction_receipt)),
        "original_scene_scripts": [
            {"order": item.order, "narration": item.narration} for item in original.scenes
        ],
        "effective_scene_scripts": [
            {"order": item.order, "narration": item.narration} for item in effective.scenes
        ],
        "narration_corrections": changes,
    }
