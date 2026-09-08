"""Validated public requests and director output; timings remain estimates."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyframeRequest(StrictModel):
    question: str = Field(min_length=5, max_length=2000)
    keyframe_count: int = Field(default=5, ge=5, le=10)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 5 or not any(char.isalpha() for char in value):
            raise ValueError("Question must contain searchable words")
        return value


class Scene(StrictModel):
    order: int = Field(ge=1, le=10)
    title: str = Field(min_length=1, max_length=120)
    educational_objective: str = Field(min_length=1, max_length=500)
    narration: str = Field(min_length=1, max_length=1800)
    start_seconds: float = Field(ge=0, le=600)
    end_seconds: float = Field(gt=0, le=600)
    visible_labels: list[str] = Field(min_length=1, max_length=12)
    scene_direction: str = Field(min_length=1, max_length=2400)
    image_prompt: str = Field(min_length=1, max_length=4000)
    needs_hand: bool
    source_ids: list[str] = Field(min_length=1, max_length=12)
    limitations: list[str] = Field(max_length=10)


class Direction(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    answer_summary: str = Field(min_length=1, max_length=1600)
    timing_basis: Literal["estimated_until_tts"]
    grounding_limitations: list[str] = Field(max_length=20)
    scenes: list[Scene] = Field(min_length=5, max_length=10)

    @model_validator(mode="after")
    def ordered_timeline(self):
        previous_end = 0.0
        for expected, scene in enumerate(self.scenes, 1):
            if scene.order != expected or abs(scene.start_seconds - previous_end) > 0.01:
                raise ValueError("Scenes must be sequential, start at zero, and have no gaps")
            if scene.end_seconds <= scene.start_seconds:
                raise ValueError("Every scene needs a positive duration")
            previous_end = scene.end_seconds
        return self


def validate_grounding(plan: Direction, context: dict, count: int) -> None:
    if len(plan.scenes) != count:
        raise ValueError("Director did not return the requested scene count")
    sources = {hit["id"]: hit for hit in context["results"]}
    for scene in plan.scenes:
        if any(source_id not in sources for source_id in scene.source_ids):
            raise ValueError("Director returned an unknown citation ID")
        flagged = any(sources[item].get("quality_flags") for item in scene.source_ids)
        if flagged and not (scene.limitations or plan.grounding_limitations):
            raise ValueError("Director omitted extraction limitations for flagged sources")
