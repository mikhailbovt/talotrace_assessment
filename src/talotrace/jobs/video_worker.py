"""Durable continuation from validated keyframes through Kokoro and whiteboard LTX video."""

import asyncio
import logging

from talotrace.artifacts.composite import composite_settings, render_composite_clip
from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.artifacts.video import (
    assemble_final,
    assemble_scene,
    assembly_timing_policy,
    inspect_clip,
    make_timeline,
    narration_track,
    validate_final,
)
from talotrace.config import Settings
from talotrace.jobs.narration_review import effective_direction, selected_review
from talotrace.jobs.preconditions import validate_extracted_sources
from talotrace.jobs.schemas import validate_grounding
from talotrace.jobs.store import JobStore
from talotrace.jobs.video_store import VideoJobStore
from talotrace.jobs.worker import pipeline_settings
from talotrace.providers.kokoro import inspect_wav, speak
from talotrace.providers.ltx import ComfyClient
from talotrace.providers.ltx_graph import build_graph, render_identity_settings, render_settings
from talotrace.providers.openai_media import reference_metadata, safe_error

log = logging.getLogger(__name__)
VIDEO_WORKER_LOCK = 472930404


def video_settings(settings: Settings) -> dict:
    if settings.kokoro_voice != "af_heart":
        raise ValueError("This pipeline requires the selected Kokoro af_heart voice")
    result = {
        "pipeline_version": "narrated-video-v1",
        **render_settings(),
        "voice": settings.kokoro_voice,
        "speech_speed": 1.0,
        "timing_policy": assembly_timing_policy(),
        "style_anchor": reference_metadata(settings.style_anchor_path, "style_anchor"),
        "hand_reference": reference_metadata(settings.hand_reference_path, "hand_reference"),
        "renderer": {"name": "ltx", "provider": "runpod_comfyui", "new_ltx_generation": True},
    }
    if settings.video_renderer == "keyframe-composite-v1":
        result["renderer"], _ = composite_settings()
        result["assembly_profile"] = {
            "width": 1024,
            "height": 576,
            "fps": 24,
            "draw_frames": result["renderer"]["draw_frames"],
            "erase_frames": result["renderer"]["erase_frames"],
        }
    return result


class VideoPipeline:
    def __init__(
        self,
        settings: Settings,
        store: VideoJobStore,
        upstream_store: JobStore,
        artifacts: ArtifactStore,
    ):
        self.settings, self.store = settings, store
        self.upstream_store, self.artifacts = upstream_store, artifacts

    async def run(self, job: dict):
        job_id = str(job["id"])
        stage, scene_order = "preflight", None
        try:
            if job["request"]["mode"] != "whiteboard_loop":
                raise ValueError("Only whiteboard_loop is enabled for new generation")
            upstream_id = str(job["keyframe_job_id"])
            review = selected_review(self.artifacts, upstream_id)
            settings = video_settings(self.settings) | {
                "keyframe_settings": pipeline_settings(self.settings),
                "narration_review": review,
            }
            if settings != job["settings"]:
                raise ValueError("Video settings or reference files changed; use a new job")
            upstream = await self.upstream_store.get(upstream_id)
            if upstream is None or upstream["state"] != "keyframes_completed":
                raise ValueError("Upstream keyframe job failed or is not completed")
            validate_extracted_sources(upstream["context"])
            plan, narration_review = effective_direction(upstream["direction"], upstream_id, review)
            validate_grounding(plan, upstream["context"], upstream["request"]["keyframe_count"])
            source = {
                name: upstream[name]
                for name in ("request", "settings", "context", "direction", "frames")
            }
            source["narration_review"] = narration_review
            source_hash = sha256(json_bytes(source))
            saved_source = self.artifacts.read_json(job_id, "source.json")
            if saved_source and saved_source["sha256"] != source_hash:
                raise ValueError("Saved video source changed")
            self.artifacts.write_json(
                job_id,
                "source.json",
                {"keyframe_job_id": upstream_id, "sha256": source_hash, **source},
            )
            profile = settings.get("assembly_profile", settings["profile"])
            composite = settings["renderer"]["name"] == "keyframe-composite-v1"
            blank_path = composite_settings()[1] if composite else None
            mode = job["request"]["mode"]
            scenes = [scene.model_dump() for scene in plan.scenes]
            audio_receipts, audio_paths = [], []
            for scene in scenes:
                stage, scene_order = "synthesizing", scene["order"]
                await self.store.update(
                    job_id,
                    state=stage,
                    progress={
                        "scene_order": scene_order,
                        "scene_count": len(scenes),
                        "completed_narrations": len(audio_receipts),
                    },
                )
                name = f"scene-{scene_order:02d}-speech"
                receipt = self.artifacts.read_json(job_id, f"{name}.json")
                if receipt:
                    if receipt["text"] != scene["narration"] or receipt["voice"] != "af_heart":
                        raise ValueError("Saved narration differs from complete validated script")
                    data = self.artifacts.path(job_id, f"{name}.wav").read_bytes()
                    if inspect_wav(data) != receipt["audio"]:
                        raise ValueError("Saved narration failed integrity validation")
                else:
                    data, receipt = await speak(
                        self.settings.kokoro_url, scene["narration"], self.settings.kokoro_voice
                    )
                    self.artifacts.write(job_id, f"{name}.wav", data)
                    self.artifacts.write_json(job_id, f"{name}.json", receipt)
                audio_paths.append(self.artifacts.path(job_id, f"{name}.wav"))
                audio_receipts.append(receipt)
            timeline = make_timeline(
                scenes, audio_receipts, mode, profile, settings["timing_policy"]
            )
            self.artifacts.write_json(job_id, "timeline.json", timeline)
            await self.store.update(job_id, timeline=timeline)
            narration_path = self.artifacts.write(
                job_id, "complete-kokoro-narration.wav", narration_track(audio_paths, timeline)
            )
            self.artifacts.write_json(
                job_id,
                "narration.json",
                {
                    "scene_receipts": audio_receipts,
                    "timeline": timeline,
                    "full_script_submitted_to_tts": all(
                        a["text"] == s["narration"]
                        for a, s in zip(audio_receipts, scenes, strict=True)
                    ),
                    "audio": inspect_wav(narration_path.read_bytes()),
                },
            )

            comfy = (
                None
                if composite
                else ComfyClient(
                    self.settings.comfyui_url,
                    self.artifacts,
                    job_id,
                    self.settings.video_comfy_timeout_seconds,
                    self.settings.video_poll_seconds,
                )
            )
            hand_data = self.settings.hand_reference_path.read_bytes()
            clip_receipts, assembled_paths = [], []
            for scene, scene_timeline in zip(scenes, timeline["scenes"], strict=True):
                stage, scene_order = "rendering", scene["order"]
                frame = next(item for item in upstream["frames"] if item["order"] == scene_order)
                self.artifacts.frame_receipt(upstream_id, scene_order, frame["input_sha256"])
                keyframe_path = self.artifacts.path(upstream_id, f"scene-{scene_order:02d}.png")
                keyframe_data = keyframe_path.read_bytes()
                references = {
                    "keyframe": {
                        "sha256": sha256(keyframe_data),
                        "keyframe_job_id": upstream_id,
                        "order": scene_order,
                    },
                    "hand_reference": {"sha256": sha256(hand_data)},
                }
                if composite:
                    references["blank"] = settings["renderer"]["blank"]
                clips = {}
                for phase in ["draw", "erase"] if mode == "whiteboard_loop" else ["draw"]:
                    name = f"scene-{scene_order:02d}-{phase}"
                    await self.store.update(
                        job_id,
                        state=stage,
                        progress={
                            "scene_order": scene_order,
                            "scene_count": len(scenes),
                            "phase": phase,
                            "completed_clips": len(clip_receipts),
                        },
                    )
                    identity = sha256(
                        json_bytes(
                            {
                                "scene": scene,
                                "references": references,
                                "settings": render_identity_settings(settings),
                                "mode": mode,
                                "phase": phase,
                            }
                        )
                    )
                    receipt = self.artifacts.read_json(job_id, f"{name}.json")
                    path = self.artifacts.path(job_id, f"{name}.mp4")
                    expected_frames = profile["erase_frames" if phase == "erase" else "draw_frames"]
                    if receipt:
                        if receipt["input_sha256"] != identity:
                            raise ValueError("Saved video clip input changed")
                        media = await inspect_clip(
                            path,
                            expected_frames,
                            profile["width"] * 2,
                            profile["height"] * 2,
                            profile["fps"],
                            self.settings.ffprobe_path,
                        )
                        if media != receipt["media"]:
                            raise ValueError("Saved video clip failed integrity validation")
                    elif composite:
                        partial = self.artifacts.path(job_id, f"{name}.partial.mp4")
                        rendering = await render_composite_clip(
                            phase=phase,
                            keyframe=keyframe_path,
                            blank=blank_path,
                            hand=self.settings.hand_reference_path,
                            output=partial,
                            profile=profile,
                            policy=settings["renderer"],
                            ffmpeg=self.settings.ffmpeg_path,
                        )
                        media = await inspect_clip(
                            partial,
                            expected_frames,
                            profile["width"] * 2,
                            profile["height"] * 2,
                            profile["fps"],
                            self.settings.ffprobe_path,
                        )
                        partial.replace(path)
                        receipt = {
                            "input_sha256": identity,
                            "references": references,
                            "rendering": rendering,
                            "media": media,
                            "dispatch": {
                                "provider": "local_ffmpeg",
                                "state": "completed",
                                "new_ltx_generation": False,
                            },
                        }
                        self.artifacts.write_json(job_id, f"{name}.json", receipt)
                    else:
                        images = {
                            "keyframe": await comfy.upload("keyframe.png", keyframe_data),
                            "hand_reference": await comfy.upload("hand-reference.png", hand_data),
                        }
                        graph, conditioning = build_graph(
                            mode=mode,
                            phase=phase,
                            scene=scene,
                            images=images,
                            profile=profile,
                            seed=int(identity[:12], 16),
                            output_prefix=f"tests/{job_id}/{name}",
                        )
                        data, dispatch = await comfy.render(name, graph, attempt=job["attempts"])
                        self.artifacts.write(job_id, f"{name}.mp4", data)
                        media = await inspect_clip(
                            path,
                            expected_frames,
                            profile["width"] * 2,
                            profile["height"] * 2,
                            profile["fps"],
                            self.settings.ffprobe_path,
                        )
                        receipt = {
                            "input_sha256": identity,
                            "references": references,
                            "conditioning": conditioning,
                            "dispatch": dispatch,
                            "media": media,
                        }
                        self.artifacts.write_json(job_id, f"{name}.json", receipt)
                    clips[phase] = path
                    clip_receipts.append(receipt)
                stage = "assembling"
                await self.store.update(
                    job_id,
                    state=stage,
                    progress={
                        "scene_order": scene_order,
                        "scene_count": len(scenes),
                        "completed_clips": len(clip_receipts),
                    },
                )
                part = self.artifacts.path(job_id, f"scene-{scene_order:02d}-assembled.partial.mp4")
                await assemble_scene(
                    draw=clips["draw"],
                    erase=clips.get("erase"),
                    keyframe=keyframe_path,
                    output=part,
                    timeline=scene_timeline,
                    profile=profile,
                    ffmpeg=self.settings.ffmpeg_path,
                )
                completed = self.artifacts.path(job_id, f"scene-{scene_order:02d}-assembled.mp4")
                part.replace(completed)
                assembled_paths.append(completed)
            scene_order, stage = None, "assembling"
            final_partial = self.artifacts.path(job_id, "final.partial.mp4")
            await assemble_final(
                scenes=assembled_paths,
                narration=narration_path,
                output=final_partial,
                ffmpeg=self.settings.ffmpeg_path,
                timeline=timeline,
            )
            media = await validate_final(
                final_partial,
                narration_path,
                timeline,
                profile,
                self.settings.ffmpeg_path,
                self.settings.ffprobe_path,
            )
            final_partial.replace(self.artifacts.path(job_id, "final.mp4"))
            result = {
                "job_id": job_id,
                "state": "video_completed",
                "mode": mode,
                "keyframe_job_id": upstream_id,
                "source_sha256": source_hash,
                "settings": settings,
                "renderer": settings["renderer"],
                "narration_review": narration_review,
                "timeline": timeline,
                "clips": clip_receipts,
                "media": media,
                "visual_review": "pending_visual_and_scientific_review",
                "video_url": f"/v1/video-jobs/{job_id}/video",
            }
            self.artifacts.write_json(job_id, "video-manifest.json", result)
            await self.store.update(
                job_id,
                state="video_completed",
                result=result,
                error=None,
                progress={"completed_scenes": len(scenes), "completed_clips": len(clip_receipts)},
            )
        except asyncio.CancelledError:
            await self.store.update(
                job_id,
                state="interrupted",
                error={
                    "category": "WorkerInterrupted",
                    "failed_stage": stage,
                    "scene_order": scene_order,
                },
            )
            raise
        except Exception as error:
            details = safe_error(error) | {"failed_stage": stage, "scene_order": scene_order}
            self.artifacts.write_json(job_id, "video-error.json", details)
            await self.store.update(job_id, state="failed", error=details)
            log.warning("Video job %s failed: %s", job_id, details["category"])


class VideoWorker:
    def __init__(self, pipeline: VideoPipeline):
        self.pipeline, self.store = pipeline, pipeline.store
        self.task = None
        self.active = False

    def start(self):
        self.task = asyncio.create_task(self.run(), name="video-job-worker")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def run(self):
        while True:
            try:
                async with await self.store.connect() as conn:
                    row = await (
                        await conn.execute(
                            "SELECT pg_try_advisory_lock(%s) AS acquired", (VIDEO_WORKER_LOCK,)
                        )
                    ).fetchone()
                    if not row["acquired"]:
                        await asyncio.sleep(3)
                        continue
                    self.active = True
                    await self.store.mark_interrupted(conn)
                    while True:
                        job = await self.store.claim(conn)
                        if not job:
                            await asyncio.sleep(1)
                            continue
                        execution = asyncio.create_task(self.pipeline.run(job))
                        try:
                            while not execution.done():
                                done, _ = await asyncio.wait({execution}, timeout=5)
                                if not done:
                                    await conn.execute("SELECT 1")
                            await execution
                        finally:
                            if not execution.done():
                                execution.cancel()
                                try:
                                    await execution
                                except asyncio.CancelledError:
                                    pass
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log.warning("Video worker unavailable: %s", type(error).__name__)
                await asyncio.sleep(3)
            finally:
                self.active = False
