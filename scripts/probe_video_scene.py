"""Real single-scene acceptance probe before enabling a full lesson on the shared GPU."""

import argparse
import asyncio
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.artifacts.video import (
    assemble_final,
    assemble_scene,
    inspect_clip,
    make_timeline,
    narration_track,
    validate_final,
)
from talotrace.config import Settings
from talotrace.jobs.preconditions import validate_extracted_sources
from talotrace.jobs.schemas import Direction, validate_grounding
from talotrace.jobs.video_worker import VIDEO_WORKER_LOCK, video_settings
from talotrace.providers.kokoro import speak
from talotrace.providers.ltx import ComfyClient
from talotrace.providers.ltx_graph import build_graph, render_identity_settings


async def main(args):
    settings = Settings(video_renderer="ltx")
    artifacts = ArtifactStore(settings.artifacts_dir)
    # Windows needs Proactor for ffmpeg, whereas psycopg async requires Selector.
    # This isolated CLI uses a synchronous DB connection for its advisory lock.
    dsn = settings.database_url.get_secret_value()
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5) as conn:
        upstream = conn.execute(
            "SELECT * FROM pipeline.jobs WHERE id=%s", (args.keyframe_job_id,)
        ).fetchone()
    if not upstream or not upstream["direction"] or not upstream["context"]:
        raise ValueError("Validated upstream direction is not ready")
    validate_extracted_sources(upstream["context"])
    plan = Direction.model_validate(upstream["direction"]["plan"])
    validate_grounding(plan, upstream["context"], upstream["request"]["keyframe_count"])
    frame = next((v for v in upstream["frames"] if v["order"] == args.scene), None)
    if frame is None:
        raise ValueError("The selected real keyframe is not ready")
    upstream_id, run_id = str(args.keyframe_job_id), str(args.run_id or uuid4())
    run_dir = artifacts.directory(run_id)
    artifacts.frame_receipt(upstream_id, args.scene, frame["input_sha256"])
    scene = plan.scenes[args.scene - 1].model_dump()
    config = video_settings(settings)
    profile = config["profile"]
    source = {
        "scope": "single_scene_probe_not_complete_lesson",
        "mode": args.mode,
        "keyframe_job_id": upstream_id,
        "scene": scene,
        "frame": frame,
        "settings": config,
    }
    saved = artifacts.read_json(run_id, "probe-source.json")
    if saved and saved != source:
        raise ValueError("Probe inputs changed; use a new run ID")
    artifacts.write_json(run_id, "probe-source.json", source)
    render_source = source | {"settings": render_identity_settings(config)}
    print(f"Probe artifacts: {run_dir}", flush=True)
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True, connect_timeout=5) as conn:
        row = conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS acquired", (VIDEO_WORKER_LOCK,)
        ).fetchone()
        if not row["acquired"]:
            raise RuntimeError("The full video worker owns the shared GPU")
        comfy = ComfyClient(
            settings.comfyui_url,
            artifacts,
            run_id,
            settings.video_comfy_timeout_seconds,
            settings.video_poll_seconds,
        )
        keyframe_path = artifacts.path(upstream_id, f"scene-{args.scene:02d}.png")
        images = {
            "keyframe": await comfy.upload("keyframe.png", keyframe_path.read_bytes()),
            "hand_reference": await comfy.upload(
                "hand-reference.png", settings.hand_reference_path.read_bytes()
            ),
        }
        keyframe_ref = {"sha256": sha256(keyframe_path.read_bytes()), "order": args.scene}
        reference_record = {"keyframe": keyframe_ref, "hand_reference": config["hand_reference"]}
        artifacts.write_json(run_id, "references.json", reference_record)
        clips = {}
        for phase in ["draw", "erase"] if args.mode == "whiteboard_loop" else ["draw"]:
            name = f"scene-{args.scene:02d}-{phase}"
            graph, conditioning = build_graph(
                mode=args.mode,
                phase=phase,
                scene=scene,
                images=images,
                profile=profile,
                seed=int(sha256(json_bytes(render_source))[:12], 16),
                output_prefix=f"tests/{run_id}/{name}",
            )
            data, dispatch = await comfy.render(name, graph, attempt=args.attempt)
            path = artifacts.write(run_id, f"{name}.mp4", data)
            report = await inspect_clip(
                path,
                conditioning["frames"],
                profile["width"] * 2,
                profile["height"] * 2,
                profile["fps"],
                settings.ffprobe_path,
            )
            artifacts.write_json(
                run_id,
                f"{name}.json",
                {
                    "dispatch": dispatch,
                    "conditioning": conditioning,
                    "media": report,
                    "references": reference_record,
                },
            )
            clips[phase] = path
            print(f"Validated real {phase} clip: {report['frames']} frames", flush=True)
        audio, receipt = await speak(settings.kokoro_url, scene["narration"])
        speech_path = artifacts.write(run_id, "speech.wav", audio)
        artifacts.write_json(run_id, "speech.json", receipt)
        timeline = make_timeline([scene], [receipt], args.mode, profile, config["timing_policy"])
        artifacts.write_json(run_id, "timeline.json", timeline)
        narration = artifacts.write(
            run_id, "complete-kokoro-narration.wav", narration_track([speech_path], timeline)
        )
        segment = artifacts.path(run_id, "scene-assembled.mp4")
        await assemble_scene(
            draw=clips["draw"],
            keyframe=keyframe_path,
            erase=clips.get("erase"),
            output=segment,
            timeline=timeline["scenes"][0],
            profile=profile,
            ffmpeg=settings.ffmpeg_path,
        )
        final = artifacts.path(run_id, "final.mp4")
        await assemble_final(
            scenes=[segment],
            narration=narration,
            output=final,
            ffmpeg=settings.ffmpeg_path,
            timeline=timeline,
        )
        report = await validate_final(
            final, narration, timeline, profile, settings.ffmpeg_path, settings.ffprobe_path
        )
        artifacts.write_json(
            run_id,
            "probe-result.json",
            {
                "scope": "single_scene_probe_not_complete_lesson",
                "media": report,
                "visual_review": "pending",
                "mode": args.mode,
            },
        )
        print(f"Completed narrated single-scene probe: {final}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-job-id", type=UUID, required=True)
    parser.add_argument("--scene", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--mode", choices=("references", "whiteboard_loop"), required=True)
    parser.add_argument("--run-id", type=UUID)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), default=1)
    asyncio.run(main(parser.parse_args()))
