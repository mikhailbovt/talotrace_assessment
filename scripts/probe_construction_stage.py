"""Unvoiced, reviewed blank-to-beaker primitive; never a complete chemistry lesson."""

import argparse
import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from talotrace.artifacts.construction import load_reviewed_stage
from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.artifacts.video import command, inspect_clip
from talotrace.config import Settings
from talotrace.jobs.video_worker import VIDEO_WORKER_LOCK
from talotrace.providers.ltx import ComfyClient
from talotrace.providers.ltx_graph import build_graph, render_settings
from talotrace.providers.openai_media import reference_metadata


async def main(args):
    settings = Settings()
    artifacts = ArtifactStore(settings.artifacts_dir)
    reviewed, files = load_reviewed_stage(
        args.manifest, args.manifest_sha256, settings.artifacts_dir
    )
    render = render_settings()
    profile = render["profile"] | {"draw_frames": reviewed.stage.frames}
    source = {
        "scope": "unvoiced_construction_primitive_not_complete_lesson",
        "manifest": reviewed.model_dump(mode="json"),
        "manifest_sha256": args.manifest_sha256,
        "render_settings": render,
        "effective_profile": profile,
        "experiment_overrides": {"draw_frames": 97, "matched_blank": True},
        "hand_reference": reference_metadata(settings.hand_reference_path, "hand_reference"),
    }
    run_id = str(args.run_id or uuid4())
    saved = artifacts.read_json(run_id, "construction-source.json")
    if saved and saved != source:
        raise ValueError("Construction probe inputs changed; use a new run UUID")
    artifacts.write_json(run_id, "construction-source.json", source)
    for role in ("blank", "outline"):
        artifacts.write(run_id, f"guide-{role}.png", files[role].read_bytes())
        artifacts.write(run_id, f"guide-{role}.receipt.json", files[f"{role}_receipt"].read_bytes())
    print(f"Construction probe: {run_id}", flush=True)
    comfy = ComfyClient(
        settings.comfyui_url,
        artifacts,
        run_id,
        settings.video_comfy_timeout_seconds,
        settings.video_poll_seconds,
    )
    images = {
        "blank": await comfy.upload("matched-blank.png", files["blank"].read_bytes()),
        "keyframe": await comfy.upload("beaker-outline.png", files["outline"].read_bytes()),
        "hand_reference": await comfy.upload(
            "hand-reference.png", settings.hand_reference_path.read_bytes()
        ),
    }
    graph, conditioning = build_graph(
        mode="whiteboard_loop",
        phase="draw",
        scene={},
        images=images,
        profile=profile,
        seed=int(sha256(json_bytes(source))[:12], 16),
        output_prefix=f"tests/{run_id}/beaker-outline",
        prompt_override=reviewed.stage.prompt,
    )
    artifacts.write_json(run_id, "construction-preflight-graph.json", graph)
    async with comfy.client() as client:
        response = await client.get("/object_info")
        response.raise_for_status()
        registry = response.json()
    for node in graph.values():
        required = registry[node["class_type"]].get("input", {}).get("required", {})
        if set(required) - node["inputs"].keys():
            raise ValueError("Construction graph misses registered required inputs")
    artifacts.write_json(
        run_id,
        "construction-preflight.json",
        {
            "phase": "before_gpu_dispatch",
            "graph_sha256": sha256(json_bytes(graph)),
            "node_schema_validation": "passed",
            "conditioning": conditioning,
        },
    )
    if not args.render:
        print("Reviewed references uploaded and graph preflight passed; no GPU work submitted.")
        return
    # As in the scene probe, Windows needs Proactor for ffmpeg; use sync DB for this CLI lock.
    with psycopg.connect(
        settings.database_url.get_secret_value(),
        row_factory=dict_row,
        autocommit=True,
        connect_timeout=5,
    ) as conn:
        row = conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS acquired", (VIDEO_WORKER_LOCK,)
        ).fetchone()
        if not row["acquired"]:
            raise RuntimeError("Another video worker or scene probe still owns the GPU")
        data, dispatch = await comfy.render("construction-draw", graph, attempt=args.attempt)
        output = artifacts.write(run_id, "construction-draw.mp4", data)
        report = await inspect_clip(output, 97, 2048, 1152, profile["fps"], settings.ffprobe_path)
        await command([settings.ffmpeg_path, "-v", "error", "-i", str(output), "-f", "null", "-"])
        artifacts.write_json(
            run_id,
            "construction-result.json",
            {
                "scope": source["scope"],
                "manifest_sha256": args.manifest_sha256,
                "source_sha256": sha256(json_bytes(source)),
                "dispatch": dispatch,
                "conditioning": conditioning,
                "media": report | {"full_decode": "passed"},
                "visual_review": "pending_actual_stroke_and_geometry_review",
                "accepted_for_lesson": False,
            },
        )
        print(f"Unvoiced construction probe completed: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run-id", type=UUID)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--render", action="store_true", help="Submit GPU work after reviewed preflight"
    )
    asyncio.run(main(parser.parse_args()))
