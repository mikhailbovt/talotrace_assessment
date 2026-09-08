"""Live HTTP acceptance for a whiteboard lesson using generated, reviewed keyframes."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import UUID

import httpx

from talotrace.artifacts.store import json_bytes, sha256


async def main(args):
    directory = Path("data/tests") / args.prefix
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    submissions = []
    modes = [args.mode]
    async with httpx.AsyncClient(base_url=args.api_url, timeout=60, trust_env=False) as client:
        health = await client.get("/health/live")
        health.raise_for_status()
        deployed = health.json()
        if args.expected_renderer:
            if deployed.get("video_renderer") != args.expected_renderer:
                raise RuntimeError(
                    "Deployed renderer does not match expected renderer; no job submitted"
                )
            if deployed.get("video_worker_enabled") is not True:
                raise RuntimeError("Video worker is disabled; no job submitted")
        for mode in modes:
            body = {"keyframe_job_id": str(args.keyframe_job_id), "mode": mode}
            key = f"{args.prefix}:{args.keyframe_job_id}:{mode}"
            submitted_at = time.perf_counter()
            response = await client.post(
                "/v1/video-jobs", json=body, headers={"Idempotency-Key": key}
            )
            response.raise_for_status()
            record = response.json()
            elapsed = time.perf_counter() - submitted_at
            if (
                response.status_code != 202
                or response.headers.get("location") != record["status_url"]
            ):
                raise RuntimeError("Submission did not return HTTP202 with the job status Location")
            renderer = record.get("renderer", {}).get("name")
            if args.expected_renderer and renderer != args.expected_renderer:
                raise RuntimeError("Persisted job renderer does not match the expected renderer")
            submissions.append(
                {
                    "job_id": record["id"],
                    "http": response.status_code,
                    "seconds": elapsed,
                    "location": response.headers["location"],
                    "retry_after": response.headers.get("retry-after"),
                    "renderer": renderer,
                }
            )
            records.append(record)
            repeat = await client.post(
                "/v1/video-jobs", json=body, headers={"Idempotency-Key": key}
            )
            repeat.raise_for_status()
            if (
                repeat.status_code != 202
                or repeat.headers.get("location") != record["status_url"]
                or repeat.json()["id"] != record["id"]
            ):
                raise RuntimeError("Idempotency replay created duplicate video work")
            print(
                json.dumps({"mode": mode, "job_id": record["id"], "state": record["state"]}),
                flush=True,
            )
        manifest = {
            "scope": "full_lesson_api_acceptance",
            "keyframe_job_id": str(args.keyframe_job_id),
            "jobs": records,
            "submissions": submissions,
            "deployed": deployed,
            "expected_renderer": args.expected_renderer,
            "status_history": [],
            "idempotency_replay": "passed",
        }
        (directory / "api-run.json").write_bytes(json_bytes(manifest))
        if args.submit_only:
            return
        started = time.monotonic()
        previous = {}
        completed = set()
        while len(completed) < len(records):
            for record in records:
                job_id = record["id"]
                if job_id in completed:
                    continue
                response = await client.get(record["status_url"])
                response.raise_for_status()
                status = response.json()
                state = (status["state"], json.dumps(status["progress"], sort_keys=True))
                if previous.get(job_id) != state:
                    manifest["status_history"].append(
                        {
                            "job_id": job_id,
                            "state": status["state"],
                            "progress": status["progress"],
                            "stage": status.get("stage"),
                            "renderer": status.get("renderer"),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    (directory / "api-run.json").write_bytes(json_bytes(manifest))
                    print(
                        json.dumps(
                            {
                                "id": job_id,
                                "mode": status["mode"],
                                "state": status["state"],
                                "progress": status["progress"],
                            }
                        ),
                        flush=True,
                    )
                    previous[job_id] = state
                if status["state"] in {"failed", "interrupted"}:
                    (directory / f"{job_id}-failure.json").write_bytes(json_bytes(status))
                    raise RuntimeError(f"Video job {job_id} needs explicit investigation/resume")
                if status["state"] == "video_completed":
                    response = await client.get(record["result_url"])
                    response.raise_for_status()
                    result = response.json()
                    (directory / f"{job_id}-result.json").write_bytes(json_bytes(result))
                    result_renderer = result["result"]["settings"]["renderer"]["name"]
                    if args.expected_renderer and result_renderer != args.expected_renderer:
                        raise RuntimeError("Final manifest has an unexpected renderer")
                    if result["result"].get("renderer") != result["result"]["settings"]["renderer"]:
                        raise RuntimeError(
                            "Final renderer provenance differs from persisted settings"
                        )
                    review = result["result"]["narration_review"]["review_sha256"]
                    if args.expected_review_sha256 and review != args.expected_review_sha256:
                        raise RuntimeError(
                            "Video did not use the expected approved narration review"
                        )
                    response = await client.get(status["video_url"], timeout=180)
                    response.raise_for_status()
                    media = result["result"]["media"]
                    if sha256(response.content) != media["sha256"]:
                        raise RuntimeError(
                            "HTTP video download checksum differs from final manifest"
                        )
                    if (
                        media["audio_streams"] != 1
                        or media["native_ltx_audio_used"]
                        or media["background_music"]
                    ):
                        raise RuntimeError("Final audio policy failed")
                    (directory / f"{job_id}.mp4").write_bytes(response.content)
                    completed.add(job_id)
            if time.monotonic() - started > args.timeout_seconds:
                raise TimeoutError("Acceptance wait expired; jobs remain persisted and retrievable")
            if len(completed) < len(records):
                await asyncio.sleep(20)
        manifest["completed_job_ids"] = sorted(completed)
        manifest["visual_review"] = "requires_actual_frame_and_audio_review"
        (directory / "api-run.json").write_bytes(json_bytes(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-job-id", required=True, type=UUID)
    parser.add_argument("--mode", choices=("whiteboard_loop",), default="whiteboard_loop")
    parser.add_argument("--prefix", default="assessment-videos-v1")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-review-sha256")
    parser.add_argument("--expected-renderer", choices=("ltx", "keyframe-composite-v1"))
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.prefix
    ):
        parser.error("prefix must be a lowercase directory name")
    asyncio.run(main(args))
