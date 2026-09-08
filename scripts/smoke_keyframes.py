"""Real HTTP acceptance run. Launch only after the final corrected corpus is active."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from talotrace.artifacts.store import inspect_png

QUESTIONS = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
]


async def run(base_url: str, prefix: str, expected_corpus: str) -> None:
    directory = Path("data/tests") / prefix
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "expected_corpus": expected_corpus,
        "jobs": [],
    }

    def save():
        (directory / "api-run.json").write_text(json.dumps(report, indent=2), "utf-8")

    async with httpx.AsyncClient(base_url=base_url, timeout=20) as client:
        for index, question in enumerate(QUESTIONS, 1):
            response = await client.post(
                "/v1/keyframe-jobs",
                headers={"Idempotency-Key": f"{prefix}-{index}"},
                json={"question": question, "keyframe_count": 5},
            )
            response.raise_for_status()
            job = response.json()
            report["jobs"].append(
                {
                    "question": question,
                    "id": job["id"],
                    "submit_http": 202,
                    "idempotency_key": f"{prefix}-{index}",
                }
            )
            print(f"SUBMITTED {index} {job['id']}", flush=True)
        save()
        previous = {}
        remaining = {job["id"] for job in report["jobs"]}
        while remaining:
            for job_id in list(remaining):
                response = await client.get(f"/v1/keyframe-jobs/{job_id}/result")
                response.raise_for_status()
                result = response.json()
                status = (result["state"], result["completed_keyframes"], bool(result["direction"]))
                if status != previous.get(job_id):
                    print(f"STATUS {job_id} {status}", flush=True)
                    previous[job_id] = status
                if result["retrieval"] and result["retrieval"]["corpus_id"] != expected_corpus:
                    raise RuntimeError("Live run used an unexpected corpus")
                entry = next(item for item in report["jobs"] if item["id"] == job_id)
                entry.update(
                    {
                        "state": result["state"],
                        "completed_keyframes": status[1],
                        "result_http": response.status_code,
                        "error": result["error"],
                    }
                )
                save()
                if result["state"] in {"failed", "interrupted"}:
                    raise RuntimeError(f"Job {job_id} stopped: {result['error']}")
                if result["state"] == "keyframes_completed":
                    entry["images"] = []
                    for frame in result["keyframes"]:
                        image = await client.get(frame["artifact_url"])
                        image.raise_for_status()
                        details = inspect_png(image.content)
                        if details != frame["image"]:
                            raise RuntimeError("Downloaded image differs from receipt")
                        entry["images"].append({"order": frame["order"], "http": 200, **details})
                    repeated = await client.post(
                        "/v1/keyframe-jobs",
                        headers={"Idempotency-Key": entry["idempotency_key"]},
                        json={"question": entry["question"], "keyframe_count": 5},
                    )
                    repeated.raise_for_status()
                    assert repeated.json()["id"] == job_id
                    assert repeated.json()["state"] == "keyframes_completed"
                    entry["idempotent_replay_same_completed_job"] = True
                    remaining.remove(job_id)
                    save()
            if remaining:
                await asyncio.sleep(5)
        listing = (await client.get("/v1/keyframe-jobs?limit=100")).json()
        report["list_contains_all_jobs"] = all(
            any(row["id"] == entry["id"] for row in listing["jobs"]) for entry in report["jobs"]
        )
        report["completed_at"] = datetime.now(UTC).isoformat()
        report["video_generated"] = False
        save()
        print(f"COMPLETE {directory / 'api-run.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix", default="assessment-keyframes-v1")
    parser.add_argument("--expected-corpus", required=True)
    args = parser.parse_args()
    if not args.prefix.replace("-", "").replace("_", "").isalnum():
        parser.error("prefix must contain only letters, digits, dashes and underscores")
    asyncio.run(run(args.base_url, args.prefix, args.expected_corpus))
