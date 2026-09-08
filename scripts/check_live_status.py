"""Observe an existing API job over real HTTP/SSE without submitting or resuming work."""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx


async def check(args):
    route = f"/v1/{args.kind}-jobs/{args.job_id}"
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "kind": args.kind,
        "job_id": str(args.job_id),
        "scope": "deployed_http_current_status_observer",
        "generation_requests": 0,
        "events": [],
        "heartbeats": 0,
        "concurrent_status_reads": [],
    }
    async with httpx.AsyncClient(base_url=args.api_url, timeout=30, trust_env=False) as client:
        initial = await client.get(route)
        initial.raise_for_status()
        report["before"] = initial.json()

        async def poll():
            for _ in range(3):
                started = time.perf_counter()
                response = await client.get(route)
                response.raise_for_status()
                report["concurrent_status_reads"].append(
                    {"http": response.status_code, "seconds": time.perf_counter() - started}
                )
                await asyncio.sleep(0.2)

        async def stream():
            try:
                async with asyncio.timeout(args.seconds):
                    async with client.stream("GET", f"{route}/events") as response:
                        response.raise_for_status()
                        report["stream_http"] = response.status_code
                        report["content_type"] = response.headers.get("content-type")
                        report["semantics"] = response.headers.get("x-event-semantics")
                        assert "text/event-stream" in report["content_type"]
                        event = None
                        async for line in response.aiter_lines():
                            if line.startswith("event: "):
                                event = line[7:]
                            elif line.startswith("data: "):
                                report["events"].append(
                                    {"event": event, "data": json.loads(line[6:])}
                                )
                            elif line == ": heartbeat":
                                report["heartbeats"] += 1
                        report["stream_end"] = "server_eof"
            except TimeoutError:
                report["stream_end"] = "observer_disconnected_at_time_limit"

        await asyncio.gather(stream(), poll())
        after = await client.get(route)
        after.raise_for_status()
        report["after"] = after.json()
        assert report["events"] and report["events"][0]["event"] == "status"
        assert report["semantics"] == "current-snapshot"
        assert not any(item["event"] == "stream_error" for item in report["events"])
        if report["before"]["terminal"]:
            expected = report["before"]["state"]
            if expected in {"keyframes_completed", "video_completed"}:
                expected = "completed"
            assert report["events"][-1]["event"] == expected
            assert report["stream_end"] == "server_eof"
        else:
            assert report["after"]["state"] != "interrupted"
        report["passed"] = True
    output = Path("data/tests/api-refinement/live")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.kind}-{args.job_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "events": [event["event"] for event in report["events"]],
                "heartbeats": report["heartbeats"],
                "stream_end": report["stream_end"],
                "after_state": report["after"]["state"],
                "evidence": str(path),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("keyframe", "video"), required=True)
    parser.add_argument("--job-id", type=UUID, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=float, default=20)
    arguments = parser.parse_args()
    if not 1 <= arguments.seconds <= 60:
        parser.error("seconds must be between 1 and 60")
    asyncio.run(check(arguments))
