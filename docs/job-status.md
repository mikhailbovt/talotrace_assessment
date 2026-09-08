# Asynchronous requests and live job status

The FastAPI handlers validate a request and commit its durable PostgreSQL job before returning
HTTP202. They do not await OpenAI, Kokoro, or LTX generation. The independent workers claim
persisted jobs and update their state. Closing the submitter connection after persistence, or
closing any status connection, does not cancel the job. If a submitter loses the acknowledgement,
repeat the same request with the same `Idempotency-Key` to recover the existing job ID.

Both `/v1/keyframe-jobs` and `/v1/video-jobs` return `status_url`, `events_url` and `result_url`.
Accepted and resumed responses set `Location` to the status URL and `Retry-After: 2` as a
suggested polling interval. Input, corpus, idempotency and recovery validation remain unchanged.

New video requests default to `whiteboard_loop`, the only accepted generation mode. Explicit
`references` submissions return HTTP422 before persistence; existing reference experiment jobs
keep their stored mode and remain available through list/status/SSE/result/download endpoints.
Omitted and explicit `whiteboard_loop` values normalize to the same idempotent request identity.
Unfinished historical reference jobs also reject resume with HTTP409 and report
`can_resume=false`. A completed historical job can still return its existing result from
resume without dispatching generation.

## Polling and progress

`GET /v1/{keyframe|video}-jobs/{id}` returns the latest persisted state. Keyframe status includes
`stage.code`, `stage.label`, and `progress={unit: keyframes, completed, total}`. This count is a
keyframe count, not an invented whole-pipeline completion percentage or an estimated finish time.

Video status retains the worker's actual per-scene/per-clip `progress`, and includes a nested
`upstream` keyframe status. `effective_stage` follows the upstream while a video waits, then the
Kokoro/rendering/assembly stage. The persisted `renderer` identifies local keyframe compositing
or LTX, and the rendering label follows that choice. `/health/live` exposes the deployed
`video_renderer` and `video_worker_enabled` configuration for read-only preflight checks.
`video_worker_enabled=false` is visible; once keyframes are ready,
`blocking_reason=video_worker_disabled` explains why rendering has not started. It describes
configuration rather than claiming that a worker heartbeat or the GPU is healthy.

`terminal` is true for `keyframes_completed`, `video_completed`, `failed`, or `interrupted`.
Failed/interrupted jobs can still be resumed explicitly if they have attempts remaining. On the
video status endpoint, `can_resume=false` and `blocking_reason=upstream_requires_resume` mean
the keyframe job must be resumed first. Errors retain the saved safe category and failed stage.

## Server-sent events

`GET /v1/keyframe-jobs/{id}/events` and `GET /v1/video-jobs/{id}/events` return
`Content-Type: text/event-stream`, `Cache-Control: no-cache, no-transform`,
`X-Accel-Buffering: no`, and `X-Event-Semantics: current-snapshot`.

The stream sends:

1. An immediate `status` event containing the current polling response and `retry: 2000`.
2. A new `status` when the persisted state/progress changes. Video upstream changes are included
   even while the video row itself remains queued. The observer reads at one-second intervals.
3. A `: heartbeat` comment after approximately 15 seconds without a changed snapshot.
4. A final `completed`, `failed`, or `interrupted` event with the final snapshot, then closes.

These are snapshots, not a durable event history. Intermediate states that change between reads
can be skipped. No event IDs are emitted; `Last-Event-ID` is ignored. Reconnecting always starts
from the current state. EventSource clients should explicitly close on a terminal event to avoid
automatically reconnecting to an already terminal job. If that job is later resumed, open a new
observer to follow its new attempt.

Invalid UUIDs return HTTP422 and unknown IDs HTTP404 before streaming headers are sent. A
database failure before streaming returns safe HTTP503 with `Retry-After: 2`. After headers are
sent, a read failure or ten-second read timeout emits `stream_error` and closes:

```text
event: stream_error
data: {"code":"status_unavailable","detail":"The status stream closed; poll or reconnect for the latest state","retryable":true,"generation_cancelled":false}
```

This is an observer failure, not a change to the job's generation state. If a previously visible
job no longer exists, the code is `job_not_found` and `retryable=false`. Exception messages,
database connection strings, and provider credentials are not serialized into these events.

Every database read closes its connection before the observer sleeps. The stream retains no
transaction or producer task. Disconnect cancellation propagates through a pending read and
ends only that observer. Reference/configuration reads and artifact checksum verification run
in threads so filesystem work does not block the HTTP event loop. MP4 checksum verification
uses bounded-memory reads; `FileResponse` supplies range requests for video playback/download.

## Verification

```powershell
uv run pytest tests/test_api_async.py tests/test_keyframes.py tests/test_video.py -q
uv run ruff check src/talotrace/api tests/test_api_async.py
```

`test_api_async.py` starts real Uvicorn HTTP listeners on ephemeral loopback ports with isolated
in-memory stores, accelerated test-only observer intervals, and no provider or generation calls.
It verifies prompt HTTP202 before any worker runs, concurrent polling, idempotent retries,
submitter disconnect after persistence, actual SSE flushing, upstream progress, terminal events,
resume limits, heartbeats, cancellation of a pending observer read, safe storage failures, and
responsive status reads while a video checksum is deliberately blocked. Its synthetic transport
file is explicitly not a generated video. Any test files are under `data/tests/unit-api/`.
The API tests also verify the whiteboard-only default and OpenAPI schema, rejection of the
retired reference mode before either job is created, and preserved historical reference reads,
terminal status streams, and downloads.

These tests exercise actual HTTP transport and async cancellation, while substituting the
persistence boundary. They do not certify a live PostgreSQL deployment, GPU generation, or the
visual/scientific quality of a completed video.

After a controlled rebuild, use existing job IDs for read-only deployment verification. These
requests do not create new paid work:

```powershell
$keyframeJobId = '00468651-fe4a-4223-9ae3-0056403ef777'
Invoke-RestMethod "http://127.0.0.1:8000/v1/keyframe-jobs/$keyframeJobId"
curl.exe --no-buffer --max-time 20 "http://127.0.0.1:8000/v1/keyframe-jobs/$keyframeJobId/events"
```

For an existing video job, make the corresponding calls to `/v1/video-jobs/{id}` and
`/v1/video-jobs/{id}/events`. Confirm real upstream/render progress and read `/result` when it
completes. A completed stream sends `status`, then `completed`, then EOF. A running stream may
hit curl's deliberate 20-second timeout; confirm that the job continues via a separate polling
request. Store live verification reports under `data/tests/` and distinguish them from these
isolated HTTP contract tests. Deploying this change must wait until active provider work is
safe to interrupt; API restart is deliberately not part of the test commands above.
