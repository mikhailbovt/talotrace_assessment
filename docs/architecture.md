# Architecture note

The backend separates HTTP requests, durable job orchestration, provider calls and media files.
The same job API serves the three required chemistry questions; a learner UI is out of scope.

## Job lifecycle

`POST /v1/video-jobs` commits a video job and its upstream keyframe job in one PostgreSQL
transaction, then returns HTTP 202 with status, events and result URLs. A prior keyframe job
can be reused instead of creating another.

```text
Keyframes: queued → retrieving → directing → generating_keyframes → keyframes_completed
Video:     waiting_for_keyframes → synthesizing → rendering → assembling → video_completed
Recovery:  failed / interrupted → explicit resume → queued
```

Async workers run outside HTTP handlers. A PostgreSQL advisory lock per worker type prevents
competing owners; provider calls do not hold a database transaction open. Status/list endpoints
read persisted progress; SSE sends current snapshots, not replayable history. Disconnecting an
observer leaves the job running. Recovery marks unfinished work interrupted. Idempotency keys
bind requests/settings; changed inputs return 409. Explicit resume verifies saved artifacts,
with at most three attempts; a failed upstream job must be resumed first.

## Persistence and artifact boundary

| Storage | Owner and contents |
| --- | --- |
| PostgreSQL | `jobs/store.py`, `jobs/video_store.py`: identities, states, attempts, progress, source context, plans, receipts and result metadata; retrieval tables hold corpus snapshots and embeddings. |
| Filesystem | `artifacts/store.py`: PNG/WAV/MP4 bytes, scripts, source snapshots and manifests under `data/tests/{job_id}/`, mounted as `/app/artifacts`. |
| Repository | Versioned references/configuration and model lockfiles; selected videos and learner questions in `examples/videos/`. Original PDFs stay outside the checkout, mounted read-only. |

Temporary writes and atomic replacement protect individual files; hashes bind receipts to bytes.
Final media is validated before the database job becomes complete, and downloads check integrity.
Database/file writes are not one atomic transaction: recovery reconciles saved artifacts with
metadata, so both storage locations must be preserved. Redis is provisioned but is not the queue.

## AI and video-generation boundary

`retrieval/` embeds questions and combines exact cosine search with BM25/rank fusion, returning
page citations and extraction warnings. `providers/` owns model calls: GPT-5.6 Luna returns
validated direction/narration; GPT Image2 generates style-anchored keyframes. Offline preparation
uses pdf-inspector plus recorded Luna visual transcription for failed pages.

`jobs/video_worker.py` consumes the saved plan/images. Kokoro speaks every complete scene;
measured WAV samples determine timing. `artifacts/composite.py` implements the selected local
renderer: blank → hand/reveal → exact keyframe/showing → erase → next scene. The optional
Comfy/LTX adapter supplies GPU clips behind the same orchestration boundary. Renderer identity
is explicit; the three final examples used composition without new LTX inference.

`artifacts/video.py` assembles H.264/AAC and verifies frames, dimensions, full decode and speech
preservation. Only Kokoro audio is retained, without music or native LTX audio. Schemas, grounding
checks, hashes and cache reuse reduce inconsistent output. Scientific/visual review is separate
from file validity: all 15 selected frames were reviewed; automatic checks alone do not certify
every newly generated diagram.

## Tradeoffs and extension

Composition preserves approved diagrams and avoids repeated GPU sampling costs. Fresh jobs
pay for embeddings, direction, images and hosting; receipts retain usage/model identities.
The measured five-keyframe pH run is estimated at [about $1.25](costs.md), including allocated pod time.
Cached resumes reuse completed assets. A lost response after billing can still cause a charged
retry; exactly-once external billing is not guaranteed. Other STEM topics can replace the corpus
and direction prompts while retaining the job, artifact and renderer contracts.

See [verified outcomes](setup-status.md), [operating details](setup.md) and [submission notes](submission.md).
