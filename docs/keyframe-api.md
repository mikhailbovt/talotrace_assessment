# Question-to-keyframes backend

The implemented boundary is question embedding → hybrid retrieval → structured direction →
5–10 style-referenced keyframes. Completion is `keyframes_completed`; no video, audio, or
measured speech timing is produced. The API runs with `docker compose up -d --build api`.
All generated test media and run manifests are stored under `data/tests/<job UUID>/` on the
host, including when the API runs in Docker. The original reference images remain versioned
under `assets/references/`.

## HTTP endpoints

| Method and path | Behavior |
| --- | --- |
| `POST /v1/keyframe-jobs` | Validates input, persists a job, and returns HTTP202 promptly |
| `GET /v1/keyframe-jobs?limit=20&offset=0` | Lists saved jobs newest first |
| `GET /v1/keyframe-jobs/{id}` | Current state, completed frame count, errors and links |
| `GET /v1/keyframe-jobs/{id}/events` | Live SSE snapshots, heartbeat comments and terminal event |
| `GET /v1/keyframe-jobs/{id}/result` | Available retrieval evidence, direction, and frame receipts |
| `GET /v1/keyframe-jobs/{id}/images/{order}` | Integrity-checked PNG for a completed frame |
| `POST /v1/keyframe-jobs/{id}/resume` | Explicitly resumes a failed/interrupted job, up to three attempts |

Submission requires an `Idempotency-Key` header of 8–128 characters and JSON:

```json
{"question":"How does the pH scale work?","keyframe_count":5}
```

Accepted and resumed jobs include `Location` and `Retry-After: 2` headers. The body includes
`status_url`, `events_url`, a named `stage`, frame `progress`, `terminal`, and `can_resume`.
The durable worker owns generation independently of the submitter or any status stream.
See [job status and streaming](job-status.md) for SSE semantics, error behavior, and read-only
verification commands. Existing polling clients remain supported.

Submission returns HTTP503 until the active corpus declares the verified
`pdf-inspector+gpt-5.6-luna-vision-v1` extraction policy. The worker repeats that check before
query embedding and checks saved evidence on resume; legacy pypdf answer fallback is rejected.

Reusing a key with identical inputs returns the same saved job, including after completion.
Reusing it with different inputs or generation settings returns HTTP409. A failed job is not
automatically restarted by a repeated submission; use its resume endpoint. Invalid input is
HTTP422. Unknown jobs and unavailable frames are HTTP404. The result endpoint also exposes
partial progress while work runs. Image paths are generated from validated UUIDs and integer
scene numbers; the client cannot select arbitrary files.

This is a local assessment service, bound to localhost through Compose. It has no authentication
or per-user authorization and must not be exposed as a public multi-tenant API as-is.

## Queue, recovery, and persistence

PostgreSQL migration `infra/postgres/004-jobs.sql` is applied at startup to both existing and
new databases. PostgreSQL holds the queue, request/settings hashes, state, retrieval context,
validated direction, frame receipts, error category, and attempt count. The API process runs
one asynchronous worker; a database advisory lock bounds execution to one job across processes.
The worker verifies its lock connection during provider work and cancels if that connection fails.
It releases the lock at shutdown. A replacement worker marks unfinished jobs `interrupted`,
requiring explicit resume rather than silently billing for an ambiguous in-flight request.

Each job gets its own retriever and async database connection. The retriever embeds the actual
question using `text-embedding-3-large` at3072 dimensions or reuses its exact-hash cached query
embedding. It retrieves eight passages using exact cosine search plus BM25 and reciprocal-rank
fusion. The saved context identifies query embedding hash/cache state, active corpus, source
file hashes, page numbers, chunk IDs/content hashes, parser identity and extraction warnings.
The source context is immutable for that job even if the active corpus subsequently changes.

Provider calls use only the explicitly configured project key and fixed official API URL.
There is no silent model fallback and no automatic SDK retry. A provider failure marks the
job failed with a safe error type, HTTP status, request ID and provider code when available.
Provider request bodies and exception text are not logged.

Validated direction and each successful PNG/receipt are written atomically before updating DB
progress. Resume reuses these outputs and checks their input identity, PNG decoding, dimensions,
and checksum. Missing or corrupt saved media fails clearly rather than regenerating it silently.
This avoids repeat charges for durably saved outputs. A process/network failure after the
provider charges but before its result is durably saved can still require another charged call;
exactly-once external billing cannot be guaranteed. Jobs allow at most three explicit attempts.

## Direction and image contract

The director is `gpt-5.6-luna`, using the Responses API with strict structured output and high
reasoning effort. Its schema requires ordered continuous estimated timestamps, scene objectives,
narration, exact visible labels, scene directions, image prompts, source IDs, and limitations.
The service rejects unknown citations, wrong scene counts, invalid timelines, and missing
limitations for cited sources with extraction warnings. Matching citation IDs proves traceability;
it does not itself prove that every generated scientific statement is supported.

Each image uses `gpt-image-2` through the Images edit endpoint with the original style-anchor
bytes attached. The model uses high-fidelity image input automatically; `input_fidelity` is
deliberately omitted. Output is one high-quality2048×1152 PNG per scene. A hand reference is
attached only when `needs_hand=true`. Every receipt records the actual reference file sizes and
SHA256 hashes, prompt, settings, request ID, usage when returned, and decoded PNG dimensions.
The service validates files, but visual and scientific acceptance requires reviewing actual images.

The anchor supplies artistic style, not chemistry authority. Prompts preserve its whiteboard,
black marker, handwritten labels and restrained colored accents while requesting clear,
scene-specific diagrams. The supplied pH formulas have source defects; the director must use
supported prose and disclose limitations rather than invent corrections. Timestamps remain
`estimated_until_tts` until a later stage measures actual speech.

Official contracts checked during implementation:

- [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation)
- [GPT Image2 model](https://developers.openai.com/api/docs/models/gpt-image-2)
- [Structured output](https://developers.openai.com/api/docs/guides/structured-outputs)

## Verification

Run deterministic validation and recovery tests:

```powershell
uv run pytest tests/test_keyframes.py -q
```

After the final corrected corpus has been activated, the live acceptance client submits the three
assessment questions with five frames each and checks image downloads, status, listing, and
idempotent replay. Use the actual activated corpus ID; this prevents silently accepting an older
snapshot in the result:

```powershell
uv run python scripts/smoke_keyframes.py --expected-corpus <active-corpus-id>
```

The client writes `data/tests/assessment-keyframes-v1/api-run.json`. Each UUID directory contains
`request.json`, `context.json`, `direction.json`, `scene-XX.png`, per-scene receipts and the final
`manifest.json`. The default idempotency prefix deliberately reuses an existing run. Choose a new
`--prefix` only when a new paid run is intended. Live results and visual QA are recorded separately
from mocked recovery tests. No keyframes are yet claimed as reviewed in this document.
