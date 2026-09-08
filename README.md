# Talotrace: Chemistry Video Request Service

FastAPI backend for chemistry questions → grounded scripts → narrated whiteboard videos.
Real OpenAI retrieval/direction/images and Kokoro speech feed a selectable video renderer.
The three selected examples use local keyframe composition with a moving hand; LTX-2.5 is
also integrated. No frontend is included.

**[Best videos and learner questions](examples/videos/)** · **[Architecture note](docs/architecture.md)**

## Setup

Requires Docker Compose with Linux containers, Python 3.12, uv, Git, OpenSSH and PowerShell 7.
Host-side media tests need FFmpeg/FFprobe; the API image includes them.

```powershell
git clone https://github.com/mikhailbovt/talotrace_assessment.git
Set-Location talotrace_assessment
uv sync --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Run all remaining commands from the repository root.

1. Edit `.env`: set `OPENAI_API_KEY`, `POSTGRES_PASSWORD` and a matching `DATABASE_URL`.
   Use a URL-safe database password. Keep the configured model IDs and `KOKORO_VOICE=af_heart`.
   Place the supplied PDFs at `../material/chem_material/Individual/`; visual references are included.
2. Start storage, then complete the one-time [corpus preparation commands](docs/rag.md#reproduce-from-the-repository-root).
   They run pdf-inspector, recorded visual fallback, embeddings and corpus activation.

   ```powershell
   docker compose up -d --wait --wait-timeout 120 postgres redis
   ```

3. Follow [RunPod setup](docs/runpod.md) for Kokoro/LTX services, your Hugging Face token and SSH key.
   Update `infra/runpod/pod.json` and keep the tunnel open:

   ```powershell
   pwsh -File scripts/runpod-tunnel.ps1
   ```

Set `VIDEO_RENDERER=keyframe-composite-v1` and `VIDEO_WORKER_ENABLED=true` in `.env`.
Use `VIDEO_RENDERER=ltx` for GPU clip generation. `.env.local`, when present, overrides `.env`
for the API. Credentials are ignored by Git. Full configuration: [setup guide](docs/setup.md).

## Run

```powershell
docker compose up -d --build --wait --wait-timeout 120
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Open [interactive API docs](http://127.0.0.1:8000/docs). Readiness checks storage; generation also
needs an activated corpus and reachable providers. `docker compose down` stops local services
while preserving database volumes; stop remote resources separately when finished.

## API

| Method and path | Purpose |
| --- | --- |
| `POST /v1/video-jobs` | Submit a question or reuse a `keyframe_job_id`; returns HTTP 202. |
| `GET /v1/video-jobs` | List jobs. |
| `GET /v1/video-jobs/{id}` | Read status, scene progress and safe errors. |
| `GET /v1/video-jobs/{id}/events` | Observe live status through SSE. |
| `GET /v1/video-jobs/{id}/result` | Read the result manifest and video URL. |
| `GET /v1/video-jobs/{id}/video` | Download a completed MP4. |
| `POST /v1/video-jobs/{id}/resume` | Explicitly resume failed/interrupted work. |

```powershell
$base = 'http://127.0.0.1:8000'
$body = @{ question='How does the pH scale work?'; keyframe_count=5 } | ConvertTo-Json
$job = Invoke-RestMethod -Method Post "$base/v1/video-jobs" `
    -Headers @{'Idempotency-Key'='ph-demo-v1'} -ContentType 'application/json' -Body $body
Invoke-RestMethod ($base + $job.status_url)
curl.exe --no-buffer ($base + $job.events_url)
$result = Invoke-RestMethod ($base + $job.result_url)
if ($result.video_generated) {
    New-Item -ItemType Directory -Force data/tests/downloads | Out-Null
    Invoke-WebRequest ($base + $result.video_url) -OutFile "data/tests/downloads/$($job.id).mp4"
}
```

The default/only mode is `whiteboard_loop`, with 5–10 keyframes. Reuse an idempotency key for
the same request; changed inputs return 409. Intentional new runs need new keys. Closing a
status stream does not cancel generation. Resume reuses valid artifacts, with at most three
attempts. Details: [video API](docs/video-api.md), [keyframe API](docs/keyframe-api.md),
[live status](docs/job-status.md).

## Tests

```powershell
uv run pytest -q
uv run ruff check src tests scripts infra/runpod
docker compose config --quiet
```

Automated tests make no paid provider calls. Live checks in [the setup guide](docs/setup.md#verification-and-reproducibility)
may incur provider costs. Verified results: [setup status](docs/setup-status.md).
All test media stays in `data/tests/`; selected videos, scripts and manifests are in `examples/videos/`.
See [recording and submission instructions](docs/submission.md) for the demo.

