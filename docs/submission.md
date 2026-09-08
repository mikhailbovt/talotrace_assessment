# Demo and submission

Source: the supplied five-page **Coding Challenge: AI Chemistry Video Request Service**.
Pages 1 and 5 distinguish the working-session recording from the demo/API walkthrough.
The suggested implementation allocation is 90–120 minutes; page 1 explicitly allows further
polishing. A later walkthrough should be identified as a walkthrough, not as the original work.

## What to record now

The author reports that the main working session was recorded, including an explanation of
unfinished work and long LTX generation. Later completion was not fully recorded. Preserve that
recording and describe its coverage accurately in the handoff. The remaining video is a short
demo of the finished backend. **5–7 minutes is a suggested demo length, not a challenge limit.**

Use the same screen-recording setup as the work session. Make terminal/API text readable and
include computer audio when playing the chemistry videos. Do a short playback check first.
Keep credentials and private configuration closed while recording.

| Time | Show and explain |
| --- | --- |
| 0:00–0:30 | Introduce the chemistry request service and three supported questions. Show the short README and API docs. |
| 0:30–1:30 | Submit a real job, show HTTP 202 and its ID, then watch live progress. Explain that the demo reuses approved keyframes to avoid repeating paid image generation. |
| 1:30–2:30 | Show the job list, status and idempotent replay. Close/reconnect an observer and show that work continues. Explain PostgreSQL state versus media files. |
| 2:30–4:30 | Open each of the three completed results through its API URL. Play about 20–30 seconds from each with sound, including a transition. Show the corresponding learner question. |
| 4:30–5:30 | Show tests and one result manifest: renderer identity, complete narration, source citations and media validation. Briefly explain the recorded failed-assembly/resume example. |
| 5:30–7:00 | Explain the local-composition/LTX tradeoff, costs, limitations and how a new STEM corpus would plug in. If the new job has completed, open it; otherwise show its honest waiting/progress state. |

The demo should show all three required concepts. There is no need to replay all three videos
in full; their complete MP4s accompany the repository. Do not describe the final composition
examples as new LTX inference. The LTX experiments and the selected renderer are separate.

## Demo commands

Run from the repository root with the API, corpus and Kokoro already ready. These commands are
for the recording; creating the job below performs real Kokoro/FFmpeg work. It does not request
new OpenAI images or LTX clips. Keep the selected renderer visible:

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/health/live"
Invoke-RestMethod "$base/health/ready"
(Invoke-RestMethod "$base/v1/video-jobs").jobs |
    Select-Object id,question,state,attempts
```

Create a new pH video job using the example's saved upstream keyframes. On a fresh installation,
generate those keyframes first; the supplied example IDs belong to the assessment database.

```powershell
$example = Get-Content examples/videos/ph-scale.manifest.json -Raw | ConvertFrom-Json
$body = @{ keyframe_job_id=$example.keyframe_job_id; mode='whiteboard_loop' } | ConvertTo-Json
$headers = @{'Idempotency-Key'="demo-$([guid]::NewGuid().ToString('N'))"}
$timer = [Diagnostics.Stopwatch]::StartNew()
$response = Invoke-WebRequest -Method Post "$base/v1/video-jobs" `
    -Headers $headers -ContentType 'application/json' -Body $body
$timer.Stop()
$job = $response.Content | ConvertFrom-Json
"HTTP $($response.StatusCode), accepted in $($timer.ElapsedMilliseconds) ms"
$job | Select-Object id,state,status_url,events_url
```

Repeat the identical request with the same key: it must return the same job ID. Show progress;
Ctrl+C stops only the observer. Reopen the stream or poll while the worker continues.

```powershell
$same = Invoke-RestMethod -Method Post "$base/v1/video-jobs" `
    -Headers $headers -ContentType 'application/json' -Body $body
$same.id -eq $job.id
Invoke-RestMethod ($base + $job.status_url)
curl.exe --no-buffer ($base + $job.events_url)
```

Open the three already completed examples through the API. Paste each printed URL into Chrome
or the API client/player. These calls are read-only and do not regenerate anything.

```powershell
Get-ChildItem examples/videos/*.manifest.json | ForEach-Object {
    $example = Get-Content $_.FullName -Raw | ConvertFrom-Json
    $result = Invoke-RestMethod "$base/v1/video-jobs/$($example.video_job_id)/result"
    [pscustomobject]@{
        Question = $example.question
        State = $result.state
        Video = $base + $result.video_url
    }
}
uv run pytest -q
```

The example files also work without the original database. See [selected videos](../examples/videos/).
To show recovery without inventing a failure, use the saved pH/contrast failure and resume
receipts in `data/tests/api-refinement/final-three-lessons-acceptance.json`. Explain that these
are recorded runs. All 30 cached speech/clip files per resumed job stayed unchanged.

## Points to explain

- A request commits durable state and returns promptly; model work happens in async workers.
- PostgreSQL owns status/identity/progress. The artifact store owns images, speech and MP4 bytes.
- Schemas, source warnings, hashes, bounded attempts and media validation reduce unreliable output.
  Three concepts and cache recovery were verified; repeated fresh stochastic image generation
  is not claimed to be byte-identical or automatically scientifically approved.
- Final examples use exact generated diagrams plus a local reveal/hand/wipe effect. LTX remains
  selectable. This preserves labels and lowers repeated generation cost.
- Explain cost as separate one-time corpus preparation, new question/direction/image calls,
  and runtime hosting. Use saved provider usage and account billing for actual dollar amounts.
  A precise per-video dollar total has not been established; do not present an invented estimate.
  Cached reuse avoids new image calls, but a running cloud pod still has hosting costs.

## Handoff checklist

| Challenge deliverable | Current preparation / remaining action |
| --- | --- |
| FastAPI backend; request/list/status/result API; async flow; visual and audio output | Implemented; all three final examples passed live API delivery. |
| Short README with setup, run, API and tests | [README](../README.md), shortened to the quick start. |
| Short architecture note covering the three boundaries | [Architecture note](architecture.md). |
| Demo video or API walkthrough covering all three queries | Record the walkthrough above. |
| Three best generated videos **committed** with their learner queries | Present in `examples/videos/`; still need to be committed and pushed. |
| GitHub repository link and reviewer read access | Remote is configured; publish changes and verify access for all three reviewers. |
| ZIP containing the work | Package the source, documentation and selected examples, excluding credentials, caches and model weights. |
| Google Drive link to full-screen/face work recording | Recording coverage is author-reported; upload the actual recording and state that later polishing happened afterward. |
| Rough per-artifact cost explanation | Usage/provenance is saved; reconcile provider pricing/billing before quoting a total. |

Send the code link, ZIP, demo link and Google Drive recording link to **careers@growtrics.ai**,
with **praveen.k@growtrics.ai** and **wayne.le@growtrics.ai** in CC. The PDF asks that all three
have read access to the repository. No submission email or access invitation has been sent.

Suggested recording-coverage note, adjusted to match what is actually in the file:

> The recording covers the main implementation session, including my explanation of the
> remaining work and LTX generation latency. I completed final assembly, validation and
> documentation afterward. The separate demo shows the finished backend and all three concepts.
