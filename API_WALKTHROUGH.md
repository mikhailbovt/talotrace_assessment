# API walkthrough

This walkthrough demonstrates all three required chemistry concepts and the asynchronous
question-to-video flow. Follow the [README setup and run steps](README.md) first, then open
[Swagger UI](http://127.0.0.1:8000/docs). Commands below use PowerShell from the repository root.

## 1. Submit a question

In Swagger, expand **POST /v1/video-jobs**, select **Try it out**, enter a unique
`Idempotency-Key` such as `walkthrough-ph-001`, and edit the request body:

```json
{
  "question": "How does the pH scale work?",
  "keyframe_count": 5,
  "mode": "whiteboard_loop"
}
```

Select **Execute**. The API returns **HTTP 202** with an `id`, `status_url`, `events_url`
and `result_url`; generation continues in the background. Repeating the same request with
the same key returns the same job. Use a new key when changing the question or starting a new run.

To repeat this flow for the other concepts, replace `question` with either:

- `Why do atoms form covalent bonds?`
- `What is the difference between ionic and covalent bonding?`

New submissions run real embedding, retrieval, direction, image generation and narration.
The completed examples in section 4 can be inspected immediately without another generation.

## 2. Watch progress

Copy the returned ID into **GET /v1/video-jobs/{job_id}** and select **Execute**.
`effective_stage` shows the current work, including upstream retrieval/direction/keyframes;
`completed_keyframes` and `progress` show progress. Swagger responses are snapshots: select
**Execute** again to refresh, or watch live server-sent events:

```powershell
$base = 'http://127.0.0.1:8000'
$jobId = '590b3c5c-5938-44b1-a622-4225edfebd42' # Replace with your returned ID.
curl.exe --no-buffer "$base/v1/video-jobs/$jobId/events"
(Invoke-RestMethod "$base/v1/video-jobs").jobs | Select-Object id,question,state
```

Closing the stream does not cancel the job. Completion is `state: "video_completed"`;
failures expose `error` and `can_resume` through the same API.

## 3. Retrieve the video

The pH question submitted through Swagger on **8 September 2026** completed on its first
attempt. Its actual **GET /v1/video-jobs/{job_id}/result** response includes:

```json
{
  "id": "590b3c5c-5938-44b1-a622-4225edfebd42",
  "question": "How does the pH scale work?",
  "state": "video_completed",
  "attempts": 1,
  "error": null,
  "video_generated": true,
  "video_url": "/v1/video-jobs/590b3c5c-5938-44b1-a622-4225edfebd42/video"
}
```

This is an excerpt, not the full manifest. [Download this pH run (2:11)](http://127.0.0.1:8000/v1/video-jobs/590b3c5c-5938-44b1-a622-4225edfebd42/video),
use **GET /v1/video-jobs/{job_id}/video** in Swagger, or save it with PowerShell:

```powershell
$result = Invoke-RestMethod "$base/v1/video-jobs/$jobId/result"
if ($result.video_generated) {
    New-Item -ItemType Directory -Force data/tests/downloads | Out-Null
    Invoke-WebRequest ($base + $result.video_url) -OutFile "data/tests/downloads/$jobId.mp4"
}
```

## 4. Watch all three required concepts

These selected, previously completed runs reuse their approved question → retrieval → script
→ keyframe results. Their MP4s are included in the repository, alongside
[complete narration and provenance manifests](examples/videos/README.md).

| Learner question | Repository video | Local API download |
| --- | --- | --- |
| How does the pH scale work? | [pH scale · 1:56](examples/videos/ph-scale.mp4) | [MP4](http://127.0.0.1:8000/v1/video-jobs/ac8ab468-c925-4fcc-b3e2-76263e01e3ce/video) |
| Why do atoms form covalent bonds? | [Covalent bonds · 1:19](examples/videos/covalent-bonds.mp4) | [MP4](http://127.0.0.1:8000/v1/video-jobs/6601dd63-d29d-4aff-a628-ce610abdb121/video) |
| What is the difference between ionic and covalent bonding? | [Ionic vs covalent · 2:06](examples/videos/ionic-vs-covalent.mp4) | [MP4](http://127.0.0.1:8000/v1/video-jobs/3fe9b204-f51b-450d-a59f-2eb65aedf695/video) |

The localhost links require the running assessment API and its original database/artifacts.
On a fresh installation, use the repository videos or submit questions to obtain new job IDs.
All four result endpoints returned HTTP 200 and their MP4 range downloads returned HTTP 206
with `video/mp4` when checked on 8 September 2026.

The videos use generated keyframes, the supplied hand, **Kokoro af_heart** narration and
`keyframe-composite-v1`: blank → drawing/reveal → showing → erase → next scene, without music.
LTX-2.5 is a separate selectable renderer; these examples use local FFmpeg composition.
See the [architecture note](docs/architecture.md) for worker, persistence and provider boundaries.
