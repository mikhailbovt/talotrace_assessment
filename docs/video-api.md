# Narrated video API

The backend turns validated question-to-keyframe jobs into Kokoro-narrated whiteboard videos.
New requests use only `whiteboard_loop`: blank board → visible hand/reveal → generated
keyframe showing → clean erase → blank, repeated for every scene. All narration comes from Kokoro
`af_heart`; the final video has no background music.

The reference-mode probe passed technical generation/audio checks, but the user rejected it
because its static board and moving hand did not demonstrate actual drawing. Its previous
visual pass is superseded by `rejected_for_drawing_requirement`. That result remains diagnostic
evidence under the earlier strict drawing criterion. The user subsequently accepted approximate
hand/reveal animation for a quick final deliverable; root reviewed and accepted the local
composite preview for that scope. Finished lessons still receive actual media and narration
review. The earlier stroke-accuracy failures do not block the selected composite renderer.

## Submit and retrieve

`POST /v1/video-jobs` requires an `Idempotency-Key` header and one of these bodies:

```json
{"question":"How does the pH scale work?","keyframe_count":5}
```

```json
{"keyframe_job_id":"00468651-fe4a-4223-9ae3-0056403ef777","mode":"whiteboard_loop"}
```

Omitting `mode` defaults to `whiteboard_loop`; explicitly supplying that value has the same
request identity. `mode="references"` returns HTTP422 with the message "Reference generation
is retired for new requests; use whiteboard_loop" before a job is persisted. Other mode values
also fail validation. The OpenAPI request schema advertises only `whiteboard_loop`.

A question creates its upstream keyframe job in the same PostgreSQL transaction. Supplying an
existing keyframe job reuses its exact script, evidence, and images. It does not repeat OpenAI
image generation. Both submission forms require the approved
pdf-inspector plus Luna-vision corpus; the video worker also validates the saved source policy.

The API returns HTTP202 promptly with the video job UUID, upstream keyframe UUID, status/result
URLs and state. `GET /v1/video-jobs` lists requests. `GET /v1/video-jobs/{id}` includes upstream
progress and current speech/render/assembly stage. `GET /v1/video-jobs/{id}/result` returns the
measured timeline and final manifest. When state is `video_completed`,
`GET /v1/video-jobs/{id}/video` returns the checksum-verified `video/mp4` file.

Historical stored jobs keep their original `mode`, including `references`, in list, status,
SSE, result, and video-download responses. These records remain accessible as experiment
evidence. The retired mode cannot be submitted again through the new-request API; retrieve an
existing reference job by its saved UUID.

An unfinished historical reference job cannot resume: its resume endpoint returns HTTP409
with "Reference generation is retired; create a new whiteboard_loop video job", and status
reports `can_resume=false`. Calling resume on a completed historical job still returns its
existing result without dispatching or repeating generation.

Accepted/resumed responses include `Location` and `Retry-After: 2`. The new
`GET /v1/video-jobs/{id}/events` sends live SSE snapshots of the same status representation,
including upstream keyframe progress while the video waits. Named stages and
`effective_stage` distinguish retrieval/direction/keyframes, Kokoro speech, local animation or LTX rendering,
and assembly. `blocking_reason` reports a disabled video worker or an upstream that requires
resume. `terminal` and `can_resume` make completion/recovery explicit. See
[job status and streaming](job-status.md) for observer, heartbeat and reconnection semantics.

Artifact checksums are computed in a worker thread with bounded-memory file reads. Downloading
a large final video does not synchronously hash the entire MP4 on the event loop. File responses
retain HTTP byte-range support, and missing/corrupt artifacts return HTTP409.

`POST /v1/video-jobs/{id}/resume` resumes failed/interrupted work, at most three attempts. If its
keyframe job failed, resume that upstream job first. Repeating a submission with the same
idempotency key returns the existing video; changed inputs, mode, reference hashes, or generation
settings return HTTP409. Resume reuses saved speech, clips, and Comfy prompt IDs. The exact
validated source/script/frame snapshot is bound and preserved when rendering begins.

`VIDEO_WORKER_ENABLED=false` leaves upstream jobs and status endpoints usable without dispatching
video work. `VIDEO_RENDERER` selects `keyframe-composite-v1` or `ltx` independently of the public
whiteboard mode. `/health/live` exposes the configured `video_renderer` and
`video_worker_enabled`; accepted jobs and status/SSE expose their persisted `renderer` dict.
The final `/result` includes the same provenance in `result.renderer` and
`result.settings.renderer`. Changing the renderer changes job identity, preventing accidental
reuse of an incompatible job. Only one video worker across processes owns the PostgreSQL lock.
The LTX branch waits for the Comfy queue; the composite branch makes no Comfy/GPU request.

## Reviewed narration corrections

A job may have `data/tests/<keyframe-job-id>/narration-review.json`. This is generated review
evidence, not a public arbitrary-script editing API. Its strict `narration-review-v1` schema
requires `approval_status=approved`, `approved_by=root`, the exact original direction-receipt
SHA256, and per-scene original narration, replacement narration, and unchanged source IDs.
Unapproved, stale, mismatched, duplicated, or extra-field corrections are rejected. The reader
changes narration only; images, labels, citations, objectives and original director receipts
remain preserved. The selected review's canonical SHA is part of video-request identity and
its original/effective scripts are written into source evidence and the final manifest.

If a review changes after submission, that video job stops with a settings mismatch. Submit a
new idempotent video request for the new reviewed script; saved originals are not overwritten.

## Whiteboard drawing and erasing

### Selected local composite renderer

`keyframe-composite-v1` uses local FFmpeg to animate the existing generated keyframes and
supplied `hand_reference.png`. A visible hand sweep accompanies a left-to-right image reveal,
followed by showing the exact finished keyframe and a clean wipe to the reviewed blank-board
image. The motion is intentionally approximate compositing; it is not new LTX inference or
stroke-level handwriting synthesis. The approved blank image, hand, and keyframes keep their
source hashes in the renderer/settings/per-clip evidence.

The policy uses 97 drawing/reveal frames and 25 erase frames at 24 fps. Its effective
`settings.assembly_profile` is separate from the retained LTX `settings.profile`; the rendered
output is 2048×1152. Renderer provenance explicitly records `provider=local_ffmpeg`,
`new_ltx_generation=false`, and `motion=supplied_hand_sweep_with_keyframe_reveal_then_clean_wipe`.
Existing accepted keyframe jobs are reused, with actual Kokoro speech generated for every
complete scene script. This continuation makes no new OpenAI image or LTX calls.

### Optional LTX renderer and historical experiments

The LTX whiteboard path uses full BF16 LTX2.5 dev, the pinned MSR LoRA and actual image bytes uploaded to
ComfyUI: MSR pic1 is the versioned `hand_reference.png`. The completed scene keyframe is
restricted to temporal conditioning, so it does not globally bias every frame toward a
finished diagram.
The fixed profile is in `workflows/video/render-profile.json`. The default is 1024×576 base,
two-stage spatial refinement to 2048×1152, 24fps, 30 base Euler steps and three refinement steps.
These are quality-oriented settings, not proof of maximum achievable visual quality.

`whiteboard_loop` renders two clips per scene. The v2 drawing clip pins a blank first latent
using native `LTXVImgToVideoInplace` at strength1, then conditions the completed keyframe only
at the final frame176. There is no early completed-middle guide. The erasing clip pins the
completed keyframe first and conditions a blank board at frame72. Both passes apply the
in-place pin before appending the end guide and hand-only MSR tokens. The v2 drawing sequence
has the same 29 conditioned visual latent frames as v1:23 content +1 end guide +5 repeated hand
reference frames, replacing16 content +3 guides +10 reference frames. Actual peak memory
must still be checked because audio duration and decoded frame count also increase.
The final intended sequence is blank → drawing → complete
diagram → optional reading hold → physical erasing → blank, repeated for each scene.

Native `LTXVAddGuide` and the MSR node append conditioning tokens to the video latent before
audio/video concatenation. Each sampling pass crops guide slots before upscaling or decoding;
the second pass rebuilds both temporal guides and MSR references at its higher resolution.
The native pin writes the actual starting samples and zeroes their noise mask; AddGuide/MSR
append to that mask, AV concatenation preserves it in the video member, and separation/crop
preserve the original content mask. Refinement re-pins the first latent after upscaling.
Refinement starts with pristine text conditioning, because the pinned MSR extension cannot
append to the explicit `None` metadata left in `LTXVCropGuides`' conditioning outputs. Its
cropped generated latent still supplies refinement. This graph compatibility correction was
verified by an actual second-stage run; the failed original graph and provider history are
retained in the probe evidence.

The graph and actual reference hashes are persisted per clip. Earlier strict experiments tested
new strokes coordinated with the marker and physical erasing; their failed reviews are retained
as diagnostics. The current composite deliverable follows the user's later relaxed motion
criterion. LTX conditioning remains probabilistic and its outputs require separate inspection.

The internal `references` graph is retained for historical experiment evidence and compatibility
tests. It is not an available mode for new public requests and its earlier output is not an
accepted example of the intended whiteboard product.

The blank board is a deterministic white `EmptyImage` conditioning node. It is deliberately a
plain layout input, rather than an image-generation substitute for any requested keyframe.

## Speech and timing

Every scene's complete validated narration is sent verbatim to Kokoro-82M, voice `af_heart`,
speed 1. Each returned mono PCM16 24 kHz WAV is checked for usable audio, and its actual sample
count supplies timing. Estimated director timestamps do not truncate or accelerate narration.

Each scene begins with its renderer's configured drawing clip: 97 frames for the selected
composite, or 177 frames in LTX v2. The separate `scene-timing-v2`
assembly policy then shows the exact generated keyframe for at least 2 seconds, extending that
hold to cover any remaining measured speech. This pause is an intentional educational pacing
choice. The configured erase (25 frames for composite, 73 for LTX v2) starts only after both speech and showing finish;
the next scene's narration starts with its own drawing, after that erase. Short speech leaves
explicit silence through the remaining drawing/showing/erase. Speech samples are never removed,
time-stretched, or overlaid with other audio. This is scene-level alignment, not word-to-stroke
synchronization. Semantic construction-stage narration alignment remains future integration work.

`settings.timing_policy` is part of the whole video job identity and timeline. It is excluded
only from the LTX clip seed/input identity, so a showing-duration change does not change the
render experiment. The v2 render profile and graph sampling settings remain unchanged. The
20.9-second pH scene still totals 575 frames / 23.9583 seconds because its existing hold exceeds 2 seconds.

In the LTX branch, its native audio latent participates in AV sampling but is never decoded or
saved. Each Comfy output must contain video only. FFmpeg explicitly maps only the concatenated
Kokoro WAV into the final MP4. No music, LTX soundtrack, speech effects, or marker sounds enter
the finished audio track. The final video gets one H264 video stream and one AAC audio stream.

Acceptance checks count actual video frames, require one audio stream, compare both measured
stream durations with the full timeline, fully decode the MP4, and compare its decoded audio
samples against the expected complete Kokoro track. AAC is lossy, so the saved PCM hash proves
the exact input and decoded signal similarity verifies the encoded output. These checks prove
the assembly contract; they do not replace listening to the narration or inspecting chemistry.

## Persistence and failure boundaries

All media, graphs, receipts, and manifests live in `data/tests/<video-job-UUID>/`. Each scene
has speech WAV/receipt, draw and erase MP4/reference receipts, plus a
silent assembled scene. `complete-kokoro-narration.wav`, `timeline.json`, `source.json`,
`final.mp4`, and `video-manifest.json` record the final result and its evidence.
LTX clips additionally retain their graph/dispatch receipts; composite clips identify local FFmpeg.

In the LTX branch, before posting a Comfy graph, the worker saves a dispatch identity. A known prompt ID is polled
again on resume. If a send was ambiguous, the provider queue/history is searched for that exact
identity; if it cannot be found, the job fails clearly instead of silently billing a duplicate.
A confirmed execution failure or HTTP400 graph rejection can be freshly dispatched only during
a later explicit job attempt; prior dispatch records are retained. A lost result after completed
remote generation is downloaded again from the same prompt. History loss after a remote restart
can require operator investigation. API restart marks active local work interrupted and preserves
its remote prompt; it does not interrupt someone else's GPU job.

## Verification

```powershell
uv run pytest tests/test_api_async.py tests/test_video.py tests/test_keyframes.py -q
```

The tests include whiteboard-only public validation, preserved historical-job reads/downloads,
async HTTP/SSE, graph topology/conditioners, complete PCM preservation, long-speech timing,
ambiguous and failed provider dispatch recovery, and real ffmpeg mux/full-decode. Internal
reference-mode cases remain as historical compatibility tests; they do not make that mode public.
Synthetic media used in deterministic tests is explicitly confined to `data/tests/unit-video/`;
it is not presented as model-generated acceptance evidence.

Rejected reference-mode experiment evidence is retained under
`data/tests/e5757ca9-6d05-4f6c-b1cb-936056f8a054/`: `final.mp4`, `probe-result.json`,
`visual-review.json`, the exact reference hashes, both dispatch records and archived graphs.
The generated portion is 121 frames at 2048×1152; full narration is 20.9 seconds, giving a measured
20.9167-second MP4 with one AAC speech track. Decoded audio similarity to the full Kokoro PCM
was 0.999926. The explicit keyframe reading hold is 15.875 seconds. The hand and correct pH
formula/subscripts/charges were inspected in actual generated frames. These checks establish
technical and scientific details of that scene, but the user rejected its drawing behavior.
It is not a product acceptance result.

The v1 whiteboard probe at `data/tests/e88d74ec-e269-43a2-b322-1a4f4e1071cd/` also failed
the drawing gate: its first frame already contains a beaker and hand, and most remaining
artwork appears abruptly between frames 51 and 52. `draw-visual-review.json` and dense contact
sheets record this failure; final chemistry fidelity does not override it. The baseline
profile, actual graphs and provider receipts remain preserved. Its erasing also failed: a
rectangular wipe removes areas independently of the visible marker.

The v2 probe at `data/tests/a8bf09bd-f5f9-48aa-b72e-385a842b4f04/` passed media integrity but
was independently rejected by root and the video agent for drawing and erasing. A blank first
frame improved, but a ghost diagram appears, the beaker appears before its contour is traced,
labels fade after the hand leaves, and generated extra text corrupts the finished scene. During
erase the marker writes again while untouched artwork disappears. Its 575-frame final MP4 retains
complete Kokoro speech, but is not an accepted lesson. Exact graphs, dense frames, reviews and
provider receipts are retained; no full lesson should be dispatched based on these failures.

## Isolated construction diagnostic

`scripts/probe_construction_stage.py` tests only a 97-frame blank-to-empty-beaker contour.
It uses two reviewed API-edited images, a matched blank and the beaker exterior outline, with
hand-only MSR, native start pins and end-only guides in both BF16 passes. An explicit isolated
motion prompt excludes all chemistry writing, liquid, ions, colors and additional objects.
Production scene generation is unchanged by this diagnostic.

The strict `construction-stage-v1` manifest under `data/tests/whiteboard-beaker-guides-v1/`
binds the exact original keyframe, both guide PNG hashes, provider-receipt byte hashes and exact
stage prompt. Root approval and the expected canonical manifest SHA are required. Paths must
stay under the artifact directory, and guides must be opaque 2048×1152 PNGs. A preflight-only run
uploads approved references and verifies the graph without submitting GPU work. Adding `--render`
submits the single unvoiced draw clip; there is no erase, narration or full-lesson dispatch.
Use the same fresh run UUID to continue its reviewed preflight. Results remain pending visual
review until dense frames establish progressive marker-contact strokes and stable geometry.

After enabling the video worker, run full API acceptance against existing reviewed keyframes:

```powershell
uv run python scripts/smoke_videos.py --keyframe-job-id 00468651-fe4a-4223-9ae3-0056403ef777 --mode whiteboard_loop --prefix final-composite-ph-v1 --expected-renderer keyframe-composite-v1 --expected-review-sha256 c57b2279550ea070f04e9d1150be2445aace047f0649a81e86a90e615830b6db
```

This checks deployed renderer/worker readiness before submitting one whiteboard-loop job,
records HTTP202/Location/acceptance latency, verifies idempotent replay, polls actual progress, checks the
approved narration and renderer identities, and verifies downloaded final-video checksums. It never resumes
a failed job automatically. Inspect frames and listen to each result before selecting examples.

Pinned contracts inspected:

- https://github.com/Comfy-Org/ComfyUI/blob/efa6c8f804bff78b46a0fd458ebd2e47bba07a30/comfy_extras/nodes_lt.py
- https://github.com/liconstudio/ComfyUI-LTX2.5-MSR/tree/98941179a82223a0a62219550272b2823b3e3a9c
- https://docs.comfy.org/tutorials/video/ltx/ltx-2-5
