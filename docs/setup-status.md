# Setup status

Updated: 2026-09-08. The backend is running, the corrected RAG corpus is active, and all three complete chemistry lessons have passed live API delivery and media review.

| Item | Status | Evidence / remaining work |
| --- | --- | --- |
| RunPod | Running | `b9ue19vhypgn1a`, approved $1.63/hour quote |
| Hardware | Verified by SSH | NVIDIA A100-SXM4-80GB; 81920 MiB; driver 580.159.04; Python 3.12.3 |
| SSH credential | Configured | Dedicated assessment key; pod PUBLIC_KEY set; private key ignored |
| SSH tunnel | Running | PID recorded in ignored `.runtime/ssh-tunnel.json`; local loopback 18188 and 18081 listening |
| Model files | Verified | All 12 files, 81,913,401,739 bytes; pinned sizes/SHA256 checked; reports under `data/tests/runpod/` |
| Kokoro inference | Verified | Real af_heart synthesis through SSH: 8.625 seconds, mono PCM16 WAV at 24 kHz; `data/tests/runpod/kokoro-probe.json` |
| ComfyUI/MSR API | Ready | A100 engine responds; both MSR custom nodes registered; Docker API can reach both GPU services through SSH |
| LTX inference | Runtime passed; text-only chemistry fidelity failed | Actual 121-frame, 1920x1088 MP4 decoded cleanly. Two electron dots merge into one visible dot, so this clip is not an accepted chemistry explanation. See `ltx-text-test.md`. |
| Local checks | Passed | Python 3.12 venv and uv.lock; all 102 tests passed, including the final composite renderer, portable assets, API guards and a real five-segment timestamp regression. Ruff, Compose configuration and diff checks pass. |
| Docker Compose | Built and healthy | API 8000, PostgreSQL 15432, Redis 16379; `/health/ready` HTTP 200. Final deployment selects `keyframe-composite-v1`, with the video worker enabled and active. `scene-timing-v2` guarantees a two-second minimum showing phase. |
| OpenAI API key | Configured and authenticated | User supplied session key in ignored `.env`; `/v1/models` HTTP200 lists Luna, Image2, and text-embedding-3-large; no additional key created |
| RAG corpus | Corrected snapshot active | 637/637 embedded chunks from 83 PDFs and 481 pages; 296 pdf-inspector pages and 185 recorded Luna visual transcriptions. All three staged/active retrieval checks and zero-call repeat ingestion pass. See `rag-status.md`. |
| Question-to-keyframes | Three accepted final frame sets | All 15 final frames passed review, download/hash checks and idempotent replay. Three rejected image attempts remain preserved. Accepted covalent revision: `e65c1a37-e28a-50b0-a148-58caa1e70200`; pH retains its approved narration review. Evidence: `data/tests/assessment-keyframes-v1/quality-review.json`. |
| Live API status and delivery | Verified on all three final video jobs | Initial HTTP 202 acknowledgements took 79–94 ms. Real progress snapshots, concurrent polling, observer disconnect and terminal completed/EOF were verified. All three HTTP MP4 downloads match their final manifests. Evidence: `data/tests/api-refinement/` and `data/tests/final-composite-*-v1/`. |
| Reference video probe | Rejected for required drawing behavior | The 20.9167-second pH scene passed media integrity and retained the diagram, but the user rejected its already-finished background and hand motion without drawing. It is not an accepted animation. Historical evidence: `data/tests/e5757ca9-6d05-4f6c-b1cb-936056f8a054/`. No further reference runs are planned. |
| Whiteboard video pipeline | Three complete lessons accepted | All 15 scenes were reviewed in actual video frames. The selected path composes accepted keyframes with the supplied hand, a progressive reveal, showing phase and clean wipe. Complete Kokoro speech, full MP4 decode and audio similarity above 0.9999 pass. These examples use local FFmpeg and do not dispatch new LTX inference. |

## Final examples

| Question | Selected video | Duration | Video job |
| --- | --- | --- | --- |
| How does the pH scale work? | [pH scale](../examples/videos/ph-scale.mp4) | 1:56 | `ac8ab468-c925-4fcc-b3e2-76263e01e3ce` |
| Why do atoms form covalent bonds? | [Covalent bonds](../examples/videos/covalent-bonds.mp4) | 1:19 | `6601dd63-d29d-4aff-a628-ce610abdb121` |
| What is the difference between ionic and covalent bonding? | [Ionic vs covalent](../examples/videos/ionic-vs-covalent.mp4) | 2:06 | `3fe9b204-f51b-450d-a59f-2eb65aedf695` |

Each example includes its narration and a manifest in [examples/videos](../examples/videos/).
Output is 2048×1152 at 24 fps, with H.264 video and complete Kokoro `af_heart` AAC narration,
without background music. These video jobs reuse the accepted question-to-keyframe jobs;
their paid upstream generation and source provenance remain preserved.

A final assembly timestamp rounding issue was corrected with explicit per-scene durations.
The pH and comparison jobs were resumed on their original IDs using saved speech and clips;
all 30 cached artifacts per job kept identical hashes, sizes and write timestamps.
Consolidated live acceptance is in `data/tests/api-refinement/final-three-lessons-acceptance.json`.
Root review and canonical media checks are in `data/tests/final-composite-review-v1/`.

The selected acceptance criteria are clear diagrams, a functional whiteboard loop and complete
scene-aligned narration. Exact physical marker-to-stroke alignment is not required. LTX-2.5
remains an available renderer, and its earlier diagnostic runs remain recorded separately.
