# Isolated text-only LTX generation test

This experiment checks a real LTX-2.5 forward pass on the approved RunPod A100.
It does not connect LTX to the question-to-keyframes service.

Run the test from the repository root with the SSH tunnel active:

```powershell
.venv/Scripts/python.exe scripts/ltx-text-test.py --run-id <unique-run-name>
```

The script rejects duplicate submissions for a run directory. It writes all
generated evidence beneath `data/tests/ltx-text/<run-name>/`. It collects the
native output from ComfyUI through the SSH tunnel, checks full decoding and stream
metadata using FFmpeg, and extracts six timestamped frames for visual review.
An interrupted local runner can collect a completed prompt without submitting
another generation:

```powershell
.venv/Scripts/python.exe scripts/ltx-text-test.py --run-id <existing-run-name> --collect
```

No OpenAI API key is used by this experiment.

## Inputs and settings

- Input: the original text in `workflows/ltx-text/prompt.txt`, plus a negative prompt.
- Subject: two hydrogen symbols and two electron dots forming an H : H Lewis diagram.
- No images, style anchor, hand reference, MSR, reference audio, or Kokoro input.
- Native LTX audio is generated jointly with the video; it is not external conditioning.
- Full BF16 dev transformer and Gemma 4 BF16 text encoder, using the pinned models manifest.
- 121 frames at 24 fps (5.0417 seconds); base 960 × 544, then latent upscale to 1920 × 1088.
- Stage 1: 30 Euler steps, LTX scheduler (max shift 2.05, base shift 0.95,
  terminal 0.1, stretched); video/audio CFG 3/7, STG 1 at block 28,
  modality guidance 3. No distilled or reference adapter in this stage.
- Stage 2: the official distilled refinement adapter at strength 1.0,
  Euler ancestral, sigmas 0.85 / 0.725 / 0.421875 / 0, CFG 1/1.
- Seeds: 20260908 and 20260909. Diffusion video VAE with tiled decoding;
  512-pixel tiles, overlap 64, temporal size 64, temporal overlap 16.
- H.264 MP4, CRF 18, 8-bit sRGB, native audio attached.

The workflow was built from the installed official ComfyUI LTX-2.5 text-to-video
template and inspected node schemas, adapting its distilled first stage to a
full-model first stage. Lightricks' source uses a 30-step full-model default for
the current model lineage, followed by distilled refinement. The installed native
dual-CFG node does not expose per-modality rescaling; this experiment therefore
does not claim exact equivalence to the Python pipeline or its separate Res2s HQ
preset. These are quality-oriented test settings, not proof of maximum quality.

Sources:

- [LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5)
- [Official ComfyUI integration](https://docs.ltx.io/open-source-model/integration-tools/comfy-ui)
- [Official text-to-video template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_ltx2_5_t2v.json)
- [Pinned full-model defaults](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py)
- [Pinned two-stage implementation](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)

## Current evidence

Run `20260908-text-bf16`, Comfy prompt
`29682277-c21d-4628-9b09-4b09eac15ba6`, completed successfully on 2026-09-08.

| Check | Result |
| --- | --- |
| Actual model execution | All 32 nodes succeeded; no inference fallback or retry |
| Native output | `data/tests/ltx-text/20260908-text-bf16/hydrogen_00001_.mp4` |
| Video | H.264, 1920 × 1088, 121 decoded frames, 24 fps, 5.041667 seconds |
| Native generated audio | Stereo AAC, 48 kHz, 5.010 seconds; no external audio input |
| Full media decode | FFmpeg exit 0, no decode errors |
| File | 691,670 bytes; SHA256 `c126dda8e2846465bd3c9acf9825b3a81555d8e3c1109a26fba6eb34da90ed95` |
| Comfy execution time | 1,281.213 seconds (21 minutes 21 seconds), including cold weight loading |
| Observed maximum GPU memory | 73,953 MiB; sampled by nvidia-smi every 5 seconds, not an exact allocator peak |
| First-stage denoising | 30 steps in approximately 202 seconds after transformer loading |
| Second-stage denoising | 3 steps in approximately 29 seconds after adapter loading |
| Chemistry/prompt fidelity | **Failed:** the two electron dots overlap into a single-looking mark |

The runtime can generate and decode this full BF16 two-stage configuration on the
A100. Visual inspection of six full-resolution frames and a 16-frame contact sheet
found clear H symbols, a clean whiteboard, and coherent motion toward the center.
However, the dots converge horizontally and overlap, rather than forming the
requested visibly distinct vertical electron pair. By the last second the blue dot
mostly covers the red one. Their initial left/right colors are also reversed.
This clip must not be treated as an accepted chemistry explanation.

The exact API graph, prompt, execution history, model revisions/checksums,
runtime versions, GPU samples, full-decode report, and visual findings are stored
under the run directory. `frame-00s.png` through `frame-05s.png` and
`contact-sheet.png` make the semantic defect reviewable. Native audio stream
integrity was checked; its sound content was not independently auditioned.

GPU ownership was released to the separate video-pipeline agent after completion.
A usable explainer still requires stronger visual constraints and review.
