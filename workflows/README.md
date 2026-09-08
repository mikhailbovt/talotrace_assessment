# Generation workflows

- `ltx-text/`: isolated text-only generation graph and prompt. Runtime and visual acceptance
  are documented in `docs/ltx-text-test.md`.
- `video/render-profile.json`: versioned settings for whiteboard drawing and erasing,
  built by `src/talotrace/providers/ltx_graph.py`. The earlier reference-only graph remains
  available internally to reproduce historical evidence; it is not a public generation mode.

Every dispatched graph and provider receipt is saved under its job in `data/tests/`.
The upstream MSR example is downloaded with the pinned model bundle on RunPod; it supplies
integration guidance and is not evidence of an accepted chemistry video.
