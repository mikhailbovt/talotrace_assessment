# LTX scene workflows

`render-profile.json` is the versioned quality and timing profile. The typed graph builder is
`src/talotrace/providers/ltx_graph.py`; every actual graph is saved alongside its MP4 in
`data/tests/<video-job-id>/scene-NN-{draw,erase}-graph.json`.

Both sampling passes bind the actual hand-reference bytes through MSR and the completed
keyframe through native temporal conditioning, crop appended guide slots, and preserve full
BF16 weights. V2 pins the actual starting latent (blank for draw, completed for erase), uses
only the corresponding end guide, and gives drawing177 frames. The completed keyframe is
not a global MSR reference. These changes address observed v1 motion failures and still
require actual-frame acceptance.

The native AV model samples an audio latent, but the graph never decodes that latent or connects
audio to `CreateVideo`. Only the later measured Kokoro track enters the final MP4.

See `docs/video-api.md` for public requests, pacing/reading holds, and acceptance boundaries.
