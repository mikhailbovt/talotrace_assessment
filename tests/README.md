# Verification scope

Run `uv run pytest -q` from the repository root. Tests cover extraction and vision-cache
integrity, corpus publication, hybrid retrieval, job contracts, grounding and artifact reuse.
Video tests cover graph conditioning, complete narration, FFmpeg assembly, recovery and
artifact delivery. Real HTTP/SSE tests cover progress, disconnect isolation and safe errors.

Live acceptance evidence is separate and lives under `data/tests/`. HTTP health, passing
tests and valid media containers do not by themselves establish chemistry or visual quality.
