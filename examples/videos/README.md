# Best generated videos

Three selected chemistry lessons, generated through the asynchronous FastAPI service.

Each video uses five GPT Image2 keyframes, the supplied hand reference, local whiteboard animation and complete Kokoro af_heart narration. The sequence is **blank → drawing/reveal → showing → erase → next scene**. There is no background music.

These final examples use keyframe-composite-v1. LTX-2.5 remains available as a separate renderer; no new LTX inference was used for these three videos.

| Learner question | Video | Duration | Narration and provenance |
| --- | --- | --- | --- |
| How does the pH scale work? | [ph-scale.mp4](ph-scale.mp4) | 1:56 | [Narration](ph-scale.narration.md) · [Manifest](ph-scale.manifest.json) |
| Why do atoms form covalent bonds? | [covalent-bonds.mp4](covalent-bonds.mp4) | 1:19 | [Narration](covalent-bonds.narration.md) · [Manifest](covalent-bonds.manifest.json) |
| What is the difference between ionic and covalent bonding? | [ionic-vs-covalent.mp4](ionic-vs-covalent.mp4) | 2:06 | [Narration](ionic-vs-covalent.narration.md) · [Manifest](ionic-vs-covalent.manifest.json) |

All three passed full MP4 decoding, HTTP download checksum verification, complete-narration checks and review of representative frames from every scene. Output is 2048×1152 at 24 fps, with H.264 video and AAC audio.

Manifests record job IDs, exact settings, source citations, model/reference hashes, measured timing and selection reviews. The examples reuse the previously accepted question → retrieval → direction → keyframe jobs.

Original test media, intermediate attempts and operational evidence remain in data/tests/. See the [project README](../../README.md) for reproduction and API usage.
