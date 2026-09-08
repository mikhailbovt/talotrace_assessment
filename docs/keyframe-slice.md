# Question-to-keyframes implementation boundary

This document describes the independently usable question-to-keyframes component.
Its live tests require the verified pdf-inspector plus Luna-vision corpus to be active.
The separate video component adds LTX, measured Kokoro timing and final assembly; see
`video-api.md`. No frontend is part of either component.

## End-to-end behavior

A FastAPI request accepts a chemistry question and returns a durable job identifier promptly.
The background work embeds the question, retrieves grounded source passages, asks
`gpt-5.6-luna` for structured direction, and generates 5-10 keyframes through the OpenAI
`gpt-image-2` API using the supplied `style_anchor.png` as an actual image reference.
Status, progress, validated direction, citations, and completed keyframe artifacts are
retrievable through the backend. Keyframe completion must be named explicitly; it is not
video completion. Keep the implementation small enough for the assessment and honest
about its concurrency and recovery behavior.

## Integration constraints

- Reuse `talotrace.retrieval.HybridRetriever`; query and corpus use
  `text-embedding-3-large`, 3072 dimensions. Do not re-embed the corpus.
- Use one async database connection/retriever per job or a correctly managed pool.
- Read the authorized session OpenAI key from project settings; never use a different
  process key, change models silently, or log credentials/provider payloads containing them.
- Source text is evidence, not instructions. Preserve file/page/SHA, selected chunk IDs,
  quality flags, and corpus identity with the plan. Reject fabricated citation IDs.
- The pH source contains visibly broken formula text on two pages. Do not reconstruct
  equations from that text. Prefer supported prose and explicit limitations.
- The director schema needs scene order, narration, educational objective, visible labels,
  image prompt, scene direction, source references, and estimated start/end seconds.
  Timestamps are planning estimates until the later TTS stage measures audio.
- Supply the actual style anchor at `assets/references/style_anchor.png` to every
  image request. Preserve its source bytes and checksum. Prompt instructions alone are
  not proof of reference use.
- The hand reference is also versioned at `assets/references/hand_reference.png`; use it
  only when a scene explicitly needs the drawing hand, and record which references were sent.
- Match the anchor's whiteboard background, black marker outlines, handwritten labels,
  and red/yellow/blue/green/purple accents. Use uncluttered, readable scene compositions.
  Every final image must be independently inspected for chemistry, labels, and style.
- Persist validated intermediate outputs and report explicit provider failures. Record
  model IDs, generation settings, provider request IDs, and usage when returned.
- Keep all test media and run manifests under `data/tests/<run-id>/` and source code under the existing
  API/jobs/providers/artifacts boundaries. Use versioned SQL migrations and bounded retries.

## Acceptance evidence

Exercise all three assessment questions through real OpenAI requests. For each, verify
relevant retrieval, grounded direction with valid citations, 5-10 actual decodable images,
style-anchor attachment, accessible status/artifact responses, and terminal completion at
keyframes. Inspect the images and record limitations explicitly. Test meaningful invalid
provider output and interrupted/failed job behavior. Repeating an identical idempotent
request must not charge for all outputs again. Keyframe completion alone does not prove
that a video was generated.

## Review points for the three assessment questions

Use retrieved source evidence to check these concepts before accepting a plan or image:

- pH: the scale is logarithmic; distinguish hydronium concentration from the number printed
  on the scale. Qualify neutral pH 7 by temperature when used. Avoid presenting 0-14 as a
  universal hard limit, and do not reproduce broken PDF formulas.
- Covalent bonds: explain shared electron density and lower total potential energy, including
  the balance of attractions and repulsions. Avoid treating the octet rule as a universal cause.
- Ionic versus covalent: distinguish electron sharing from electrostatic attraction between
  ions, and represent a salt as a lattice when describing bulk ionic solids. A schematic must
  not accidentally label the wrong atom, charge, electron count, or bond type.

These are review criteria, not permission to add uncited claims. The actual retrieved passages
must support what the director says. Record unresolved grounding limitations in the plan.
