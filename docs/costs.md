# Cost per generated lesson

**Estimated marginal cost: $1.2453, rounded to $1.25**, using the completed pH run from
8 September 2026. This is a calculation from saved usage and published Standard API rates,
not an invoice reconciliation or a production-wide average.

- Video job: `590b3c5c-5938-44b1-a622-4225edfebd42`.
- Keyframe job: `b38998a8-dde8-4f09-8227-1484256db021`.
- Output: 130.67 seconds, five high-quality 2048×1152 keyframes, Kokoro `af_heart`,
  `keyframe-composite-v1`; no LTX inference and no regeneration attempts.

| Component | Recorded usage and calculation | USD |
| --- | --- | ---: |
| GPT Image 2 | 7,540 image-input tokens × $8/M + 2,239 text-input tokens × $5/M + 28,250 image-output tokens × $30/M | 0.919015 |
| GPT-5.6 Luna direction | 3 ordinary input tokens × $0.20/M + 13,165 cache-write tokens × $0.25/M + 7,792 output tokens × $1.20/M | 0.012642 |
| Question embedding | Exact-question cache hit; no embedding API request in this run | 0 |
| Allocated RunPod hosting | 692.717973 seconds ÷ 3,600 × $1.63/hour | 0.313647 |
| **Total** | | **1.245305** |

API rates were checked on 8 September 2026 against [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
The 13,165 cache-write tokens are part of the 13,168 input tokens, not additional input.
Output includes reasoning tokens. Images are conservatively charged at uncached input rates.

Hosting allocates the pod to this job for its full request-to-completion interval,
13:16:54.897585–13:28:27.615558 UTC, including time waiting for external image generation.
The actual session rate is recorded in [pod configuration](../infra/runpod/pod.json).
Kokoro ran on that pod; composition and storage ran locally.

The estimate excludes one-time PDF parsing/vision fallback/corpus embeddings, model downloads,
earlier experiments, idle time outside this job, local CPU/database/storage, taxes and egress.
It therefore does not equal the total assessment bill. A fresh question adds its embedding
charge; 10 keyframes, image revisions or provider retries cost more. Reusing accepted images
avoids their API cost. Concurrent jobs can share hosting overhead; a CPU-only Kokoro deployment
could remove the A100 rental from the composite path, but that alternative was not benchmarked.

Local source receipts are `data/tests/b38998a8-dde8-4f09-8227-1484256db021/scene-01.json`
through `scene-05.json`, `direction.json` and `context.json`. The calculation is also saved in
`data/tests/api-walkthrough/cost-estimate.json`; the table above preserves the relevant totals
for readers without the local test cache.
