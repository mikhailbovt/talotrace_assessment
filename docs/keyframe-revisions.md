# Reviewed image revisions

`scripts/revise_keyframes.py` creates a new durable job for a reviewed, hash-bound
image correction. It makes no provider calls itself. The existing worker generates
only the changed scenes; unchanged frames retain their original image bytes and
provider receipts, with a new artifact URL and explicit reuse provenance.

The CLI defaults to a dry run. It validates the completed original job against its
saved context, direction and all frame receipts; verifies PNG hashes, current style
reference, active corrected corpus, source chunk content and PDF hashes; and checks
that the exact proposal matches the approved canonical JSON SHA-256. Only scene
direction, image prompt and visible labels can change. Narration, citations, timing
and every other scene field remain unchanged.

Example for the reviewed assessment correction, from the repository root:

```powershell
uv run python scripts/revise_keyframes.py `
  --source-job 6509d611-650d-4c8e-b303-5a9609d41043 `
  --proposal data/tests/assessment-keyframes-v1/covalent-image-corrections.proposed.json `
  --approved-proposal-sha256 aa560cee7b92a4f053c1b895eba0667e8e800c6c0aba992f96f05aca2191dee8 `
  --idempotency-key assessment-keyframes-v1-covalent-reviewed-v1 `
  --after-job 00468651-fe4a-4223-9ae3-0056403ef777 `
  --after-job 6509d611-650d-4c8e-b303-5a9609d41043 `
  --after-job f3bcd500-0133-4b26-9403-2d0c58a54b10
```

Adding `--queue` publishes paid work. This particular correction has root approval
for two image calls, but publication is coordinated after the controlled API rebuild.
The UUID is deterministic from the idempotency key. Repeating the same published
revision returns its existing job and never creates another job. A different patch
or source with that key is refused.

The publication transaction holds an advisory lock, validates evidence, prepares
the new files, then inserts the queued row. The worker cannot see a partial job.
An interrupted transaction may leave unreferenced prepared files; retry accepts
them only when their revision lineage matches. Successful original jobs and files
are never modified or deleted. The existing worker's normal explicit recovery
rules still apply to an interrupted provider call after publication.

New evidence lives under `data/tests/<new-job-UUID>/`: `revision.json`, a copy of
the exact approved proposal, unchanged context, effective direction containing the
original provider output and reviewed patch lineage, copied reusable receipts and
images, and later the worker's generated frames and manifest. A copied receipt's
`reused_from.provider_usage_is_original` prevents interpreting its historical token
usage as new spending. New generated receipts have their own request IDs.

Visual review remains necessary after generation: a successful image API response
and valid PNG do not establish correct science or clear labels.
