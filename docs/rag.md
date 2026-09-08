# Chemistry corpus and hybrid retrieval

Original PDFs remain read-only in `../material/chem_material`. The extraction policy is
`pdf-inspector+gpt-5.6-luna-vision-v1`: pdf-inspector 1.18.0 supplies clean pages, and
GPT-5.6 Luna transcribes actual rendered images of rejected pages. pypdf 6.18.0 is used
only to diagnose and compare defects; it never supplies final answer text.

## Reproduce from the repository root

```powershell
uv sync --frozen
docker compose up -d --wait --wait-timeout 120 postgres
uv run python -m talotrace.retrieval inspect
uv run python -m talotrace.retrieval.vision --concurrency 6
uv run python scripts/audit-rag-vision.py
uv run python -m talotrace.retrieval prepare
uv run python -m talotrace.retrieval register
uv run python -m talotrace.retrieval ingest
uv run python -m talotrace.retrieval validate
uv run python -m talotrace.retrieval activate
uv run python -m talotrace.retrieval smoke
uv run python -m talotrace.retrieval status
uv run python -m talotrace.retrieval query --question "How does the pH scale work?"
uv run pytest tests/test_retrieval.py -q
```

`inspect` writes native extraction and diagnostics to `data/rag/inspection.json`, without
calling OpenAI. The vision stage processes only flagged pages and resumes from per-page
caches. `prepare` requires every visual page to be cached before creating answer chunks.
`register` applies migrations 002/003 and stores a new immutable snapshot. `ingest` embeds
only missing inputs and marks the snapshot ready for validation; it does not publish it.
`validate` queries the staged snapshot. `activate` repeats required-question checks and
atomically publishes a complete, current snapshot without unresolved review flags.

The CLI reads `DATABASE_URL` and `OPENAI_API_KEY` explicitly from this repository's `.env`,
without process-environment credential fallback or dotenv interpolation. Never put keys
on a command line. Provider error logging exposes only category and HTTP status.

For a different location of the same supplied PDFs, pass `--corpus` to `inspect`, the vision
command and `prepare`, for example `uv run python -m talotrace.retrieval inspect --corpus
'D:/chemistry/chem_material'`. Point to the directory containing `Individual/`. Keep the
relative paths and original bytes unchanged so the recorded extraction reviews still match.
Also update `MATERIALS_DIR` for the API's read-only material mount. The remaining ingestion
commands read the prepared snapshot and do not need `--corpus`.

## Extraction, selection and source quality

All 83 chemistry PDFs and 481 physical pages are retained with provenance. Answer retrieval
uses 53 teaching sections. The 13 one-page chapter overviews, 12 homework sheets, index,
three licensing files and placeholder glossary are excluded with recorded reasons.
Identical files, pages and embedding inputs are deduplicated. The overviews are not full
chapters. Physical page citations are one-based; inspector indices are converted once.

Native extraction has confirmed reference-number interleaving on 178 pages, requests OCR
on one further page, and has four additional manually reviewed interleaving pages.
Both known source-error pages also use vision: 185 visual pages and 296 native pages.
Native output, routing reasons and diagnostic pypdf text remain stored for audit, but are
omitted from default retrieval responses so the director receives one selected transcription.

Each visual request uses pypdfium2 5.13.0 at 200 DPI: a full page plus overlapping top/bottom
61%-height detail views, explicitly identified as one physical page. The structured response
contains page Markdown, visible diagram descriptions, unreadable spans, source errors,
warnings and qualitative confidence. The prompt forbids invented formulas or repairs to
source chemistry and requests explicit LaTeX subscripts/superscripts. Confidence is an
uncalibrated model self-assessment, not proof of transcription accuracy.

Cache identity includes PDF SHA256, physical page, requested model, renderer version,
resolution, view strategy and prompt version/hash. Records preserve the actual returned
model, request/response IDs, usage and rendered-image hashes. Document identity also includes
selected text, quality flags and the actual visual response record hash. Four representative
pages were independently compared with original renders before the full batch. This does
not certify every symbol on all pages.

Known source limitations must reach downstream consumers:

- Section 10.1 physical page 3 and section 10.2 physical page 1 contain actual
  `Unknown node type: sup` overlays. Covered symbols stay `[unreadable]`; readable prose is
  retained. Hits carry `source_formula_rendering_error`, `vision_source_errors` and
  `vision_unreadable_spans`.
- Section 6.1 physical page 4 labels a hydrogen potential-energy graph `0.74` on a `pm`
  axis. The transcription stays literal with `source_unit_inconsistency`. The independently
  reviewed [NIST H2 table](https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000)
  reports ground-state equilibrium separation 0.74144 angstrom. The director must not teach
  0.74 pm as the hydrogen bond length; omit that numeric label unless separately verified.
  Hash-bound review evidence is in `configs/rag-extraction-reviews.json`.
- Small historical table labels and other unresolved spans are explicitly reported,
  never reconstructed using modern chemistry knowledge.

Consumers must preserve quality flags and `extraction_provenance`, including `source_errors`,
`unreadable_spans`, `vision_warnings` and `source_quality_reviews`. Do not use corrupted
formulas or uncertain diagram labels in narration or images. The completion gate verifies
extraction/ingestion and representative retrieval, not all scientific/transcription details.

## Chunks, embeddings and retrieval

Chunks remain within one physical page and include the section title in their embedding
input: at most 700 cl100k_base tokens, 80-token overlap and preferred sentence boundaries.
UTF-8 code points survive token-boundary splits. Credits and print footers stay in raw records
but leave answer text; separately extracted visible diagram descriptions remain searchable.

`text-embedding-3-large` uses all 3072 dimensions. Published standard input price is $0.13
per million tokens; reports estimate embedding cost using returned usage. No unverified
Luna dollar rate is assumed: the visual audit reports actual input, cached input, output
and reasoning tokens, with superseded samples accounted for separately.

- [Embedding model and price](https://developers.openai.com/api/docs/models/text-embedding-3-large)
- [Embedding guide](https://developers.openai.com/api/docs/guides/embeddings)
- [Request limits](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
- [pdf-inspector Python API](https://github.com/firecrawl/pdf-inspector)

Embedding batches contain at most 32 inputs and 16,000 tokens, with 90-second timeout and
at most two SDK retries. Responses require the correct model, indices, dimensions, finite
values and normalized nonzero vectors. Visual requests run with bounded concurrency
(default 4; permitted 1-8), 180-second timeout and at most two SDK retries.

Dense retrieval uses exact pgvector cosine search on float32 3072-dimensional vectors.
BM25 scores the same corpus. The top 30 from each ranking are fused with reciprocal rank
fusion (constant 60), returning six hits and at most two from one page. Scores are ranking
signals, not calibrated confidence. Query vectors are cached in PostgreSQL. Exact search
avoids ordinary float32 HNSW's 2000-dimension limit without shortening vectors; see
[pgvector storage/index limits](https://github.com/pgvector/pgvector).

```python
from talotrace.retrieval import HybridRetriever

retriever = HybridRetriever(async_psycopg_connection, explicit_async_openai_client)
result = await retriever.search(question, top_k=6, candidates=30)
```

Results contain corpus ID, model/strategy, and hits with content, citation, source path/title/
SHA256/URL, physical page, actual extractor/version, quality flags and extraction provenance.
An optional third constructor argument `corpus_id` pins a ready staged snapshot for validation.
The default retriever reloads BM25 when the active corpus changes. Use a worker-local
instance/connection, not a shared mutable instance across concurrent jobs.

Before paid director/image work, require active `rag.corpora.summary` to contain
`extraction_policy='pdf-inspector+gpt-5.6-luna-vision-v1'`, `extraction_verified=true`, and
equal integer `vision_required_pages` / `vision_completed_pages`. Only activation sets the
verified marker. Reject old pypdf-answer snapshots or pypdf result suppliers.

## Resume, history and evidence

Embeddings are keyed by input hash, model and dimensions. Each successful embedding batch
and usage record commits together; advisory locking prevents duplicate concurrent ingestion.
A crash after provider charge but before saving/committing can charge missing work again.
Saved token totals exclude unobserved retries. Do not run simultaneous vision processes
against one cache. Preparing or embedding a snapshot leaves the active corpus untouched.
Old snapshots and paid vectors remain available and are reused. Archives are in
`data/rag/snapshots/<id>/`; database extraction history stays immutable.

All comparisons, renders and test reports live in ignored `data/tests/rag/`: `vision/` holds
PNG views, cached JSON/Markdown responses and usage/audit reports; `source-qa/` holds earlier
renders; `parser-comparison/` holds before/after evidence; `retrieval-checks/` holds full
staged/active query results. Production inspection/preparation/ingestion reports live in
ignored `data/rag/`. `docs/rag-status.md` records final verified totals and source limitations.
