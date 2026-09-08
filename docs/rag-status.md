# Verified RAG preparation status

Verified on 2026-09-08. The corrected corpus is active and ready for downstream jobs.

Active corpus: `5ca36c3eb893c60ad909db9776e60714f6bdd1d5f1cc76a8ce56aa5d4e1278e7`.

| Check | Verified result |
|---|---|
| Source integrity | All 83 original chemistry PDF SHA256 hashes unchanged |
| Corpus | 83 PDFs, 481 physical pages, 53 teaching documents selected |
| Final page suppliers | 296 pdf-inspector 1.18.0; 185 GPT-5.6 Luna visual transcriptions |
| Answer chunks | 637, each tied to one physical page; maximum 696 embedding-input tokens |
| Embeddings | 637/637, text-embedding-3-large, 3072 dimensions |
| Chunk suppliers | 296 native-parser chunks; 341 Luna visual chunks; zero pypdf answer chunks |
| Publication gate | Current policy, extraction_verified=true, required/completed visual pages=185 |
| History | Old snapshots and paid vectors retained; exactly one active corpus |
| Required queries | All three pass staged and active topic/concept checks |
| Repeat vision | 185/185 cached, zero new requests |
| Repeat registration/ingestion | Zero embedding calls with a client that fails on any API call; DB counts unchanged |
| Focused tests | 24 passed; Ruff passed for retrieval and RAG scripts/tests |

The three required queries retrieve source evidence for the requested explanations:

- pH scale: section 10.2 pages 2/1/4/5, including logarithmic behavior and hydronium.
- Why covalent bonds form: section 6.1 page 4 ranks first, including shared-electron
  attraction to both nuclei and lower potential energy.
- Ionic versus covalent bonding: section 6.1 pages 7/1/4/8/9, including sharing, ions and
  electrostatic attraction.

The source-unit warning appears on the covalent query's first hit. The pH query carries
source-overlay errors and unreadable spans. Default retrieval omits the preserved native
Markdown and diagnostic pypdf text from director-facing provenance.

## Actual provider usage

Final visual policy: 185 completed responses, **1,650,015 input tokens + 253,655 output
 tokens = 1,903,670 total**. Output includes 70,118 reasoning tokens; cached input is zero.
Including four superseded initial samples: 189 saved responses, **1,663,283 input + 259,404
output = 1,922,687 total** (71,633 reasoning tokens included in output). No verified current
Luna dollar rate was available, so no dollar estimate is invented.

Final embedding rebuild: **341 changed inputs**, 180,645 billed input tokens in 12 requests,
approximately **$0.023484** at $0.13/million. The other 296 vectors came from existing cached
inputs. Embedding the full 310,410-token final corpus without a cache would cost about
$0.040353. Across all saved preparation revisions, ingestion used 550,378 tokens and the
three query embeddings used 29 tokens: approximately $0.071553 total embedding cost.
These usage totals exclude unobserved provider retries and do not represent an invoice.

## Source limitations and validation boundary

Vision routing is data-derived, not a fixed count: 178 confirmed native interleaving pages,
one native OCR rejection, four manually reviewed additional interleaving pages, and two
source-overlay pages. The final sweep found three additional pages after the first 182-page
batch, reused those 182 cached responses, and transcribed only the three new pages.

The final page flags retain both original parser diagnostic history and current limitations:
390 math-transcription-unverified, 81 table-layout checks, 6 vision-reported source errors,
5 pages with unreadable spans, 2 source formula-rendering overlays and 1 source-unit issue.
The 8 possible-interleaving flags include three pages subsequently transcribed by vision;
the five remaining native suspects are four URL hashes and fractional formula subscripts.
Six low-text teaching pages are omitted from chunks; another low-text page is in an excluded
source. All raw pages remain stored.

Section 10.1 p3 and 10.2 p1 contain visible `Unknown node type: sup` overlays. Covered text
was not reconstructed. Section 6.1 p4's graph literally labels 0.74 on a pm axis; it remains
literal, but source-hash-bound review metadata forbids teaching 0.74 pm as H2 bond length.
The primary [NIST H2 reference](https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000)
reports 0.74144 angstrom. Small historical table symbols remain explicitly unreadable.

The model's 180 high / 5 medium confidence labels are uncalibrated self-assessments. Four
representative v2 pages were independently reviewed before the full batch; further spot
checks covered the second overlay page and newly discovered molecular-polarity interleaving.
Cache integrity, coverage checks and three retrieval queries do not certify every chemical
symbol or downstream generated narration/image. Director jobs must retain the quality
metadata and avoid relying on corrupted or uncertain formulas.

## Evidence and chronology

All paths below are relative to the repository and ignored by Git:

- `data/tests/rag/vision/audit-report.json`: final 185-page cache/image integrity and usage.
- `data/tests/rag/vision/batch-first-run-report.json`: first 182 pages (four v2 samples reused).
- `data/tests/rag/vision/batch-additional-pages-report.json`: three further pages, 182 reused.
- `data/tests/rag/vision/batch-report.json`: final repeat, all 185 responses reused.
- `data/tests/rag/vision/root-sample-review.json`: independent representative visual review.
- `data/tests/rag/parser-comparison/luna-vision/`: original pypdf versus native inspector
  versus selected visual text, plus unchanged source-hash checks.
- `data/tests/rag/retrieval-checks/validate-report.json`: staged three-question validation.
- `data/tests/rag/retrieval-checks/idempotence-report.json`: repeat check **before publication**;
  its `active=false` describes the staged snapshot at that time, not current readiness.
- `data/tests/rag/retrieval-checks/activate-report.json`: successful atomic publication.
- `data/tests/rag/retrieval-checks/smoke-report.json`: three queries against the active corpus.
- `data/tests/rag/retrieval-checks/active-database-receipt.json`: final read-only DB receipt,
  verified gate, actual page suppliers, quality counts and retained inactive history.
- `data/rag/preparation-summary.json`, `ingest-report.json`, `status-report.json`: final
  preparation, charged embedding batches and current active state. The prepared manifest's
  verified marker stays false by design; the authoritative publication marker is in the DB.

Reproduction and stable async retrieval interface are documented in `docs/rag.md`.
