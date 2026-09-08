"""Read-only PDF extraction and reproducible, page-cited chemistry chunks."""

import hashlib
import json
import re
import shutil
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

from .pdf_extraction import INSPECTOR_VERSION, extract_pdf

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
PRICE_PER_MILLION = 0.13
PIPELINE_VERSION = "chem-pdf-inspector-luna-vision-v3"
EXTRACTION_POLICY = "pdf-inspector+gpt-5.6-luna-vision-v1"
ENCODING = tiktoken.get_encoding("cl100k_base")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def token_count(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


@dataclass
class Page:
    number: int
    raw_text: str
    text: str
    flags: list[str]
    extraction_metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    id: str
    path: str
    title: str
    sha256: str
    selection: str
    source_url: str | None
    pages: list[Page]
    extraction_metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    page: int
    ordinal: int
    content: str
    input_text: str
    input_sha256: str
    tokens: int


def clean_page(raw: str) -> str:
    """Remove print furniture, never infer or repair chemical expressions."""
    text = unicodedata.normalize("NFC", raw).replace("\x00", "")
    # Visual descriptions are structured extraction content appended after the literal page.
    # Preserve them even if the printed page ends with attribution boilerplate.
    text, diagram_separator, diagram_text = text.partition(
        "\n\n## Visible diagram descriptions\n\n"
    )
    # Credits remain in raw page records; they are not teaching passages.
    text = re.split(r"(?im)^(?:#+\s*)?Contributors and Attributions\s*$|^This page titled ", text)[
        0
    ]
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*[\d.]+\s+\[?https://chem\.libretexts\.org/", line):
            continue
        if re.match(r"^\s*https://chem\.libretexts\.org/@go/page/\d+\s*$", line):
            continue
        # Private-use glyphs are decorative PDF icons, not chemistry symbols.
        line = "".join(c for c in line if not (0xE000 <= ord(c) <= 0xF8FF))
        line = line.replace("<u>", "").replace("</u>", "")
        lines.append(line.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if diagram_separator:
        cleaned += diagram_separator + diagram_text.strip()
    return cleaned


def selection_reason(path: Path, pages: list[Page]) -> str:
    name = path.stem.lower()
    if "licensing" in name:
        return "excluded_licensing"
    if "!a index" in name:
        return "excluded_index"
    if "glossary" in name and "sample definition" in " ".join(p.text.lower() for p in pages):
        return "excluded_placeholder_glossary"
    if "homework" in name:
        return "excluded_exercises"
    if re.match(r"^\d+!", name):
        return "excluded_chapter_overview"
    return "included"


def split_page(text: str, title: str, max_tokens: int = 700, overlap: int = 80) -> list[str]:
    """Split at token boundaries, prefer a nearby sentence/newline, keep page provenance."""
    budget = max_tokens - token_count(title + "\n\n") - 4
    if budget <= overlap or overlap < 0:
        raise ValueError("Chunk budget must exceed title and overlap")
    tokens = ENCODING.encode(text, disallowed_special=())
    byte_offsets = [0]
    for token in tokens:
        byte_offsets.append(byte_offsets[-1] + len(ENCODING.decode_single_token_bytes(token)))
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + budget, len(tokens))
        # BPE tokens may each contain only part of a UTF-8 code point.
        while end > start:
            try:
                ENCODING.decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                end -= 1
        if end == start:
            raise ValueError("Chunk budget cannot fit a Unicode character")
        if end < len(tokens):
            candidate = ENCODING.decode(tokens[start:end])
            boundaries = [m.end() for m in re.finditer(r"[.!?]\s+|\n", candidate)]
            if boundaries and boundaries[-1] > len(candidate) * 0.7:
                target = byte_offsets[start] + len(candidate[: boundaries[-1]].encode("utf-8"))
                end = bisect_right(byte_offsets, target) - 1
                while end > start:
                    try:
                        ENCODING.decode(tokens[start:end], errors="strict")
                        break
                    except UnicodeDecodeError:
                        end -= 1
        content = ENCODING.decode(tokens[start:end]).strip()
        if content:
            chunks.append(content)
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
        while start < end:
            try:
                ENCODING.decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                start += 1
    return chunks


def apply_visual_transcription(extracted: dict, source_sha: str, vision_root: Path) -> dict:
    from .vision import load_cached, requires_vision

    if not requires_vision(extracted):
        return extracted
    record = load_cached(vision_root, source_sha, extracted["number"])
    if record is None:
        raise ValueError(
            f"Required visual page transcription is missing: {source_sha} "
            f"page {extracted['number']}; run the vision transcription stage first"
        )
    transcription = record["transcription"]
    if (
        "source_formula_rendering_error" in extracted["flags"]
        and not transcription["source_errors"]
    ):
        raise ValueError("Visual transcription omitted a known visible source error")
    meta = extracted["extraction_metadata"]
    meta.setdefault("native_markdown", extracted["raw_text"])
    meta.update(
        {
            "extractor": "gpt-5.6-luna-vision",
            "extractor_version": record["model_resolved"],
            "requires_vision": True,
            "vision_completed": True,
            "vision_cache_identity": record["identity"],
            "vision_record_sha256": digest(json.dumps(record, sort_keys=True)),
            "vision_response_id": record["response_id"],
            "vision_request_id": record["request_id"],
            "vision_usage": record["usage"],
            "vision_confidence": transcription["confidence"],
            "vision_confidence_basis": "uncalibrated model self-assessment of legibility",
            "unreadable_spans": transcription["unreadable_spans"],
            "source_errors": transcription["source_errors"],
            "diagram_descriptions": transcription["diagram_descriptions"],
            "vision_warnings": transcription["warnings"],
        }
    )
    flags = [flag for flag in extracted["flags"] if flag != "vision_required"]
    flags.extend(["vision_transcribed", "math_transcription_unverified"])
    flags.extend(record["quality_flags"])
    if transcription["unreadable_spans"]:
        flags.append("vision_unreadable_spans")
    if transcription["source_errors"]:
        flags.append("vision_source_errors")
    raw = transcription["page_markdown"]
    if transcription["diagram_descriptions"]:
        raw += "\n\n## Visible diagram descriptions\n\n" + "\n\n".join(
            transcription["diagram_descriptions"]
        )
    return extracted | {"raw_text": raw, "flags": sorted(set(flags)), "extraction_metadata": meta}


def inspect(corpus_root: Path, output: Path, review_file: Path) -> dict:
    """Create a read-only extraction manifest before the paid vision stage."""
    from .vision import requires_vision

    reviews = json.loads(review_file.read_text(encoding="utf-8")) if review_file.exists() else {}
    paths = sorted(corpus_root.rglob("*.pdf"))
    if not paths:
        raise ValueError("No PDF files found under the chemistry corpus path")
    documents = []
    for path in paths:
        source_hash = digest(path.read_bytes())
        metadata, pages = extract_pdf(path, source_hash, reviews)
        documents.append(
            {
                "path": path.relative_to(corpus_root).as_posix(),
                "sha256": source_hash,
                "extraction_metadata": metadata,
                "pages": pages,
            }
        )
    summary = {
        "stage": "inspection_only_not_answer_content",
        "documents": len(documents),
        "pages": sum(len(d["pages"]) for d in documents),
        "vision_required_pages": sum(
            requires_vision(page) for doc in documents for page in doc["pages"]
        ),
        "extraction_identity": documents[0]["extraction_metadata"]["extraction_identity"],
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "inspection.json.tmp"
    temporary.write_text(
        json.dumps({"summary": summary, "documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output / "inspection.json")
    return summary


def prepare(
    corpus_root: Path,
    output: Path,
    review_file: Path | None = None,
    vision_root: Path | None = None,
) -> dict:
    review_file = review_file or Path("configs/rag-extraction-reviews.json")
    vision_root = vision_root or Path("data/tests/rag/vision")
    reviews = json.loads(review_file.read_text(encoding="utf-8")) if review_file.exists() else {}
    paths = sorted(corpus_root.rglob("*.pdf"))
    if not paths:
        raise ValueError("No PDF files found under the chemistry corpus path")
    documents, chunks = [], []
    seen_files, seen_pages, seen_chunks = {}, set(), set()
    for path in paths:
        source_hash = digest(path.read_bytes())
        relative = path.relative_to(corpus_root).as_posix()
        document_meta, extracted_pages = extract_pdf(path, source_hash, reviews)
        pages = []
        for extracted in extracted_pages:
            extracted = apply_visual_transcription(extracted, source_hash, vision_root)
            number, raw = extracted["number"], extracted["raw_text"]
            clean = clean_page(raw)
            flags = extracted["flags"]
            if len(re.findall(r"[A-Za-z]", clean)) < 30:
                flags.append("low_text_content")
            pages.append(Page(number, raw, clean, flags, extracted["extraction_metadata"]))
        document_meta["extraction_policy"] = EXTRACTION_POLICY
        selected_identity = [
            {
                "page": page.number,
                "text_sha256": digest(page.text),
                "flags": page.flags,
                "extractor": page.extraction_metadata["extractor"],
                "version": page.extraction_metadata["extractor_version"],
                "vision_record": page.extraction_metadata.get("vision_record_sha256"),
            }
            for page in pages
        ]
        doc_id = digest(
            relative
            + "\n"
            + source_hash
            + "\n"
            + document_meta["extraction_identity"]
            + PIPELINE_VERSION
            + json.dumps(selected_identity, sort_keys=True)
        )
        if not pages or not any(p.text for p in pages):
            raise ValueError(f"PDF has no extractable text: {relative}; OCR review required")
        reason = selection_reason(path, pages)
        if source_hash in seen_files:
            reason = "excluded_duplicate_file"
        seen_files[source_hash] = relative
        title = path.stem.replace("!", ": ", 1)
        # A vision model may omit print furniture; retain URLs from original native evidence.
        source_url_text = "\n".join(
            str(page.extraction_metadata.get("native_markdown", "")) + "\n" + page.raw_text
            for page in pages
        )
        urls = re.findall(r"https://chem\.libretexts\.org/@go/page/\d+", source_url_text)
        document = Document(
            doc_id,
            relative,
            title,
            source_hash,
            reason,
            urls[0] if urls else None,
            pages,
            document_meta,
        )
        documents.append(document)
        if reason != "included":
            continue
        for page in pages:
            page_hash = digest(re.sub(r"\s+", " ", page.text))
            if page_hash in seen_pages or "low_text_content" in page.flags:
                page.flags.append("not_indexed_duplicate_or_low_text")
                continue
            if "quarantined_requires_ocr_review" in page.flags:
                continue
            seen_pages.add(page_hash)
            for ordinal, content in enumerate(split_page(page.text, title)):
                text = title + "\n\n" + content
                input_hash = digest(text)
                if input_hash in seen_chunks:
                    continue
                seen_chunks.add(input_hash)
                chunk_id = digest(
                    f"{PIPELINE_VERSION}|{doc_id}|{page.number}|{ordinal}|{input_hash}"
                )
                chunks.append(
                    Chunk(
                        chunk_id,
                        doc_id,
                        page.number,
                        ordinal,
                        content,
                        text,
                        input_hash,
                        token_count(text),
                    )
                )
    if not chunks:
        raise ValueError("Corpus selection produced no answer chunks")
    corpus_id = digest(
        json.dumps(
            {
                "version": PIPELINE_VERSION,
                "model": MODEL,
                "dimensions": DIMENSIONS,
                "chunks": [c.id for c in chunks],
                "sources": [(d.id, d.selection) for d in documents],
            },
            sort_keys=True,
        )
    )
    total_tokens = sum(c.tokens for c in chunks)
    summary = {
        "corpus_id": corpus_id,
        "pipeline_version": PIPELINE_VERSION,
        "primary_extractor": "pdf-inspector",
        "primary_extractor_version": INSPECTOR_VERSION,
        "extraction_policy": EXTRACTION_POLICY,
        "extraction_verified": False,
        "vision_required_pages": sum(
            bool(p.extraction_metadata.get("requires_vision")) for d in documents for p in d.pages
        ),
        "vision_completed_pages": sum(
            bool(p.extraction_metadata.get("vision_completed")) for d in documents for p in d.pages
        ),
        "vision_unresolved_review_pages": [
            f"{d.path}#page={p.number}"
            for d in documents
            for p in d.pages
            if any(
                flag in p.flags
                for flag in ("vision_low_confidence_review", "vision_low_text_coverage_review")
            )
        ],
        "page_extractor_counts": dict(
            Counter(p.extraction_metadata["extractor"] for d in documents for p in d.pages)
        ),
        "embedding_model": MODEL,
        "dimensions": DIMENSIONS,
        "documents": len(documents),
        "pages": sum(len(d.pages) for d in documents),
        "selection_counts": dict(Counter(d.selection for d in documents)),
        "chunks": len(chunks),
        "embedding_input_tokens": total_tokens,
        "max_chunk_tokens": max(c.tokens for c in chunks),
        "estimated_embedding_cost_usd": round(total_tokens / 1_000_000 * PRICE_PER_MILLION, 6),
        "quality_flags": dict(Counter(f for d in documents for p in d.pages for f in p.flags)),
    }
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "prepared.json"
    if previous.exists():
        previous_id = json.loads(previous.read_text(encoding="utf-8"))["summary"]["corpus_id"]
        if previous_id != corpus_id:
            archive = output / "snapshots" / previous_id
            archive.mkdir(parents=True, exist_ok=True)
            for old_file in output.glob("*.json"):
                shutil.copy2(old_file, archive / old_file.name)
    payload = {
        "summary": summary,
        "documents": [asdict(d) for d in documents],
        "chunks": [asdict(c) for c in chunks],
    }
    temporary = output / "prepared.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output / "prepared.json")
    (output / "preparation-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
