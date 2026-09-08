"""pdf-inspector first; pypdf diagnoses defects but never supplies final answer text."""

import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

import pdf_inspector
from pypdf import PdfReader

INSPECTOR_VERSION = version("pdf-inspector")
FALLBACK_VERSION = version("pypdf")
EXTRACTION_IDENTITY = (
    f"pdf-inspector-{INSPECTOR_VERSION}-diagnostic-pypdf-{FALLBACK_VERSION}-vision-v1"
)


def extraction_identity(reviews: dict) -> str:
    review_hash = hashlib.sha256(json.dumps(reviews, sort_keys=True).encode()).hexdigest()
    return f"{EXTRACTION_IDENTITY}-reviews-{review_hash}"


def interleaved_reference_tokens(markdown: str) -> list[str]:
    # Detection only: these tokens are never rewritten into invented chemistry.
    # Ignore URLs and credits so their alphanumeric paths do not trigger fallback.
    text = re.split(r"(?im)^(?:#+\s*)?Contributors and Attributions|^This page titled", markdown)[0]
    text = re.sub(r"https?://\S+", "", text)
    pattern = r"[A-Za-z][A-Za-z0-9.*-]*[0-9][A-Za-z][A-Za-z0-9.*-]*"
    return sorted({m.group() for m in re.finditer(pattern, text) if m.group().count(".") >= 2})


def confirmed_interleaving(markdown: str, alternate: str) -> list[str]:
    """A malformed native token must resolve to letters actually present in alternate PDF text."""
    alternate_letters = re.sub(r"[^a-z]", "", alternate.casefold())
    confirmed = []
    for token in interleaved_reference_tokens(markdown):
        letters = re.sub(r"[^a-z]", "", token.casefold())
        if len(letters) >= 5 and letters in alternate_letters and token not in alternate:
            confirmed.append(token)
    return confirmed


def extract_pdf(
    path: Path, source_sha256: str = "", reviews: dict | None = None
) -> tuple[dict, list[dict]]:
    reviews = reviews or {}
    if reviews and reviews.get("primary_extractor_version") != INSPECTOR_VERSION:
        raise ValueError("Reviewed fallback decisions require revalidation for this parser version")
    classification = pdf_inspector.classify_pdf(str(path))
    result = pdf_inspector.extract_pages_markdown(str(path))
    expected = set(range(classification.page_count))
    observed = [page.page for page in result.pages]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f"pdf-inspector returned invalid page provenance for {path.name}")
    document_meta = {
        "primary_extractor": "pdf-inspector",
        "primary_version": INSPECTOR_VERSION,
        "extraction_identity": extraction_identity(reviews),
        "pdf_type": classification.pdf_type,
        "confidence": classification.confidence,
        "is_complex": result.is_complex,
        "pages_with_tables": list(result.pages_with_tables),
        "pages_with_columns": list(result.pages_with_columns),
        "pages_needing_ocr": list(result.pages_needing_ocr),
    }
    reader = None
    pages = []
    for native in sorted(result.pages, key=lambda page: page.page):
        number = native.page + 1  # This API alone uses 0-based page indices.
        raw = native.markdown
        flags = []
        reasons, confirmed = [], []
        suspects = interleaved_reference_tokens(native.markdown)
        reviewed = [
            review
            for review in reviews.get("reviews", [])
            if review["source_sha256"] == source_sha256
            and review["physical_page"] == number
            and review["action"] == "vision"
        ]
        meta = {
            "extractor": "pdf-inspector",
            "extractor_version": INSPECTOR_VERSION,
            "primary_extractor": "pdf-inspector",
            "primary_version": INSPECTOR_VERSION,
            "physical_page": number,
            "native_needs_ocr": native.needs_ocr,
            "native_ocr_reason": native.ocr_reason,
            "has_tables": number in result.pages_with_tables,
            "has_columns": number in result.pages_with_columns,
            "suspected_interleaved_tokens": suspects,
        }
        source_reviews = [
            review
            for review in reviews.get("reviews", [])
            if review["source_sha256"] == source_sha256
            and review["physical_page"] == number
            and review["action"] == "warning"
        ]
        if source_reviews:
            flags.extend(review["reason"] for review in source_reviews)
            meta["source_quality_reviews"] = source_reviews
        if native.needs_ocr or suspects or reviewed:
            if reader is None:
                reader = PdfReader(path)
                if len(reader.pages) != classification.page_count:
                    raise ValueError(f"Parsers disagree about page count for {path.name}")
            alternate = reader.pages[native.page].extract_text() or ""
            confirmed = confirmed_interleaving(native.markdown, alternate)
            if native.needs_ocr:
                flags.append("pdf_inspector_requires_ocr")
                reasons.append(native.ocr_reason or "native_requires_ocr")
            if confirmed:
                flags.append("pdf_inspector_layout_interleaving")
                reasons.append("reference_digits_interleaved_into_words_confirmed_by_alternate")
            if reviewed:
                flags.append("reviewed_page_vision_required")
                reasons.extend(review["reason"] for review in reviewed)
                meta["reviewed_fallback_evidence"] = [review["evidence"] for review in reviewed]
            if reasons:
                # Both native and diagnostic outputs are evidence, never an answer fallback.
                meta["native_markdown"] = native.markdown
                meta["vision_routing_reasons"] = reasons
                meta["confirmed_interleaved_tokens"] = confirmed
                meta["diagnostic_pypdf_text"] = alternate
                meta["diagnostic_pypdf_version"] = FALLBACK_VERSION
                meta["requires_vision"] = True
                flags.append("vision_required")
        if suspects and not confirmed:
            flags.append("possible_layout_interleaving_review")
        combined = native.markdown + "\n" + raw
        if re.search(r"Unknown\s+node|node\s+type:\s*sup", combined, re.IGNORECASE):
            flags.append("source_formula_rendering_error")
            flags.append("vision_required")
            meta["requires_vision"] = True
        if "<sup>" in raw or "<sub>" in raw or re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]", raw):
            flags.append("math_transcription_unverified")
        if meta["has_tables"]:
            flags.append("table_layout_requires_verification")
        if "\ufffd" in raw:
            flags.append("replacement_character")
        pages.append(
            {
                "number": number,
                "raw_text": raw,
                "flags": sorted(set(flags)),
                "extraction_metadata": meta,
            }
        )
    return document_meta, pages
