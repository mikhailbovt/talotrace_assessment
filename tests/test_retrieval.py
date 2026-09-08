import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from talotrace.retrieval import pdf_extraction, vision
from talotrace.retrieval.corpus import (
    DIMENSIONS,
    MODEL,
    Page,
    apply_visual_transcription,
    clean_page,
    selection_reason,
    split_page,
    token_count,
)
from talotrace.retrieval.embeddings import batches, local_config, openai_client, validated_vectors
from talotrace.retrieval.hybrid import reciprocal_rank_fusion, tokenize
from talotrace.retrieval.store import visual_extraction_complete


def test_long_chemical_page_preserves_unicode_and_token_budget():
    text = "H₃O⁺, OH⁻ and ΔG describe chemistry. 共价键形成。\n" * 300
    chunks = split_page(text, "Chemistry", max_tokens=160, overlap=20)
    assert len(chunks) > 1
    assert all("\ufffd" not in chunk for chunk in chunks)
    assert all(token_count("Chemistry\n\n" + chunk) <= 160 for chunk in chunks)
    assert chunks[0].startswith("H₃O⁺")
    assert chunks[-1].endswith("共价键形成。")


def test_cleanup_keeps_source_error_and_chemistry_but_removes_footer_and_credits():
    raw = (
        "10.2.1 https://chem.libretexts.org/@go/page/431450\n"
        "pH = −log [H₃O⁺]\nUnknown node type: sup\n"
        "Contributors and Attributions\nCredit names"
    )
    clean = clean_page(raw)
    assert "−log [H₃O⁺]" in clean
    assert "Unknown node type: sup" in clean
    assert "https://" not in clean
    assert "Credit names" not in clean
    with_diagram = raw + "\n\n## Visible diagram descriptions\n\nVisible potential-energy curve."
    assert "Visible potential-energy curve." in clean_page(with_diagram)
    assert "Credit names" not in clean_page(with_diagram)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("6! Chemical Bonding.pdf", "excluded_chapter_overview"),
        ("6.1! Ionic and Covalent Bonding.pdf", "included"),
        ("6.E! Homework.pdf", "excluded_exercises"),
        ("99999!A Index.pdf", "excluded_index"),
        ("99999!C Detailed Licensing.pdf", "excluded_licensing"),
        ("99999!B Glossary.pdf", "excluded_placeholder_glossary"),
    ],
)
def test_non_teaching_sources_do_not_pollute_answer_index(name, expected):
    pages = [Page(1, "", "Sample Word 1 | Sample Definition 1", [])]
    assert selection_reason(Path(name), pages) == expected


def test_batches_obey_both_limits_without_losing_work():
    rows = [{"token_count": n, "id": i} for i, n in enumerate([6000, 6000, 4000, 3000, 8000])]
    result = list(batches(rows, max_tokens=12_000, max_items=2))
    assert [row for batch in result for row in batch] == rows
    assert all(len(batch) <= 2 for batch in result)
    assert all(sum(row["token_count"] for row in batch) <= 12_000 for batch in result)
    with pytest.raises(ValueError):
        list(batches([{"token_count": 8192}]))


def test_provider_vectors_require_count_dimensions_indices_and_finite_norm():
    vector = [1.0] + [0.0] * (DIMENSIONS - 1)
    response = SimpleNamespace(model=MODEL, data=[SimpleNamespace(index=0, embedding=vector)])
    assert validated_vectors(response, 1) == [vector]
    response.data[0].embedding = [float("nan")] + vector[1:]
    with pytest.raises(ValueError):
        validated_vectors(response, 1)
    response.data[0].embedding = [0.0] * DIMENSIONS
    with pytest.raises(ValueError):
        validated_vectors(response, 1)
    response.data[0].embedding = vector[:-1]
    with pytest.raises(ValueError):
        validated_vectors(response, 1)


def test_fusion_rewards_agreement_and_ignores_duplicate_rank_entries():
    fused = reciprocal_rank_fusion([["dense", "shared", "dense"], ["lexical", "shared"]])
    assert fused[0][0] == "shared"
    assert len(fused) == 3


def test_tokenization_retains_ph_and_chemical_formulas():
    assert tokenize("How does the pH of H2O work?") == ["ph", "h2o", "work"]


def test_missing_project_key_never_falls_back_to_process_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-key-not-authorized")
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://local\nOPENAI_API_KEY=\n")
    values = local_config(tmp_path)
    with pytest.raises(ValueError, match="no process-key fallback"):
        openai_client(values)


def inspector_result(monkeypatch, pages):
    monkeypatch.setattr(
        pdf_extraction.pdf_inspector,
        "classify_pdf",
        lambda _: SimpleNamespace(page_count=len(pages), pdf_type="text_based", confidence=1.0),
    )
    monkeypatch.setattr(
        pdf_extraction.pdf_inspector,
        "extract_pages_markdown",
        lambda _: SimpleNamespace(
            pages=pages,
            pages_with_tables=[],
            pages_with_columns=[],
            pages_needing_ocr=[],
            is_complex=False,
        ),
    )


def native_page(number, text, needs_ocr=False):
    return SimpleNamespace(
        page=number,
        markdown=text,
        needs_ocr=needs_ocr,
        ocr_reason="suspected_garbled_text" if needs_ocr else None,
    )


def test_inspector_markdown_is_primary_and_page_indices_are_converted_once(monkeypatch):
    markdown = "Hydrogen H₂ forms a stable molecule with a covalent bond."
    inspector_result(monkeypatch, [native_page(0, markdown), native_page(1, markdown)])
    monkeypatch.setattr(pdf_extraction, "PdfReader", lambda _: pytest.fail("Fallback not expected"))
    _, pages = pdf_extraction.extract_pdf(Path("unused.pdf"))
    assert [page["number"] for page in pages] == [1, 2]
    assert all(page["raw_text"] == markdown for page in pages)
    assert all(page["extraction_metadata"]["extractor"] == "pdf-inspector" for page in pages)


def test_native_corruption_routes_to_vision_without_using_diagnostic_answer_text(monkeypatch):
    native = "The potential6e.n1e.4rgy decreases as the atoms approach each other."
    alternate = "The potential energy decreases as the atoms approach each other."
    inspector_result(monkeypatch, [native_page(0, native)])
    monkeypatch.setattr(
        pdf_extraction,
        "PdfReader",
        lambda _: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: alternate)]),
    )
    _, pages = pdf_extraction.extract_pdf(Path("unused.pdf"))
    page = pages[0]
    assert page["raw_text"] == native
    assert page["extraction_metadata"]["native_markdown"] == native
    assert page["extraction_metadata"]["diagnostic_pypdf_text"] == alternate
    assert page["extraction_metadata"]["extractor"] == "pdf-inspector"
    assert page["extraction_metadata"]["requires_vision"]
    assert "pdf_inspector_layout_interleaving" in page["flags"]


def test_rejected_page_without_native_text_requires_visual_transcription(monkeypatch):
    inspector_result(monkeypatch, [native_page(0, "", True)])
    monkeypatch.setattr(
        pdf_extraction,
        "PdfReader",
        lambda _: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "")]),
    )
    _, pages = pdf_extraction.extract_pdf(Path("unused.pdf"))
    assert "vision_required" in pages[0]["flags"]
    assert vision.requires_vision(pages[0])


def test_invalid_page_mapping_fails_instead_of_assigning_wrong_citations(monkeypatch):
    inspector_result(monkeypatch, [native_page(0, "first"), native_page(0, "duplicate")])
    with pytest.raises(ValueError, match="page provenance"):
        pdf_extraction.extract_pdf(Path("unused.pdf"))


def test_chemical_digits_are_not_treated_as_interleaved_references():
    assert pdf_extraction.interleaved_reference_tokens("H2O, C6H12O6, 1.0e-7 and pH 7.00") == []
    assert tokenize("H<sub>3</sub>O<sup>+</sup>") == ["h3o"]


def test_missing_visual_cache_blocks_preparation_instead_of_falling_back(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "load_cached", lambda *_: None)
    extracted = {
        "number": 3,
        "raw_text": "corrupt native text",
        "flags": ["vision_required"],
        "extraction_metadata": {"extractor": "pdf-inspector", "requires_vision": True},
    }
    with pytest.raises(ValueError, match="Required visual page transcription is missing"):
        apply_visual_transcription(extracted, "source-sha", tmp_path)


def visual_record():
    return {
        "identity": {"model": vision.VISION_MODEL},
        "model_resolved": vision.VISION_MODEL,
        "response_id": "response",
        "request_id": "request",
        "usage": {"input_tokens": 1},
        "quality_flags": [],
        "transcription": {
            "page_markdown": "Visible formula [unreadable]: Unknown node type: sup",
            "unreadable_spans": ["covered exponent"],
            "source_errors": ["error overlay"],
            "diagram_descriptions": [],
            "warnings": [],
            "confidence": "high",
        },
    }


def test_visual_source_errors_and_original_parser_evidence_reach_provenance(monkeypatch, tmp_path):
    record = visual_record()
    monkeypatch.setattr(vision, "load_cached", lambda *_: record)
    extracted = {
        "number": 1,
        "raw_text": "original native output",
        "flags": ["vision_required", "source_formula_rendering_error"],
        "extraction_metadata": {"extractor": "pdf-inspector", "requires_vision": True},
    }
    result = apply_visual_transcription(extracted, "source-sha", tmp_path)
    assert result["raw_text"] == record["transcription"]["page_markdown"]
    assert {"vision_unreadable_spans", "vision_source_errors"} <= set(result["flags"])
    assert "vision_required" not in result["flags"]
    provenance = result["extraction_metadata"]
    assert provenance["extractor"] == "gpt-5.6-luna-vision"
    assert provenance["native_markdown"] == "original native output"
    assert provenance["unreadable_spans"] == ["covered exponent"]
    assert provenance["source_errors"] == ["error overlay"]
    record["transcription"]["source_errors"] = []
    with pytest.raises(ValueError, match="omitted a known visible source error"):
        apply_visual_transcription(extracted, "source-sha", tmp_path)


def test_vision_cache_identity_changes_with_prompt_model_and_view_strategy(monkeypatch, tmp_path):
    original = vision.cache_path(tmp_path, "source", 1)
    assert original != vision.cache_path(tmp_path, "source", 2)
    assert original != vision.cache_path(tmp_path, "different-source", 1)
    for name in ["PROMPT_VERSION", "VISION_MODEL", "VIEW_STRATEGY", "RENDER_VERSION"]:
        with monkeypatch.context() as scope:
            scope.setattr(vision, name, "changed-version")
            assert original != vision.cache_path(tmp_path, "source", 1)


def test_visual_publication_gate_rejects_missing_counts_pending_reviews_and_legacy_fallback():
    summary = {
        "vision_required_pages": 182,
        "vision_completed_pages": 182,
        "vision_unresolved_review_pages": [],
        "page_extractor_counts": {"pdf-inspector": 299, "gpt-5.6-luna-vision": 182},
    }
    assert visual_extraction_complete(summary)
    assert not visual_extraction_complete({})
    assert not visual_extraction_complete(summary | {"vision_completed_pages": 181})
    assert not visual_extraction_complete(summary | {"vision_unresolved_review_pages": ["p1"]})
    assert not visual_extraction_complete(summary | {"page_extractor_counts": {"pypdf": 1}})


def test_reviewed_source_unit_warning_is_hash_bound_and_never_rewrites_text(monkeypatch):
    native = "Internuclear distance (pm): 0.74"
    inspector_result(monkeypatch, [native_page(0, native)])
    review = {
        "primary_extractor_version": pdf_extraction.INSPECTOR_VERSION,
        "reviews": [
            {
                "source_sha256": "exact-source",
                "physical_page": 1,
                "action": "warning",
                "reason": "source_unit_inconsistency",
                "evidence": "Reviewed original labels",
                "downstream_constraint": "Avoid 0.74 pm",
            }
        ],
    }
    _, pages = pdf_extraction.extract_pdf(Path("unused.pdf"), "exact-source", review)
    assert pages[0]["raw_text"] == native
    assert "source_unit_inconsistency" in pages[0]["flags"]
    assert pages[0]["extraction_metadata"]["source_quality_reviews"][0]["downstream_constraint"]
    _, other_pages = pdf_extraction.extract_pdf(Path("unused.pdf"), "changed-source", review)
    assert "source_unit_inconsistency" not in other_pages[0]["flags"]


def test_vision_sends_three_actual_high_detail_views_and_reuses_verified_cache(
    monkeypatch, tmp_path
):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"test PDF source bytes")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
        "fN8AAAAASUVORK5CYII="
    )

    def render_stub(pdf_path, number, image_path):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(png)
        views = []
        for label in ["top", "bottom"]:
            path = image_path.with_name(f"{number}.{label}.png")
            path.write_bytes(png)
            views.append({"name": label, "filename": path.name, "image_sha256": vision.sha(png)})
        return {"width": 1, "height": 1, "image_sha256": vision.sha(png), "views": views}

    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            model=vision.VISION_MODEL,
            id="response",
            _request_id="request",
            output_parsed=vision.PageTranscription(
                page_markdown="Visible chemistry text",
                unreadable_spans=[],
                source_errors=[],
                diagram_descriptions=[],
                warnings=[],
                confidence="high",
            ),
            usage=SimpleNamespace(total_tokens=2, model_dump=lambda: {"total_tokens": 2}),
        )

    monkeypatch.setattr(vision, "render_page", render_stub)
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    source = {"sha256": vision.sha(pdf.read_bytes()), "path": pdf.name}
    page = {"number": 1, "raw_text": "Visible chemistry text", "extraction_metadata": {}}
    first = asyncio.run(vision.transcribe_page(client, source, page, tmp_path, tmp_path / "cache"))
    second = asyncio.run(vision.transcribe_page(client, source, page, tmp_path, tmp_path / "cache"))
    assert not first["cached"] and second["cached"] and len(calls) == 1
    inputs = calls[0]["input"][0]["content"]
    images = [item for item in inputs if item["type"] == "input_image"]
    assert len(images) == 3 and all(item["detail"] == "high" for item in images)
    assert all(base64.b64decode(item["image_url"].split(",", 1)[1]) == png for item in images)
    image_path = tmp_path / "cache" / source["sha256"] / "1.png"
    image_path.write_bytes(b"corrupted image")
    with pytest.raises(ValueError, match="image is missing or changed"):
        vision.load_cached(tmp_path / "cache", source["sha256"], 1)
