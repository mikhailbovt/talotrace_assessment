"""Compare preserved extraction snapshots; writes review artifacts only under data/tests."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, default=Path("data/rag/prepared.json"))
    parser.add_argument("--output", type=Path, default=Path("data/tests/rag/parser-comparison"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    before = {document["path"]: document for document in baseline["documents"]}
    after = {document["path"]: document for document in current["documents"]}
    assert before.keys() == after.keys(), "Source inventory changed"
    assert all(before[path]["sha256"] == doc["sha256"] for path, doc in after.items())
    old_hashes = {chunk["input_sha256"] for chunk in baseline["chunks"]}
    changed = [chunk for chunk in current["chunks"] if chunk["input_sha256"] not in old_hashes]
    page_extractors = Counter(
        page["extraction_metadata"]["extractor"]
        for document in current["documents"]
        for page in document["pages"]
    )
    pages_by_id = {
        (doc["id"], page["number"]): page for doc in current["documents"] for page in doc["pages"]
    }
    chunk_extractors = Counter(
        pages_by_id[(chunk["document_id"], chunk["page"])]["extraction_metadata"]["extractor"]
        for chunk in current["chunks"]
    )
    summary = {
        "baseline_corpus": baseline["summary"]["corpus_id"],
        "new_corpus": current["summary"]["corpus_id"],
        "source_hashes_unchanged": len(after),
        "page_extractor_counts": dict(page_extractors),
        "chunk_extractor_counts": dict(chunk_extractors),
        "reused_inputs": len(current["chunks"]) - len(changed),
        "changed_inputs": len(changed),
        "changed_input_tokens": sum(c["tokens"] for c in changed),
        "quality_flags": current["summary"]["quality_flags"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    samples = [
        ("10.2!", 1),
        ("10.2!", 4),
        ("6.1!", 4),
        ("6.1!", 7),
        ("6.1!", 8),
        ("11.5!", 4),
        ("13.2!", 1),
    ]
    for prefix, page_number in samples:
        path = next(path for path in after if Path(path).name.startswith(prefix))
        old_page = before[path]["pages"][page_number - 1]
        new_page = after[path]["pages"][page_number - 1]
        meta = new_page["extraction_metadata"]
        native = meta.get("native_markdown", new_page["raw_text"])
        review = (
            f"# {path}, physical PDF page {page_number}\n\n"
            f"Actual supplier: {meta['extractor']} {meta['extractor_version']}\n\n"
            f"Flags: {', '.join(new_page['flags']) or 'none'}\n\n"
            f"Vision routing reasons: {meta.get('vision_routing_reasons', [])}\n\n"
            f"Vision cache identity: {meta.get('vision_cache_identity', {})}\n\n"
            f"Source errors: {meta.get('source_errors', [])}\n\n"
            f"Unreadable spans: {meta.get('unreadable_spans', [])}\n\n"
            f"Source quality reviews: {meta.get('source_quality_reviews', [])}\n\n"
            "## Previous pypdf text\n\n```text\n" + old_page["raw_text"] + "\n```\n\n"
            "## Original pdf-inspector Markdown\n\n```text\n" + native + "\n```\n\n"
            "## Selected answer text\n\n```text\n" + new_page["text"] + "\n```\n"
        )
        (args.output / f"{prefix.rstrip('!')}-page-{page_number}-comparison.md").write_text(
            review, encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
