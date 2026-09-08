"""Verify every required visual page and aggregate provider usage without API calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

from talotrace.retrieval.vision import cache_path, load_cached, requires_vision


def usage_totals(records):
    return {
        "responses": len(records),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in records),
        "cached_input_tokens": sum(
            r["usage"].get("input_tokens_details", {}).get("cached_tokens", 0) for r in records
        ),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in records),
        "reasoning_tokens": sum(
            r["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0) for r in records
        ),
        "total_tokens": sum(r["usage"]["total_tokens"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, default=Path("data/rag/inspection.json"))
    parser.add_argument("--vision", type=Path, default=Path("data/tests/rag/vision"))
    args = parser.parse_args()
    inspection = json.loads(args.inspection.read_text(encoding="utf-8"))
    jobs = [(d, p) for d in inspection["documents"] for p in d["pages"] if requires_vision(p)]
    records, missing, issues = [], [], []
    for source, page in jobs:
        record = load_cached(args.vision, source["sha256"], page["number"])
        citation = f"{source['path']}#page={page['number']}"
        if record is None:
            missing.append(citation)
            continue
        records.append(record)
        transcription = record["transcription"]
        if (
            any(transcription[key] for key in ("source_errors", "unreadable_spans", "warnings"))
            or record["quality_flags"]
        ):
            issues.append(
                {
                    "citation": citation,
                    "cache_path": str(cache_path(args.vision, source["sha256"], page["number"])),
                    "quality_flags": record["quality_flags"],
                    "source_errors": transcription["source_errors"],
                    "unreadable_spans": transcription["unreadable_spans"],
                    "warnings": transcription["warnings"],
                    "confidence": transcription["confidence"],
                }
            )
    all_records = []
    for path in args.vision.glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and "usage" in record:
            all_records.append(record)
    report = {
        "passed": not missing and len(records) == len(jobs),
        "required_pages": len(jobs),
        "completed_pages": len(records),
        "missing": missing,
        "current_policy_usage": usage_totals(records),
        "all_saved_response_usage_including_superseded_samples": usage_totals(all_records),
        "model_resolved_counts": dict(Counter(r["model_resolved"] for r in records)),
        "quality_flags": dict(Counter(f for r in records for f in r["quality_flags"])),
        "confidence_self_assessment_counts": dict(
            Counter(r["transcription"]["confidence"] for r in records)
        ),
        "confidence_basis": "Uncalibrated model self-assessment; not independent verification",
        "pages_with_reported_issues": issues,
        "billing_limitation": (
            "Token totals count saved provider responses, not unobserved retries. "
            "No unverified dollar rate is assumed."
        ),
        "verification_api_calls": 0,
    }
    (args.vision / "audit-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "pages_with_reported_issues"}, indent=2)
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
