"""Run with: uv run python -m talotrace.retrieval --help."""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import APIError

from .corpus import PRICE_PER_MILLION, inspect, prepare
from .embeddings import ingest, local_config, openai_client
from .hybrid import HybridRetriever
from .store import activate, connect, migrate, read_prepared, register_prepared, status

QUESTIONS = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
]


def write_report(output: Path, name: str, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


async def run(args) -> dict:
    repository = args.repository.resolve()
    output = repository / args.output
    report_output = (
        repository / "data/tests/rag/retrieval-checks"
        if args.command in {"query", "smoke", "validate", "activate"}
        else output
    )
    if args.command == "inspect":
        return inspect(
            (repository / args.corpus).resolve(),
            output,
            repository / "configs/rag-extraction-reviews.json",
        )
    if args.command == "prepare":
        return prepare(
            (repository / args.corpus).resolve(),
            output,
            repository / "configs/rag-extraction-reviews.json",
            repository / "data/tests/rag/vision",
        )
    values = local_config(repository)
    async with await connect(values["DATABASE_URL"]) as conn:
        await migrate(conn, repository)
        payload = read_prepared(output / "prepared.json")
        corpus_id = payload["summary"]["corpus_id"]
        if args.command == "register":
            await register_prepared(conn, payload)
            result = await status(conn, corpus_id)
        elif args.command == "status":
            result = await status(conn, corpus_id)
        else:
            async with openai_client(values) as client:
                if args.command == "ingest":
                    await register_prepared(conn, payload)
                    result = await ingest(conn, corpus_id, client)
                    result["estimated_ingestion_cost_usd"] = round(
                        result["billed_ingestion_tokens"] / 1_000_000 * PRICE_PER_MILLION, 6
                    )
                else:
                    retriever = HybridRetriever(
                        conn,
                        client,
                        corpus_id if args.command in {"validate", "activate"} else None,
                    )
                    questions = [args.question] if args.command == "query" else QUESTIONS
                    results = [await retriever.search(question) for question in questions]
                    result = {"queries": results}
                    if args.command in {"smoke", "validate", "activate"}:
                        expected = ["10.2:", "6.1:", "6.1:"]
                        checks = [
                            any(hit["title"].startswith(prefix) for hit in item["results"])
                            for prefix, item in zip(expected, results, strict=True)
                        ]
                        result["expected_topic_in_top_6"] = checks
                        concept_patterns = [
                            [r"logarithm|base.?10", r"hydronium"],
                            [r"potential energy", r"shar", r"nuclei"],
                            [r"shar", r"ions|cations", r"electrostatic"],
                        ]
                        concept_checks = [
                            all(
                                re.search(
                                    pattern,
                                    " ".join(hit["content"] for hit in item["results"]),
                                    re.IGNORECASE,
                                )
                                for pattern in patterns
                            )
                            for patterns, item in zip(concept_patterns, results, strict=True)
                        ]
                        result["required_concept_evidence"] = concept_checks
                        result["passed"] = all(checks) and all(concept_checks)
                        if not result["passed"]:
                            write_report(report_output, "smoke-report.json", result)
                            raise ValueError(
                                "Required-topic retrieval smoke failed; inspect report"
                            )
                        if args.command == "activate":
                            await activate(conn, corpus_id)
                            result["activated_corpus"] = await status(conn, corpus_id)
        write_report(report_output, f"{args.command}-report.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, ingest and verify chemistry RAG")
    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "prepare",
            "register",
            "ingest",
            "validate",
            "activate",
            "status",
            "query",
            "smoke",
        ],
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--corpus", type=Path, default=Path("../material/chem_material"))
    parser.add_argument("--output", type=Path, default=Path("data/rag"))
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.command == "query" and not args.question:
        parser.error("query requires --question")
    try:
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with asyncio.Runner(loop_factory=factory) as runner:
            result = runner.run(run(args))
    except APIError as exc:
        # API exception bodies may include sensitive data; log only category and HTTP status.
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "retry": "rerun ingest to resume",
                }
            )
        )
        raise SystemExit(1) from None
    except Exception as exc:
        # Never serialize database connection strings or provider response bodies.
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "stage": args.command,
                    "message": str(exc)
                    if isinstance(exc, ValueError)
                    else "Operation failed; inspect local configuration and service readiness",
                }
            )
        )
        raise SystemExit(1) from None
    if args.command in {"query", "smoke", "validate", "activate"}:
        compact = {
            "passed": result.get("passed"),
            "queries": [
                {
                    "question": item["question"],
                    "citations": [hit["citation"] for hit in item["results"]],
                }
                for item in result["queries"]
            ],
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
