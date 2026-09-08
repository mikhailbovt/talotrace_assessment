"""Repeat registration/ingestion with an API client that fails on any embedding call."""

import asyncio
import json
import sys
from pathlib import Path

from talotrace.retrieval.embeddings import ingest, local_config
from talotrace.retrieval.store import connect, read_prepared, register_prepared, status


class NoEmbeddingCalls:
    def __init__(self):
        self.embeddings = self
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        raise AssertionError("Idempotence check attempted an additional paid embedding call")


async def database_totals(conn):
    return await (
        await conn.execute(
            """SELECT (SELECT count(*) FROM rag.embeddings) vectors,
               (SELECT count(*) FROM rag.embedding_usage) usage_rows,
               (SELECT coalesce(sum(prompt_tokens),0) FROM rag.embedding_usage) billed_tokens,
               (SELECT count(*) FROM rag.documents) documents,
               (SELECT count(*) FROM rag.pages) pages,
               (SELECT count(*) FROM rag.chunks) chunks"""
        )
    ).fetchone()


async def main():
    repository = Path.cwd()
    payload = read_prepared(repository / "data/rag/prepared.json")
    corpus_id = payload["summary"]["corpus_id"]
    client = NoEmbeddingCalls()
    async with await connect(local_config(repository)["DATABASE_URL"]) as conn:
        before = await database_totals(conn)
        await register_prepared(conn, payload)
        repeated = await ingest(conn, corpus_id, client)
        after = await database_totals(conn)
        state = await status(conn, corpus_id)
    report = {
        "passed": before == after and client.calls == 0 and repeated["new_embedding_inputs"] == 0,
        "corpus_id": corpus_id,
        "embedding_api_calls": client.calls,
        "before": before,
        "after": after,
        "repeated_ingestion": repeated,
        "final_status": state,
    }
    output = repository / "data/tests/rag/retrieval-checks/idempotence-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(main())
