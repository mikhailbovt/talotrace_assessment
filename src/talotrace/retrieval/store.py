"""PostgreSQL persistence; incomplete snapshots never become active retrieval corpora."""

import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .corpus import DIMENSIONS, EXTRACTION_POLICY, MODEL


async def connect(url: str) -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(url, autocommit=True, row_factory=dict_row)


async def migrate(conn: psycopg.AsyncConnection, repository: Path) -> None:
    async with conn.transaction():
        for name in ("002-rag.sql", "003-rag-extraction-provenance.sql"):
            await conn.execute((repository / "infra/postgres" / name).read_text(encoding="utf-8"))


async def register_prepared(conn: psycopg.AsyncConnection, payload: dict) -> None:
    summary = payload["summary"]
    async with conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            """INSERT INTO rag.corpora(id, model, dimensions, state, summary)
               VALUES (%s,%s,%s,'prepared',%s) ON CONFLICT (id) DO NOTHING""",
            (summary["corpus_id"], MODEL, DIMENSIONS, Jsonb(summary)),
        )
        for doc in payload["documents"]:
            await cur.execute(
                """INSERT INTO rag.documents
                   (id,path,title,file_sha256,source_url,page_count,extraction_metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO NOTHING""",
                (
                    doc["id"],
                    doc["path"],
                    doc["title"],
                    doc["sha256"],
                    doc["source_url"],
                    len(doc["pages"]),
                    Jsonb(doc.get("extraction_metadata", {})),
                ),
            )
            await cur.executemany(
                """INSERT INTO rag.pages
                   (document_id,page_number,raw_text,clean_text,quality_flags,extraction_metadata)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (document_id,page_number) DO NOTHING""",
                [
                    (
                        doc["id"],
                        p["number"],
                        p["raw_text"],
                        p["text"],
                        Jsonb(p["flags"]),
                        Jsonb(p.get("extraction_metadata", {})),
                    )
                    for p in doc["pages"]
                ],
            )
            await cur.execute(
                "INSERT INTO rag.corpus_documents VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (summary["corpus_id"], doc["id"], doc["selection"]),
            )
        await cur.executemany(
            """INSERT INTO rag.chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO NOTHING""",
            [
                (
                    c["id"],
                    c["document_id"],
                    c["page"],
                    c["ordinal"],
                    c["content"],
                    c["input_text"],
                    c["input_sha256"],
                    c["tokens"],
                )
                for c in payload["chunks"]
            ],
        )
        await cur.executemany(
            "INSERT INTO rag.corpus_chunks VALUES (%s,%s) ON CONFLICT DO NOTHING",
            [(summary["corpus_id"], c["id"]) for c in payload["chunks"]],
        )


async def pending(conn: psycopg.AsyncConnection, corpus_id: str) -> list[dict]:
    cur = await conn.execute(
        """SELECT DISTINCT c.input_sha256,c.input_text,c.token_count
           FROM rag.chunks c JOIN rag.corpus_chunks cc ON cc.chunk_id=c.id
           LEFT JOIN rag.embeddings e ON e.input_sha256=c.input_sha256
             AND e.model=%s AND e.dimensions=%s
           WHERE cc.corpus_id=%s AND e.input_sha256 IS NULL ORDER BY c.input_sha256""",
        (MODEL, DIMENSIONS, corpus_id),
    )
    return await cur.fetchall()


async def status(conn: psycopg.AsyncConnection, corpus_id: str) -> dict:
    cur = await conn.execute(
        """SELECT c.id,c.state,c.active,c.model,c.dimensions,
           (SELECT count(*) FROM rag.corpus_documents d WHERE d.corpus_id=c.id) documents,
           (SELECT count(*) FROM rag.corpus_documents d WHERE d.corpus_id=c.id
             AND d.selection='included') included_documents,
           (SELECT count(*) FROM rag.corpus_documents d JOIN rag.pages p
             ON p.document_id=d.document_id WHERE d.corpus_id=c.id) pages,
           (SELECT count(*) FROM rag.corpus_chunks cc WHERE cc.corpus_id=c.id) chunks,
           (SELECT count(*) FROM rag.corpus_chunks cc JOIN rag.chunks ch ON ch.id=cc.chunk_id
             JOIN rag.embeddings e ON e.input_sha256=ch.input_sha256 AND e.model=c.model
             AND e.dimensions=c.dimensions WHERE cc.corpus_id=c.id) embedded_chunks,
           (SELECT coalesce(sum(u.prompt_tokens),0) FROM rag.embedding_usage u
             WHERE u.corpus_id=c.id AND u.purpose='ingest') billed_ingestion_tokens,
           (SELECT count(*) FROM rag.embedding_usage u
             WHERE u.corpus_id=c.id AND u.purpose='ingest') ingestion_batches,
           (SELECT coalesce(sum(u.prompt_tokens),0) FROM rag.embedding_usage u
             WHERE u.corpus_id=c.id AND u.purpose='query') billed_query_tokens
           FROM rag.corpora c WHERE c.id=%s""",
        (corpus_id,),
    )
    result = await cur.fetchone()
    if result is None:
        raise ValueError("Prepared corpus is not registered in the database")
    return result


async def activate(conn: psycopg.AsyncConnection, corpus_id: str) -> None:
    async with conn.transaction():
        metadata = await (
            await conn.execute("SELECT summary FROM rag.corpora WHERE id=%s", (corpus_id,))
        ).fetchone()
        summary = metadata["summary"] if metadata else {}
        if summary.get("extraction_policy") != EXTRACTION_POLICY:
            raise ValueError("Refusing to publish a corpus with an obsolete extraction policy")
        if not visual_extraction_complete(summary):
            raise ValueError("Visual extraction is incomplete or has unresolved quality reviews")
        result = await status(conn, corpus_id)
        if result["chunks"] == 0 or result["chunks"] != result["embedded_chunks"]:
            raise ValueError("Refusing to activate incomplete corpus")
        await conn.execute("SELECT pg_advisory_xact_lock(728421)")
        await conn.execute(
            "UPDATE rag.corpora SET active=false WHERE active AND id<>%s", (corpus_id,)
        )
        await conn.execute(
            """UPDATE rag.corpora SET active=true,state='ready',
               summary=jsonb_set(summary,'{extraction_verified}','true'::jsonb) WHERE id=%s""",
            (corpus_id,),
        )


async def mark_ready(conn: psycopg.AsyncConnection, corpus_id: str) -> None:
    result = await status(conn, corpus_id)
    if result["chunks"] == 0 or result["chunks"] != result["embedded_chunks"]:
        raise ValueError("Cannot mark an incomplete corpus ready for validation")
    await conn.execute("UPDATE rag.corpora SET state='ready' WHERE id=%s", (corpus_id,))


def visual_extraction_complete(summary: dict) -> bool:
    required = summary.get("vision_required_pages")
    completed = summary.get("vision_completed_pages")
    return (
        type(required) is int
        and type(completed) is int
        and required >= 0
        and required == completed
        and summary.get("vision_unresolved_review_pages") == []
        and set(summary.get("page_extractor_counts", {}))
        <= {"pdf-inspector", "gpt-5.6-luna-vision"}
    )


def read_prepared(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary["embedding_model"] != MODEL or summary["dimensions"] != DIMENSIONS:
        raise ValueError("Prepared corpus embedding configuration does not match this runtime")
    return payload
