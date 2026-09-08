"""Exact pgvector cosine retrieval + BM25, fused using reciprocal rank."""

import re
from collections import defaultdict

import psycopg
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from .corpus import DIMENSIONS, MODEL, digest, token_count
from .embeddings import validated_vectors, vector_literal

STOP_WORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or the to "
    "was what when where which why with".split()
)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", "", text)
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in STOP_WORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """Reuse one instance per worker; refreshes BM25 when the active corpus changes."""

    def __init__(
        self, conn: psycopg.AsyncConnection, client: AsyncOpenAI, corpus_id: str | None = None
    ):
        self.conn = conn
        self.client = client
        self.corpus_id = None
        self.rows = []
        self.bm25 = None
        self.requested_corpus_id = corpus_id

    async def _load(self) -> None:
        corpus = await (
            await self.conn.execute(
                """SELECT id,model,dimensions FROM rag.corpora WHERE state='ready'
                   AND ((%s::text IS NULL AND active) OR id=%s)""",
                (self.requested_corpus_id, self.requested_corpus_id),
            )
        ).fetchone()
        if not corpus:
            raise ValueError("RAG corpus is not ready; complete embedding ingestion first")
        if corpus["model"] != MODEL or corpus["dimensions"] != DIMENSIONS:
            raise ValueError("Active corpus model does not match the retriever")
        if self.corpus_id == corpus["id"]:
            return
        cur = await self.conn.execute(
            """SELECT c.id,c.content,c.input_text,d.path,d.title,d.file_sha256,d.source_url,
               c.page_number,p.quality_flags,
               p.extraction_metadata->>'extractor' AS extractor,
               p.extraction_metadata->>'extractor_version' AS extractor_version,
               p.extraction_metadata - 'native_markdown' - 'diagnostic_pypdf_text'
                 AS extraction_provenance
               FROM rag.chunks c
               JOIN rag.corpus_chunks cc ON cc.chunk_id=c.id
               JOIN rag.documents d ON d.id=c.document_id
               JOIN rag.pages p ON p.document_id=c.document_id AND p.page_number=c.page_number
               WHERE cc.corpus_id=%s ORDER BY c.id""",
            (corpus["id"],),
        )
        self.rows = await cur.fetchall()
        if not self.rows:
            raise ValueError("Active RAG corpus is empty")
        self.bm25 = BM25Okapi([tokenize(row["input_text"]) for row in self.rows])
        self.corpus_id = corpus["id"]

    async def search(self, question: str, top_k: int = 6, candidates: int = 30) -> dict:
        question = question.strip()
        if not question or not tokenize(question) or token_count(question) > 8191:
            raise ValueError("Question must contain searchable text within 8191 tokens")
        if not 1 <= top_k <= candidates <= 100:
            raise ValueError("Require 1 <= top_k <= candidates <= 100")
        await self._load()
        input_hash = digest(question)
        cached = await (
            await self.conn.execute(
                """SELECT embedding::text AS vector FROM rag.embeddings
               WHERE input_sha256=%s AND model=%s AND dimensions=%s""",
                (input_hash, MODEL, DIMENSIONS),
            )
        ).fetchone()
        if cached:
            vector = cached["vector"]
        else:
            response = await self.client.embeddings.create(
                input=[question], model=MODEL, dimensions=DIMENSIONS, encoding_format="float"
            )
            vector = vector_literal(validated_vectors(response, 1)[0])
            async with self.conn.transaction():
                await self.conn.execute(
                    """INSERT INTO rag.embeddings(input_sha256,model,dimensions,embedding)
                       VALUES (%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING""",
                    (input_hash, MODEL, DIMENSIONS, vector),
                )
                await self.conn.execute(
                    """INSERT INTO rag.embedding_usage
                       (corpus_id,purpose,model,input_count,prompt_tokens,request_id)
                       VALUES (%s,'query',%s,1,%s,%s)""",
                    (
                        self.corpus_id,
                        MODEL,
                        response.usage.prompt_tokens,
                        getattr(response, "_request_id", None),
                    ),
                )
        dense = await (
            await self.conn.execute(
                """SELECT c.id,1-(e.embedding <=> %s::vector) AS similarity
               FROM rag.chunks c JOIN rag.corpus_chunks cc ON cc.chunk_id=c.id
               JOIN rag.embeddings e ON e.input_sha256=c.input_sha256
                 AND e.model=%s AND e.dimensions=%s
               WHERE cc.corpus_id=%s ORDER BY e.embedding <=> %s::vector,c.id LIMIT %s""",
                (vector, MODEL, DIMENSIONS, self.corpus_id, vector, candidates),
            )
        ).fetchall()
        lexical_scores = self.bm25.get_scores(tokenize(question))
        lexical = sorted(
            (
                (row["id"], float(score))
                for row, score in zip(self.rows, lexical_scores, strict=True)
                if score > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:candidates]
        dense_ids = [row["id"] for row in dense]
        lexical_ids = [item[0] for item in lexical]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
        by_id = {row["id"]: row for row in self.rows}
        results = []
        page_counts = defaultdict(int)
        for chunk_id, score in fused:
            row = by_id[chunk_id]
            page_key = (row["path"], row["page_number"])
            if page_counts[page_key] >= 2:
                continue
            page_counts[page_key] += 1
            results.append(
                {k: v for k, v in row.items() if k != "input_text"}
                | {
                    "rrf_score": score,
                    "dense_rank": dense_ids.index(chunk_id) + 1 if chunk_id in dense_ids else None,
                    "bm25_rank": lexical_ids.index(chunk_id) + 1
                    if chunk_id in lexical_ids
                    else None,
                    "citation": f"{row['path']}#page={row['page_number']}",
                }
            )
            if len(results) == top_k:
                break
        return {
            "question": question,
            "corpus_id": self.corpus_id,
            "model": MODEL,
            "strategy": "exact-cosine+bm25+rrf",
            "results": results,
        }
