"""Bounded OpenAI embedding batches and durable resume checkpoints."""

import math
from collections.abc import Iterator
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from openai import AsyncOpenAI

from .corpus import DIMENSIONS, MODEL
from .store import mark_ready, pending, status


def local_config(repository: Path) -> dict:
    # Do not silently consume this machine's pre-existing process API credential.
    values = dotenv_values(repository / ".env", interpolate=False)
    if not values.get("DATABASE_URL"):
        raise ValueError("DATABASE_URL must be configured in the repository .env")
    return values


def openai_client(values: dict) -> AsyncOpenAI:
    key = values.get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("Project OpenAI key is missing from .env; no process-key fallback is used")
    return AsyncOpenAI(
        api_key=key,
        base_url="https://api.openai.com/v1",
        organization="",
        project="",
        timeout=90.0,
        max_retries=2,
    )


def batches(
    rows: list[dict], max_tokens: int = 16_000, max_items: int = 32
) -> Iterator[list[dict]]:
    batch, size = [], 0
    for row in rows:
        count = row["token_count"]
        if not 0 < count <= min(8191, max_tokens):
            raise ValueError("Embedding input is empty or exceeds the safe token limit")
        if batch and (size + count > max_tokens or len(batch) >= max_items):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += count
    if batch:
        yield batch


def validated_vectors(response, expected: int) -> list[list[float]]:
    if response.model != MODEL or len(response.data) != expected:
        raise ValueError("Embedding provider returned the wrong model or item count")
    if sorted(item.index for item in response.data) != list(range(expected)):
        raise ValueError("Embedding provider returned invalid item indices")
    vectors = []
    for item in sorted(response.data, key=lambda item: item.index):
        vector = item.embedding
        if len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding provider returned invalid vector dimensions or values")
        norm = math.sqrt(sum(value * value for value in vector))
        if not 0.95 <= norm <= 1.05:
            raise ValueError("Embedding provider returned a zero or non-normalized vector")
        vectors.append(vector)
    return vectors


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


async def ingest(conn: psycopg.AsyncConnection, corpus_id: str, client: AsyncOpenAI) -> dict:
    lock = await (await conn.execute("SELECT pg_try_advisory_lock(728420) AS locked")).fetchone()
    if not lock["locked"]:
        raise ValueError("Another RAG ingestion is running; retry after it completes")
    try:
        rows = await pending(conn, corpus_id)
        completed_batches, prompt_tokens = 0, 0
        if rows:
            await conn.execute(
                "UPDATE rag.corpora SET state='embedding' WHERE id=%s AND NOT active", (corpus_id,)
            )
        for index, batch in enumerate(batches(rows), 1):
            response = await client.embeddings.create(
                model=MODEL,
                dimensions=DIMENSIONS,
                input=[row["input_text"] for row in batch],
                encoding_format="float",
            )
            vectors = validated_vectors(response, len(batch))
            async with conn.transaction(), conn.cursor() as cur:
                await cur.executemany(
                    """INSERT INTO rag.embeddings(input_sha256,model,dimensions,embedding)
                       VALUES (%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING""",
                    [
                        (row["input_sha256"], MODEL, DIMENSIONS, vector_literal(vector))
                        for row, vector in zip(batch, vectors, strict=True)
                    ],
                )
                await cur.execute(
                    """INSERT INTO rag.embedding_usage
                       (corpus_id,purpose,model,input_count,prompt_tokens,request_id)
                       VALUES (%s,'ingest',%s,%s,%s,%s)""",
                    (
                        corpus_id,
                        MODEL,
                        len(batch),
                        response.usage.prompt_tokens,
                        getattr(response, "_request_id", None),
                    ),
                )
            print(
                f"Embedded batch {index}: {len(batch)} chunks, "
                f"{response.usage.prompt_tokens} tokens",
                flush=True,
            )
            completed_batches += 1
            prompt_tokens += response.usage.prompt_tokens
        # Publication is separate: validate this ready snapshot before activating it.
        await mark_ready(conn, corpus_id)
        return await status(conn, corpus_id) | {
            "new_embedding_batches": completed_batches,
            "new_embedding_inputs": len(rows),
            "new_prompt_tokens": prompt_tokens,
        }
    finally:
        await conn.execute("SELECT pg_advisory_unlock(728420)")
