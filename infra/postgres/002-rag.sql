-- Also applied idempotently by the RAG CLI for existing Docker volumes.
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.corpora (
    id text PRIMARY KEY,
    model text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = 3072),
    state text NOT NULL CHECK (state IN ('prepared', 'embedding', 'ready')),
    active boolean NOT NULL DEFAULT false,
    summary jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (NOT active OR state = 'ready')
);
CREATE UNIQUE INDEX IF NOT EXISTS rag_one_active_corpus ON rag.corpora (active) WHERE active;
CREATE TABLE IF NOT EXISTS rag.documents (
    id text PRIMARY KEY,
    path text NOT NULL,
    title text NOT NULL,
    file_sha256 text NOT NULL,
    source_url text,
    page_count integer NOT NULL CHECK (page_count > 0)
);
CREATE TABLE IF NOT EXISTS rag.pages (
    document_id text REFERENCES rag.documents(id),
    page_number integer NOT NULL CHECK (page_number > 0),
    raw_text text NOT NULL,
    clean_text text NOT NULL,
    quality_flags jsonb NOT NULL,
    PRIMARY KEY (document_id, page_number)
);
CREATE TABLE IF NOT EXISTS rag.corpus_documents (
    corpus_id text REFERENCES rag.corpora(id),
    document_id text REFERENCES rag.documents(id),
    selection text NOT NULL,
    PRIMARY KEY (corpus_id, document_id)
);
CREATE TABLE IF NOT EXISTS rag.chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL,
    page_number integer NOT NULL,
    ordinal integer NOT NULL,
    content text NOT NULL CHECK (length(content) > 0),
    input_text text NOT NULL,
    input_sha256 text NOT NULL,
    token_count integer NOT NULL CHECK (token_count BETWEEN 1 AND 8191),
    FOREIGN KEY (document_id, page_number) REFERENCES rag.pages(document_id, page_number)
);
CREATE INDEX IF NOT EXISTS rag_chunks_input_hash ON rag.chunks(input_sha256);
CREATE TABLE IF NOT EXISTS rag.corpus_chunks (
    corpus_id text REFERENCES rag.corpora(id),
    chunk_id text REFERENCES rag.chunks(id),
    PRIMARY KEY (corpus_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS rag.embeddings (
    input_sha256 text NOT NULL,
    model text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = 3072),
    embedding vector(3072) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (input_sha256, model, dimensions)
);
-- Exact cosine search retains all 3072 float32 dimensions for this small corpus.
-- Standard pgvector vector HNSW supports at most 2000 dimensions; no lossy index is used.
CREATE TABLE IF NOT EXISTS rag.embedding_usage (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    corpus_id text REFERENCES rag.corpora(id),
    purpose text NOT NULL,
    model text NOT NULL,
    input_count integer NOT NULL,
    prompt_tokens integer NOT NULL,
    request_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);
