CREATE SCHEMA IF NOT EXISTS pipeline;
CREATE TABLE IF NOT EXISTS pipeline.jobs (
    id uuid PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    request_sha256 text NOT NULL,
    request jsonb NOT NULL,
    settings jsonb NOT NULL,
    state text NOT NULL DEFAULT 'queued' CHECK (state IN
      ('queued','retrieving','directing','generating_keyframes',
       'keyframes_completed','failed','interrupted')),
    context jsonb,
    direction jsonb,
    frames jsonb NOT NULL DEFAULT '[]',
    error jsonb,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipeline_jobs_queue ON pipeline.jobs(created_at) WHERE state='queued';
