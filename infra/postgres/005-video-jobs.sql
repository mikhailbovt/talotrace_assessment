CREATE TABLE IF NOT EXISTS pipeline.video_jobs (
    id uuid PRIMARY KEY,
    keyframe_job_id uuid NOT NULL REFERENCES pipeline.jobs(id),
    idempotency_key text NOT NULL UNIQUE,
    request_sha256 text NOT NULL,
    request jsonb NOT NULL,
    settings jsonb NOT NULL,
    state text NOT NULL DEFAULT 'waiting_for_keyframes' CHECK (state IN
      ('waiting_for_keyframes','queued','synthesizing','rendering','assembling',
       'video_completed','failed','interrupted')),
    timeline jsonb,
    progress jsonb NOT NULL DEFAULT '{}',
    result jsonb,
    error jsonb,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipeline_video_jobs_queue ON pipeline.video_jobs(created_at)
    WHERE state IN ('waiting_for_keyframes','queued');
