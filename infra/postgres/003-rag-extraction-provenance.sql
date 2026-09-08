-- Immutable document IDs include extraction identity from the pdf-inspector rebuild onward.
-- Existing pypdf document/page records remain unchanged and addressable by older snapshots.
ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS extraction_metadata jsonb
    NOT NULL DEFAULT '{"primary_extractor":"pypdf","primary_version":"legacy-not-recorded"}';
ALTER TABLE rag.pages ADD COLUMN IF NOT EXISTS extraction_metadata jsonb
    NOT NULL DEFAULT '{"extractor":"pypdf","extractor_version":"legacy-not-recorded"}';
