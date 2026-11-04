-- Runs once on first container start (empty volume).
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per indexed chunk. embedding dim is set at runtime by the app
-- (ALTER / recreate) if it does not match GATEWAY embedding size.
-- Default 1024 matches common Titan V2 / many gateway defaults; the Python
-- layer will ensure the column exists with the correct dimensions.

CREATE TABLE IF NOT EXISTS repo_collections (
    name            TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS repo_chunks (
    id              TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL REFERENCES repo_collections(name) ON DELETE CASCADE,
    repo_id         TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL DEFAULT '',
    document        TEXT NOT NULL DEFAULT '',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Placeholder vector; real dimension applied when collection is created.
    embedding       vector(1024),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repo_chunks_collection
    ON repo_chunks (collection_name);

CREATE INDEX IF NOT EXISTS idx_repo_chunks_repo
    ON repo_chunks (collection_name, repo_id);

CREATE INDEX IF NOT EXISTS idx_repo_chunks_metadata
    ON repo_chunks USING GIN (metadata);
