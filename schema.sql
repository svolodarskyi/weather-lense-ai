-- WeatherLens AI — Lakebase schema
-- Run order matters: extension before tables, tables before index.
-- All statements are idempotent (IF NOT EXISTS).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents (
    id               TEXT             PRIMARY KEY,
    location         TEXT             NOT NULL,
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    source_type      TEXT             NOT NULL
                         CHECK (source_type IN ('alert', 'forecast')),
    event            TEXT,
    headline         TEXT,
    narrative_text   TEXT             NOT NULL,
    issued_at        TIMESTAMPTZ,
    effective_at     TIMESTAMPTZ,
    expires_at       TIMESTAMPTZ,
    severity         TEXT,
    payload          JSONB            NOT NULL,
    content_hash     TEXT             NOT NULL,
    synced_at        TIMESTAMPTZ      DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id              TEXT        PRIMARY KEY,
    document_id     TEXT        NOT NULL
                        REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    chunk_text      TEXT        NOT NULL,
    embedding       VECTOR(384) NOT NULL,
    model_name      TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

-- ORDER BY must use the bare <=> operator, not the expression form
-- "1 - (embedding <=> query) DESC" — only the bare form uses this index.
CREATE INDEX IF NOT EXISTS weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
