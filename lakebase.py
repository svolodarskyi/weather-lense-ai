"""
WeatherLens AI — Lakebase (pgvector) connection and schema management.

Two entry points:
  get_lakebase_connection() — context manager yielding a psycopg2 connection
  apply_schema()            — idempotent DDL: extension + tables + HNSW index

Connection resolution order:
  1. LAKEBASE_CONNECTION_URL  (full libpq URL; whitespace stripped to catch copy-paste artifacts)
  2. PGHOST + PGDATABASE + PGUSER + PGPASSWORD + PGPORT  (Databricks App auto-injection)
"""

import contextlib
import os
import re

import psycopg2


class LakebaseError(Exception):
    """Raised when a Lakebase connection cannot be configured."""


def _build_dsn() -> str:
    url = os.getenv("LAKEBASE_CONNECTION_URL", "")
    if url.strip():
        # Copy-pasted URLs often carry leading/trailing whitespace or embedded
        # newlines. psycopg2 passes the string to libpq directly; libpq treats
        # an unrecognised host character as a DNS name, producing a confusing
        # "could not translate host name" error rather than "bad URL".
        return re.sub(r"\s+", "", url)

    host = os.getenv("PGHOST", "").strip()
    if not host:
        raise LakebaseError(
            "No Lakebase connection configured. "
            "Set LAKEBASE_CONNECTION_URL (full URL) or "
            "PGHOST / PGDATABASE / PGUSER / PGPASSWORD / PGPORT."
        )

    database = os.getenv("PGDATABASE", "databricks_postgres")
    user     = os.getenv("PGUSER", "")
    password = os.getenv("PGPASSWORD", "")
    port     = os.getenv("PGPORT", "5432")

    return (
        f"host={host} dbname={database} user={user} "
        f"password={password} port={port} sslmode=require"
    )


@contextlib.contextmanager
def get_lakebase_connection():
    """Yield a psycopg2 connection; commit on clean exit, roll back on error.

    Usage::

        with get_lakebase_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    dsn  = _build_dsn()
    conn = psycopg2.connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema DDL — executed in order by apply_schema()
# ---------------------------------------------------------------------------

_DDL = [
    # 1. Extension must exist before any VECTOR column can be created.
    "CREATE EXTENSION IF NOT EXISTS vector",

    # 2. Document store (raw NWS text).
    """CREATE TABLE IF NOT EXISTS weather_documents (
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
)""",

    # 3. Vector store.  CASCADE ensures purging a document removes its vectors.
    """CREATE TABLE IF NOT EXISTS weather_embeddings (
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
)""",

    # 4. HNSW index for cosine similarity search.
    #    The ORDER BY clause MUST use the bare <=> operator — the expression
    #    form "1 - (... <=>  ...) DESC" cannot be matched by this index.
    """CREATE INDEX IF NOT EXISTS weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops)""",
]


def apply_schema() -> None:
    """Create the pgvector extension, both tables, and the HNSW index.

    All statements use IF NOT EXISTS so this is safe to call on every startup.
    The first statement (CREATE EXTENSION) will raise immediately if the role
    lacks CREATE EXTENSION permission, which surfaces the problem before any
    vector insert attempt.
    """
    with get_lakebase_connection() as conn:
        with conn.cursor() as cur:
            for stmt in _DDL:
                cur.execute(stmt)
