"""
WeatherLens AI — database operations.

Sprint 3: write side (upsert_documents, clear_stale_embeddings).
Sprint 5: read side (search) added later.
"""

from psycopg2.extras import execute_values, Json
from weather_client import WeatherDoc

# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
    INSERT INTO weather_documents (
        id, location, latitude, longitude, source_type,
        event, headline, narrative_text, issued_at, effective_at,
        expires_at, severity, payload, content_hash
    ) VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        narrative_text = EXCLUDED.narrative_text,
        content_hash   = EXCLUDED.content_hash,
        headline       = EXCLUDED.headline,
        issued_at      = EXCLUDED.issued_at,
        effective_at   = EXCLUDED.effective_at,
        expires_at     = EXCLUDED.expires_at,
        synced_at      = NOW()
"""

# Deletes embeddings whose content_hash no longer matches the parent document.
# USING ... WHERE is a Postgres extension that lets us join in a DELETE without
# a subquery. The result is the same as a NOT IN subquery but the planner can
# use a hash join, which matters when weather_embeddings grows large.
_CLEAR_STALE_SQL = """
    DELETE FROM weather_embeddings we
    USING weather_documents wd
    WHERE we.document_id = wd.id
      AND we.content_hash != wd.content_hash
"""


def _to_row(doc: WeatherDoc) -> tuple:
    row = doc.as_row()
    return (
        row["id"], row["location"], row["latitude"], row["longitude"],
        row["source_type"], row["event"], row["headline"], row["narrative_text"],
        row["issued_at"], row["effective_at"], row["expires_at"],
        row["severity"], Json(row["payload"]), row["content_hash"],
    )


def upsert_documents(conn, docs: list[WeatherDoc]) -> int:
    """Batch-upsert WeatherDocs into weather_documents.

    Uses a single execute_values call so one round-trip handles an entire
    harvest batch. Returns the number of documents submitted (new + updated).
    """
    if not docs:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, _UPSERT_SQL, [_to_row(d) for d in docs])
    return len(docs)


def clear_stale_embeddings(conn) -> int:
    """Remove embeddings invalidated by an in-place NWS amendment.

    NWS re-issues alerts under their original id with changed text, so after
    upsert the document's content_hash may differ from the hash stored on its
    embeddings. Deleting those rows signals the embedding job to re-embed them
    on its next run. Returns the number of embedding rows removed.
    """
    with conn.cursor() as cur:
        cur.execute(_CLEAR_STALE_SQL)
        return cur.rowcount
