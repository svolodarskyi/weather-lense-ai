"""
WeatherLens AI — database operations.

Sprint 3: write side (upsert_documents, clear_stale_embeddings).
Sprint 5: read side (search).
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


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

# Both queries repeat the embedding expression in ORDER BY rather than
# referencing the SELECT alias `distance`. PostgreSQL can only use the HNSW
# index when the ORDER BY expression matches the indexed expression exactly;
# an alias reference or 1-(...) DESC defeats the index and forces a seq scan.
_SEARCH_SQL = """
    SELECT d.id, d.location, d.event, d.source_type, e.chunk_text,
           e.document_id,
           e.embedding <=> %s::vector AS distance
    FROM weather_embeddings e
    JOIN weather_documents d ON d.id = e.document_id
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s
"""

_SEARCH_FILTERED_SQL = """
    SELECT d.id, d.location, d.event, d.source_type, e.chunk_text,
           e.document_id,
           e.embedding <=> %s::vector AS distance
    FROM weather_embeddings e
    JOIN weather_documents d ON d.id = e.document_id
    WHERE d.source_type = %s
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s
"""


def search(
    conn,
    embedding: list[float],
    top_k: int,
    source_type: str | None = None,
) -> list[dict]:
    """Semantic similarity search over weather_embeddings.

    Fetches top_k * 3 rows from the DB (already ordered by distance), then
    deduplicates in Python to one result per document_id — the DB-ordered rows
    guarantee the first occurrence of each document_id has the lowest distance.
    Returns at most top_k results with similarity = 1 - cosine_distance.
    """
    vec = str(embedding)   # "[0.1, 0.2, ...]" matches pgvector text input format
    fetch = top_k * 3

    with conn.cursor() as cur:
        if source_type:
            cur.execute(_SEARCH_FILTERED_SQL, (vec, source_type, vec, fetch))
        else:
            cur.execute(_SEARCH_SQL, (vec, vec, fetch))
        rows = cur.fetchall()

    seen: set[str] = set()
    results: list[dict] = []
    for _, location, event, src_type, chunk_text, doc_id, distance in rows:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        results.append({
            "location": location,
            "event": event,
            "source_type": src_type,
            "chunk_text": chunk_text,
            "similarity": round(1.0 - float(distance), 6),
        })
        if len(results) >= top_k:
            break

    return results


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
