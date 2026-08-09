"""
WeatherLens AI — embedding job.

Runs as a Databricks notebook or a plain CLI script:
    python notebooks/ingest_weather_embeddings.py

Responsibilities:
  1. Ensure schema exists (idempotent).
  2. Find documents that have no embeddings OR whose text has changed since
     the last embedding run (anti-join keyed on document_id + content_hash
     + model_name).
  3. Chunk each document's narrative, embed the chunks in batches, then
     DELETE old embeddings and INSERT the new ones in one transaction per
     document.  DELETE-then-INSERT avoids the surplus-chunk problem that an
     UPSERT on (document_id, chunk_index) leaves behind when an amended alert
     produces fewer chunks than the original.
"""

import os
import sys

# Make project root importable when run as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from psycopg2.extras import execute_values

from embeddings import BATCH_SIZE, MODEL_NAME, Encoder, chunk_text
from lakebase import apply_schema, get_lakebase_connection

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Anti-join: documents with no current embeddings for this model+hash pair.
# "Current" means (document_id, content_hash, model_name) all match.
# A stale row (old hash) does not satisfy the join condition, so its document
# appears in the result and gets re-embedded.
_FIND_SQL = """
    SELECT d.id, d.narrative_text, d.content_hash
    FROM weather_documents d
    LEFT JOIN weather_embeddings e
        ON  e.document_id  = d.id
        AND e.content_hash = d.content_hash
        AND e.model_name   = %s
    WHERE e.id IS NULL
"""

_DELETE_SQL = "DELETE FROM weather_embeddings WHERE document_id = %s"

_INSERT_SQL = """
    INSERT INTO weather_embeddings
        (id, document_id, chunk_index, chunk_text, embedding, model_name, content_hash)
    VALUES %s
"""

# %s::vector on the embedding column writes a real VECTOR(384) on the first
# insert.  Omitting the cast lets psycopg2 send the list as a text array,
# which pgvector rejects (or accepts as text, depending on the version) and
# leaves the column in a state where <=> returns wrong results silently.
_ROW_TEMPLATE = "(%s, %s, %s, %s, %s::vector, %s, %s)"


# ---------------------------------------------------------------------------
# Helpers (factored out so they can be unit-tested)
# ---------------------------------------------------------------------------

def find_unembedded(conn, model_name: str) -> list[tuple]:
    """Return (id, narrative_text, content_hash) rows needing embedding."""
    with conn.cursor() as cur:
        cur.execute(_FIND_SQL, (model_name,))
        return cur.fetchall()


def write_document(conn, encoder: Encoder, doc_id: str, text: str, content_hash: str) -> int:
    """Delete stale embeddings and insert fresh ones for one document.

    Returns the number of chunks written.  Runs inside its own transaction
    (the caller's get_lakebase_connection context manager handles commit).
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Embed in batches to avoid OOM on very long documents.
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        vectors.extend(encoder.encode(chunks[i : i + BATCH_SIZE]))

    rows = [
        (
            f"{doc_id}::{idx}",   # id: composite string PK
            doc_id,               # document_id (FK)
            idx,                  # chunk_index
            chunk,                # chunk_text
            vector,               # embedding (cast to VECTOR by template)
            encoder.model_name,   # model_name
            content_hash,         # mirrors weather_documents.content_hash
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    with conn.cursor() as cur:
        cur.execute(_DELETE_SQL, (doc_id,))
        execute_values(cur, _INSERT_SQL, rows, template=_ROW_TEMPLATE)

    return len(chunks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(encoder: Encoder) -> tuple[int, int]:
    """Run the full embedding job. Returns (n_docs_embedded, n_chunks_written)."""
    apply_schema()

    # Fetch the work list once before opening per-document transactions.
    with get_lakebase_connection() as conn:
        docs = find_unembedded(conn, encoder.model_name)

    total = len(docs)
    print(f"Documents to embed: {total}")
    if not total:
        print("Nothing to do.")
        return 0, 0

    embedded = 0
    total_chunks = 0

    for i, (doc_id, narrative_text, content_hash) in enumerate(docs, 1):
        # Each document gets its own transaction so a failure on document N
        # does not roll back the work already done on documents 1..N-1.
        with get_lakebase_connection() as conn:
            n = write_document(conn, encoder, doc_id, narrative_text, content_hash)

        embedded += 1
        total_chunks += n
        short_id = doc_id[:60] + ("…" if len(doc_id) > 60 else "")
        print(f"  [{i}/{total}] {short_id} — {n} chunk(s)")

    print(f"\nDone. {embedded} document(s) embedded, {total_chunks} chunk(s) written.")
    return embedded, total_chunks


if __name__ == "__main__":
    enc = Encoder(MODEL_NAME)
    run(enc)
