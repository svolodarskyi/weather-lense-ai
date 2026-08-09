# Databricks notebook source

# MAGIC %pip install -q sentence-transformers

# COMMAND ----------

# Restart the interpreter so sentence-transformers is importable.
# psycopg2 is NOT installed here — Databricks Runtime ships psycopg2 2.9.11
# already. Adding psycopg2-binary on top of it creates two competing native
# builds in the same interpreter, which aborts with SIGABRT on import.
dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Resolve project root from this notebook's workspace path so the import of
# lakebase / repository / etc. works regardless of where the bundle is deployed.

sys.path.append("../")

# COMMAND ----------

dbutils.widgets.text(
    "locations",
    "Chicago, IL;Miami, FL;New York, NY;Los Angeles, CA;Houston, TX",
    "Locations (semicolon-separated)",
)
dbutils.widgets.text("limit", "50", "Max docs per location")
dbutils.widgets.text("secret_scope", "database", "Databricks secret scope")
dbutils.widgets.text("secret_key", "lakebase-url", "Secret key for connection URL")

LOCATIONS = [
    l.strip() for l in dbutils.widgets.get("locations").split(";") if l.strip()
]
LIMIT = int(dbutils.widgets.get("limit"))
SECRET_SCOPE = dbutils.widgets.get("secret_scope")
SECRET_KEY = dbutils.widgets.get("secret_key")

os.environ["LAKEBASE_CONNECTION_URL"] = dbutils.secrets.get(
    scope=SECRET_SCOPE, key=SECRET_KEY
)

# COMMAND ----------

from lakebase import get_lakebase_connection
from weather_client import WeatherClient
from embeddings import Encoder, MODEL_NAME, chunk_text
import repository
from psycopg2.extras import execute_values

# COMMAND ----------

print(f"Syncing {len(LOCATIONS)} location(s), limit={LIMIT}.")

# COMMAND ----------

# ── Step 1: sync NWS documents ──────────────────────────────────────────────

client = WeatherClient(user_agent="WeatherLensAI/1.0 sergii.volodarskyi@gmail.com")
docs = client.fetch(LOCATIONS, limit=LIMIT)

with get_lakebase_connection() as conn:
    n_synced = repository.upsert_documents(conn, docs)
    n_cleared = repository.clear_stale_embeddings(conn)

print(f"Synced {n_synced} docs.  Cleared {n_cleared} stale embedding rows.")
if client.errors:
    print(f"NWS fetch errors: {client.errors}")

# COMMAND ----------

# ── Step 2: embed documents that have no vectors yet ────────────────────────

_FIND_SQL = """
    SELECT d.id, d.narrative_text, d.content_hash
    FROM   weather_documents d
    LEFT   JOIN weather_embeddings e ON e.document_id = d.id
    WHERE  e.document_id IS NULL
"""

_INSERT_SQL = """
    INSERT INTO weather_embeddings
        (id, document_id, chunk_index, chunk_text, embedding, model_name, content_hash)
    VALUES %s
    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
        chunk_text   = EXCLUDED.chunk_text,
        embedding    = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash
"""

encoder = Encoder(MODEL_NAME)

with get_lakebase_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(_FIND_SQL)
        to_embed = cur.fetchall()  # [(id, narrative_text, content_hash), ...]

    rows = []
    for doc_id, narrative, content_hash in to_embed:
        chunks = chunk_text(narrative)
        if not chunks:
            continue
        vecs = encoder.encode(chunks)
        for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
            rows.append(
                (
                    f"{doc_id}_{idx}",
                    doc_id,
                    idx,
                    chunk,
                    str(vec),
                    MODEL_NAME,
                    content_hash,
                )
            )

    if rows:
        with conn.cursor() as cur:
            execute_values(cur, _INSERT_SQL, rows)

print(f"Embedded {len(to_embed)} docs → {len(rows)} chunk rows written.")

# COMMAND ----------

summary = {
    "synced": n_synced,
    "cleared_stale": n_cleared,
    "docs_embedded": len(to_embed),
    "chunks_written": len(rows),
    "errors": client.errors,
}
print(f"Run complete: {summary}")
dbutils.notebook.exit(str(summary))
