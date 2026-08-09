# WeatherLens AI

A weather intelligence pipeline that harvests National Weather Service (NWS) text, embeds it with a sentence transformer, stores vectors in Databricks Lakebase (pgvector), and serves semantic search via a Flask REST API deployed as a Databricks App.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Data sources](#data-sources)
3. [Schema decisions](#schema-decisions)
4. [End-to-end walkthrough](#end-to-end-walkthrough)
5. [API reference](#api-reference)
6. [Local development](#local-development)
7. [Deployment](#deployment)

---

## Architecture overview

```
NWS API (api.weather.gov)
        │
        ▼
  weather_client.py          ← harvest alerts + forecasts, geocode cities
        │
        ▼
  weather_documents           ← raw NWS text in Lakebase (Postgres)
        │
        ▼
  ingest_weather_embeddings   ← chunk → embed → write VECTOR(384) rows
        │
        ▼
  weather_embeddings          ← pgvector table with HNSW index
        │
        ▼
  Flask REST API (app.py)     ← /weather/sync  /weather/search  /health
```

Two write paths, one read path:

- **Sync** (`POST /weather/sync`): the API fetches NWS documents for a list of locations and upserts them into `weather_documents`. Content hashes detect amendments so unchanged text is never re-embedded.
- **Embed** (`notebooks/ingest_weather_embeddings.py`): a separate job (run as a Databricks notebook or a cron script) finds documents without current embeddings, chunks the text, and writes `VECTOR(384)` rows into `weather_embeddings`.
- **Search** (`POST /weather/search`): the API encodes a free-text query with the same model, runs an HNSW cosine-similarity search, deduplicates to one result per source document, and returns ranked results.

---

## Data sources

### National Weather Service (api.weather.gov)

NWS is the primary data source for two reasons:

1. **No API key required.** Any client with a `User-Agent` header can query the public API, which means zero provisioning friction for a bootcamp project and no rate-limit billing.
2. **Structured + narrative text.** Each alert and forecast carries both machine-readable metadata (event type, severity, effective/expiry timestamps) and a human-readable `description` field. The narrative text is what gets embedded; the metadata drives filtering and display.

Two NWS endpoints are used:

| Endpoint                                    | What it returns                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `GET /alerts/active?area={state}`           | Active weather alerts (Flood Watch, Tornado Warning, Heat Advisory, …) for a US state |
| `GET /gridpoints/{office}/{x},{y}/forecast` | 7-day period-by-period forecast for a lat/lon grid cell                               |

### Open-Meteo geocoding

NWS forecasts require a grid office and grid coordinates, not a plain city name. Open-Meteo's free geocoding API (`geocoding-api.open-meteo.com`) resolves a city name to lat/lon, which is then fed to NWS's `/points/{lat},{lon}` endpoint to get the correct grid reference. Open-Meteo was chosen because it requires no API key and covers global cities.

### Amendment detection

NWS re-issues alerts under their original ID when conditions change (e.g. a Flood Watch is upgraded to a Flood Warning). The pipeline handles this with a `content_hash = SHA-256(narrative_text)` column on both tables:

- On sync: the upsert writes the new hash; any mismatch between `weather_documents.content_hash` and `weather_embeddings.content_hash` is detected by `clear_stale_embeddings()` and those embedding rows are deleted.
- On the next embedding run: the anti-join query finds the document (its embeddings are gone) and re-embeds it.

---

## Schema decisions

### Two-table design

```sql
weather_documents   -- one row per NWS alert or forecast period
weather_embeddings  -- one row per text chunk (many per document)
```

A single denormalised table (document columns repeated on every chunk row) was considered and rejected because:

- NWS metadata (location, event, severity, payload) is queried and filtered without touching embeddings.
- The HNSW index sits entirely on `weather_embeddings`; keeping that table narrow (chunk text + vector + FK) keeps index build time and memory low.
- `ON DELETE CASCADE` on the FK lets `DELETE FROM weather_documents` cleanly cascade to all its chunks without a manual join.

### `content_hash` on both tables

The hash is stored on `weather_documents` so the sync path can detect amendments, and mirrored onto `weather_embeddings` so `clear_stale_embeddings()` can delete stale chunks with a single `DELETE … USING` join rather than a correlated subquery:

```sql
DELETE FROM weather_embeddings we
USING weather_documents wd
WHERE we.document_id = wd.id
  AND we.content_hash != wd.content_hash
```

### HNSW index with cosine ops

```sql
CREATE INDEX weather_embeddings_hnsw
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);
```

HNSW (Hierarchical Navigable Small World) was chosen over IVFFlat because it does not require a training phase (`CREATE INDEX` on IVFFlat must scan all existing rows to build cluster centroids; HNSW inserts incrementally). For a pipeline that adds rows continuously, HNSW provides consistent index quality without a periodic rebuild step.

Cosine similarity (`vector_cosine_ops`) rather than L2 distance (`vector_l2_ops`) is used because the embeddings are L2-normalised by the sentence transformer. For unit vectors, cosine distance and L2 distance produce the same ranking, but the `<=>` cosine operator is the idiomatic choice for text embeddings and makes the similarity arithmetic (`1 - distance`) self-documenting.

**Critical**: the `ORDER BY` clause in all search queries repeats the bare `<=>` expression rather than referencing the `distance` alias:

```sql
ORDER BY e.embedding <=> %s::vector   -- ✅ HNSW index used
-- ORDER BY distance                  -- ❌ alias ref, seq scan
-- ORDER BY 1 - (...) DESC            -- ❌ expression mismatch, seq scan
```

### Chunking strategy (800 chars, 100-char overlap)

NWS alert text ranges from a single sentence to multi-paragraph advisories. A fixed 800-character window keeps each chunk under the model's 256-token context window while carrying enough context for a meaningful embedding. The 100-character overlap prevents information loss at boundaries.

Boundary preference (high to low):

1. **Paragraph break (`\n\n`)** — clean semantic boundary; no overlap applied across it.
2. **Sentence end (`.!?` + whitespace)** — most natural split; overlap applied.
3. **Word boundary (any whitespace)** — fallback to avoid mid-token splits.
4. **Hard break** — only when the window contains no whitespace (rare in NWS text).

### Embedding model: `all-MiniLM-L6-v2`

384 dimensions, ~80 MB weights, runs on CPU in reasonable time (~5 ms/chunk). It produces good semantic similarity for short English paragraphs, which is exactly the NWS narrative style. Larger models (e.g. `all-mpnet-base-v2`, 768 dims) showed marginal relevance gains in informal testing but double the storage and index memory.

---

## End-to-end walkthrough

### Step 1 — Apply the schema

On first run (or after a database reset), create the extension, tables, and index:

```bash
python - <<'EOF'
from lakebase import apply_schema
apply_schema()
print("Schema ready.")
EOF
```

`apply_schema()` is idempotent (`CREATE … IF NOT EXISTS` throughout), so it is also safe to call on every app startup.

### Step 2 — Sync NWS data

```bash
curl -X POST http://localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Miami, FL", "Denver, CO"]}'
```

```json
{ "synced": 47, "errors": [] }
```

`synced` is the number of documents upserted (new + amended). `errors` lists any locations that failed geocoding or returned no NWS data.

### Step 3 — Embed documents

Run the embedding job against unembedded documents:

```bash
python notebooks/ingest_weather_embeddings.py
```

```
Documents to embed: 47
  [1/47] urn:oid:2.49.0.1.840.0.KOAX.202608... — 3 chunk(s)
  [2/47] urn:oid:2.49.0.1.840.0.KOAX.202608... — 1 chunk(s)
  ...
Done. 47 document(s) embedded, 112 chunk(s) written.
```

Each document gets its own transaction so a failure on document N does not roll back work already done on documents 1…N−1.

### Step 4 — Search

```bash
curl -X POST http://localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

```json
{
  "query": "flash flood risk this weekend",
  "count": 5,
  "results": [
    {
      "location": "Miami, FL",
      "event": "Friday",
      "source_type": "forecast",
      "chunk_text": "A slight chance of showers and thunderstorms after 2pm...",
      "similarity": 0.428577
    },
    ...
  ]
}
```

Filter by source type to narrow results to active alerts only:

```bash
curl -X POST http://localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "coastal storm surge", "top_k": 3, "source_type": "alert"}'
```

---

## API reference

### `GET /health`

Liveness probe. Returns immediately without touching the database or the embedding model.

```json
{ "status": "ok" }
```

### `POST /weather/sync`

Harvest NWS alerts and forecasts for a list of locations and upsert them into `weather_documents`.

**Request body**

| Field       | Type       | Default  | Description                        |
| ----------- | ---------- | -------- | ---------------------------------- |
| `locations` | `string[]` | required | City names, e.g. `["Chicago, IL"]` |
| `limit`     | `int`      | `50`     | Max documents to fetch per run     |

**Response**

| Field    | Type       | Description                                    |
| -------- | ---------- | ---------------------------------------------- |
| `synced` | `int`      | Documents upserted (new + amended)             |
| `errors` | `string[]` | Locations that failed (geocoding or NWS error) |

### `POST /weather/search`

Semantic similarity search over stored embeddings.

**Request body**

| Field         | Type     | Default  | Description                               |
| ------------- | -------- | -------- | ----------------------------------------- |
| `query`       | `string` | required | Free-text query                           |
| `top_k`       | `int`    | `5`      | Results to return (clamped to 1–20)       |
| `source_type` | `string` | `null`   | `"alert"`, `"forecast"`, or omit for both |

**Response**

| Field     | Type       | Description                |
| --------- | ---------- | -------------------------- |
| `query`   | `string`   | Stripped query text        |
| `count`   | `int`      | Number of results returned |
| `results` | `object[]` | Ranked results (see below) |

Each result object:

| Field         | Type     | Description                                        |
| ------------- | -------- | -------------------------------------------------- |
| `location`    | `string` | City / NWS zone name                               |
| `event`       | `string` | Alert event type or forecast period                |
| `source_type` | `string` | `"alert"` or `"forecast"`                          |
| `chunk_text`  | `string` | The matching text chunk                            |
| `similarity`  | `float`  | Cosine similarity to query (0–1, higher is closer) |

---

## Local development

### Prerequisites

- Python 3.11+
- A Databricks workspace with a Lakebase project (pgvector extension available)
- Databricks CLI v0.294.0+

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure the connection

**Option A — full URL (local dev):**

```bash
export LAKEBASE_CONNECTION_URL="postgresql://user:token@host/databricks_postgres?sslmode=require"
```

**Option B — individual vars (matches Databricks App injection):**

```bash
export PGHOST="ep-xxx.database.us-east-2.cloud.databricks.com"
export PGDATABASE="databricks_postgres"
export PGUSER="you@example.com"
export PGPASSWORD="<oauth-token>"
export PGPORT="5432"
```

Generate a short-lived token with the Databricks CLI:

```bash
EP="projects/<project-id>/branches/production/endpoints/primary"
export PGPASSWORD=$(databricks postgres generate-database-credential $EP --profile DEFAULT -o json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
```

### Run the Flask dev server

```bash
NWS_USER_AGENT="(YourApp yourname@example.com)" flask run --port 8000
```

### Run the tests

```bash
pytest tests/ -v
```

All external boundaries (HTTP, Postgres wire, sentence-transformer encoder) are mocked. No real database or NWS connection is needed to run the test suite.

---

## Deployment

The app is deployed as a Databricks App with Lakebase wired as a resource (auto-injects `PG*` environment variables).

```bash
# From the project root
databricks apps deploy weather-lens-ai --profile DEFAULT
```

The app's service principal must be deployed before running locally so it can create and own the schema. See the Lakebase skill docs on schema ownership if you hit `permission denied for schema`.

The embedding job (`notebooks/ingest_weather_embeddings.py`) is intended to run as a scheduled Databricks notebook, triggered after each sync or on a fixed cadence (e.g. every 15 minutes).
