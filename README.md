# WeatherLens AI

A weather intelligence pipeline that harvests National Weather Service (NWS) text, embeds it with a sentence transformer, stores vectors in Databricks Lakebase (pgvector), and serves semantic search and AI-powered chat via a Flask REST API.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Data sources](#data-sources)
3. [Schema decisions](#schema-decisions)
4. [End-to-end walkthrough](#end-to-end-walkthrough)
5. [API reference](#api-reference)
6. [Local development](#local-development)
7. [Deployment](#deployment)
8. [CI/CD](#cicd)
9. [Deliverables](#deliverables)

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
  refresh_weather_index job   ← chunk → embed → write VECTOR(384) rows
  (scheduled Databricks Job,  ← runs every 30 min via Databricks Asset Bundles)
  notebooks/refresh_weather_index.py
        │
        ▼
  weather_embeddings          ← pgvector table with HNSW index
        │
        ▼
  Flask REST API (app.py)     ← /weather/sync  /weather/search  /weather/chat  /health
        │
        ▼
  Llama 3.3 70B (Databricks Foundation Models)  ← RAG answers via /weather/chat
```

Two write paths, one read path:

- **Sync** (`POST /weather/sync`): the API fetches NWS documents for a list of locations and upserts them into `weather_documents`. Content hashes detect amendments so unchanged text is never re-embedded.
- **Scheduled job** (`notebooks/refresh_weather_index.py`): a Databricks Job (deployed via Asset Bundles, scheduled every 30 min) finds documents without current embeddings, chunks the text, and writes `VECTOR(384)` rows into `weather_embeddings`. It also re-syncs NWS data before embedding, so the job is fully self-contained.
- **Search** (`POST /weather/search`): the API encodes a free-text query with the same model, runs an HNSW cosine-similarity search, and returns ranked results.
- **Chat** (`POST /weather/chat`): retrieves the top-k relevant chunks (same as search), then synthesises a plain-language answer with Llama 3.3 70B via Databricks Foundation Models.

---

## Data sources

### National Weather Service (api.weather.gov)

NWS is the primary data source for two reasons:

1. **No API key required.** Any client with a `User-Agent` header can query the public API, which means zero provisioning friction and no rate-limit billing.
2. **Structured + narrative text.** Each alert and forecast carries both machine-readable metadata (event type, severity, effective/expiry timestamps) and a human-readable `description` field. The narrative text is what gets embedded; the metadata drives filtering and display.

Two NWS endpoints are used:

| Endpoint                                    | What it returns                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `GET /alerts/active?area={state}`           | Active weather alerts (Flood Watch, Tornado Warning, Heat Advisory, …) for a US state |
| `GET /gridpoints/{office}/{x},{y}/forecast` | 7-day period-by-period forecast for a lat/lon grid cell                               |

### Open-Meteo geocoding

NWS forecasts require a grid office and grid coordinates, not a plain city name. Open-Meteo's free geocoding API (`geocoding-api.open-meteo.com`) resolves a city name to lat/lon, which is then fed to NWS's `/points/{lat},{lon}` endpoint to get the correct grid reference.

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

A single denormalised table was considered and rejected because:

- NWS metadata (location, event, severity, payload) is queried and filtered without touching embeddings.
- The HNSW index sits entirely on `weather_embeddings`; keeping that table narrow (chunk text + vector + FK) keeps index build time and memory low.
- `ON DELETE CASCADE` on the FK lets `DELETE FROM weather_documents` cleanly cascade to all its chunks without a manual join.

### `content_hash` on both tables

The hash is stored on `weather_documents` so the sync path can detect amendments, and mirrored onto `weather_embeddings` so `clear_stale_embeddings()` can delete stale chunks with a single `DELETE … USING` join:

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

HNSW was chosen over IVFFlat because it does not require a training phase (`CREATE INDEX` on IVFFlat must scan all existing rows to build cluster centroids; HNSW inserts incrementally). For a pipeline that adds rows continuously, HNSW provides consistent index quality without a periodic rebuild step.

Cosine similarity (`vector_cosine_ops`) is used because the embeddings are L2-normalised by the sentence transformer. For unit vectors, cosine and L2 produce the same ranking; the `<=>` operator is the idiomatic choice for text embeddings and makes the similarity arithmetic (`1 - distance`) self-documenting.

**Critical**: the `ORDER BY` clause in all search queries repeats the bare `<=>` expression rather than referencing a computed alias:

```sql
ORDER BY e.embedding <=> %s::vector   -- HNSW index used
-- ORDER BY distance                  -- alias ref → seq scan
```

### Chunking strategy (800 chars, 100-char overlap)

NWS alert text ranges from a single sentence to multi-paragraph advisories. A fixed 800-character window keeps each chunk under the model's 256-token context window while carrying enough context for a meaningful embedding. The 100-character overlap prevents information loss at boundaries.

Boundary preference (high to low):

1. **Paragraph break (`\n\n`)** — clean semantic boundary; no overlap applied across it.
2. **Sentence end (`.!?` + whitespace)** — most natural split; overlap applied.
3. **Word boundary (any whitespace)** — fallback to avoid mid-token splits.
4. **Hard break** — only when the window contains no whitespace (rare in NWS text).

### Embedding model: `all-MiniLM-L6-v2`

384 dimensions, ~80 MB weights, runs on CPU. It produces good semantic similarity for short English paragraphs, which matches the NWS narrative style. The model is forced to `device="cpu"` to avoid Apple MPS instability in long-running server processes.

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

`apply_schema()` is idempotent (`CREATE … IF NOT EXISTS` throughout).

### Step 2 — Sync and embed (via scheduled job)

The Databricks Asset Bundle deploys a job that syncs NWS data and embeds new documents in one pass:

```bash
# Deploy the bundle (only needed on first deploy or after changes)
databricks bundle deploy --profile DEFAULT

# Run the job immediately
databricks bundle run refresh_weather_index --profile DEFAULT
```

The job is also scheduled every 30 minutes (paused by default — unpause in the Databricks Jobs UI to enable automatic runs).

To trigger a manual sync+embed directly from the API instead:

```bash
curl -X POST http://localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Miami, FL", "Denver, CO"]}'
```

```json
{ "synced": 47, "errors": [] }
```

### Step 3 — Search

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
    }
  ]
}
```

Filter by source type to narrow results to active alerts only:

```bash
curl -X POST http://localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "coastal storm surge", "top_k": 3, "source_type": "alert"}'
```

### Step 4 — AI chat

```bash
curl -X POST http://localhost:8000/weather/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is it safe to travel to Florida this weekend?"}'
```

```json
{
  "question": "Is it safe to travel to Florida this weekend?",
  "answer": "Based on current NWS data, there are active rip current statements along northeast Florida beaches through this evening...",
  "sources": [...]
}
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

### `POST /weather/chat`

RAG endpoint: retrieves relevant NWS chunks, then synthesises a plain-language answer with Llama 3.3 70B via Databricks Foundation Models.

**Request body**

| Field         | Type     | Default  | Description                               |
| ------------- | -------- | -------- | ----------------------------------------- |
| `question`    | `string` | required | Natural-language question                 |
| `top_k`       | `int`    | `5`      | Context chunks to retrieve (clamped 1–10) |
| `source_type` | `string` | `null`   | `"alert"`, `"forecast"`, or omit for both |

**Response**

| Field      | Type       | Description                          |
| ---------- | ---------- | ------------------------------------ |
| `question` | `string`   | Stripped question text               |
| `answer`   | `string`   | AI-generated answer                  |
| `sources`  | `object[]` | Same schema as `/weather/search` results |

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

```bash
export LAKEBASE_CONNECTION_URL="postgresql://user:token@host/databricks_postgres?sslmode=require"
export NWS_USER_AGENT="(YourApp yourname@example.com)"
export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-pat-or-oauth-token>"
```

Generate a short-lived Lakebase token with the CLI:

```bash
EP="projects/<project-id>/branches/production/endpoints/primary"
TOKEN=$(databricks postgres generate-database-credential $EP --profile DEFAULT -o json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
HOST=$(databricks postgres get-endpoint $EP --profile DEFAULT -o json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['status']['hosts']['host'])")
export LAKEBASE_CONNECTION_URL="postgresql://massive_app:$TOKEN@$HOST/databricks_postgres?sslmode=require"
```

### Run the server

```bash
python app.py
```

The Flask dev server binds to `0.0.0.0:8000` by default. Override with the `PORT` environment variable.

### Run the tests

```bash
pytest tests/ -v
```

All external boundaries (HTTP, Postgres wire, sentence-transformer encoder) are mocked. No real database or NWS connection is needed to run the test suite.

---

## Deployment

Everything is managed by a single Databricks Asset Bundle (`databricks.yml`). The bundle defines two targets — `dev` (default, user-scoped workspace path) and `prod` (fixed shared path under `/Workspace/Shared/.bundle/`, used by CI/CD). Both targets deploy the same app resource and refresh job; only resource name prefixes and variable values differ.

### Prerequisites

- Databricks CLI v0.294.0+ authenticated against the target workspace
- Secret scope `database` with key `lakebase-url` containing the Lakebase connection URL:
  ```bash
  databricks secrets create-scope database --profile DEFAULT          # skip if scope exists
  databricks secrets put-secret database lakebase-url \
    --string-value "postgresql://user:pass@host/databricks_postgres?sslmode=require" \
    --profile DEFAULT
  ```
- Schema initialised in Lakebase (one-time):
  ```bash
  LAKEBASE_CONNECTION_URL="<url>" python - <<'EOF'
  from lakebase import apply_schema
  apply_schema()
  print("Schema ready.")
  EOF
  ```

### Deploy

```bash
# 1. Clone
git clone https://github.com/svolodarskyi/weather-lense-ai.git
cd weather-lense-ai

# 2. Deploy bundle (creates app + job resources)
databricks bundle deploy --profile DEFAULT

# 3. Deploy source code to the app
databricks apps deploy weather-lens-ai \
  --source-code-path /Workspace/Users/$(databricks current-user me --profile DEFAULT -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['user_name'])")/.bundle/weather-lens-ai/dev/files \
  --profile DEFAULT

# 4. Start the app (if not already running)
databricks bundle run weather_lens_ai --profile DEFAULT
```

The app's service principal automatically gets `READ` access to the `database/lakebase-url` secret via the bundle resource definition — no manual permission grants needed.

`DATABRICKS_TOKEN` is **not** needed in the deployed app. Databricks Apps auto-injects `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`; the LLM client exchanges these for a short-lived M2M OAuth token on each request. `DATABRICKS_TOKEN` is only required for local development (see [Local development](#local-development)).

### Run the index refresh job

```bash
# Trigger immediately (for testing or first load)
databricks bundle run refresh_weather_index --profile DEFAULT
```

The job is scheduled every 30 minutes but starts paused. To enable automatic runs, change `pause_status: UNPAUSED` in `resources/refresh_weather_index_job.yml` and redeploy, or unpause it directly in the Databricks Jobs UI.

### Redeploy after code changes (dev)

```bash
git pull
databricks bundle deploy --profile DEFAULT
databricks apps deploy weather-lens-ai \
  --source-code-path /Workspace/Users/<your-email>/.bundle/weather-lens-ai/dev/files \
  --profile DEFAULT
```

---

## CI/CD

The repository ships two GitHub Actions workflows:

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **CI** | `.github/workflows/ci.yml` | PR to `main`, push to any non-main branch | Runs pytest, then validates the DABs bundle against the workspace |
| **CD** | `.github/workflows/cd.yml` | Push / merge to `main` | Re-runs tests, deploys bundle to `prod` target, pushes app source code |

### Required GitHub secrets

Add these under **Settings → Secrets → Actions** in the repository:

| Secret | Value |
|---|---|
| `DATABRICKS_HOST` | `https://dbc-432266ff-3fab.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | A Databricks personal access token (or use `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` for service-principal auth) |

### How prod deploys work

The `prod` target in `databricks.yml` sets a fixed workspace root path:

```
/Workspace/Shared/.bundle/weather-lens-ai/prod/
```

This makes the path predictable regardless of which user or service principal runs the workflow. The CD workflow deploys to this path and then calls:

```bash
databricks apps deploy weather-lens-ai \
  --source-code-path /Workspace/Shared/.bundle/weather-lens-ai/prod/files
```

No hard-coded usernames, no dynamic path resolution at deploy time.

### Bundle variables

Key parameters are declared as bundle variables in `databricks.yml` and can be overridden on the CLI or per-target:

```bash
# Deploy with a custom set of locations
databricks bundle deploy --target prod \
  --var="locations=Boston, MA;Portland, OR;Nashville, TN"

# Use a different secret scope/key
databricks bundle deploy --var="secret_scope=myapp" --var="secret_key=db-url"
```

---

## Deliverables

This section maps each assignment deliverable to the implementation, explains design decisions, and documents known limitations.

---

### Data source: National Weather Service API

**Chosen source:** [api.weather.gov](https://api.weather.gov) — NWS is the recommended source and the natural fit: no API key, generous rate limits, and rich free-text narrative fields ideal for embedding. Two endpoints are used:

| Endpoint | Narrative field embedded |
|---|---|
| `GET /alerts/active?area={state}` | `description` + `instruction` (concatenated) |
| `GET /gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

Geocoding (city → lat/lon → NWS grid) is done via Open-Meteo's free geocoding API, removing any manual coordinate lookup.

---

### Required deliverables

#### 1. `weather_client.py` — NWS harvesting client

Mirrors the `massive_client.py` pattern from the reference app. Given a list of city/state strings, `WeatherClient.fetch()`:

1. Geocodes each location (Open-Meteo → lat/lon).
2. Resolves NWS grid point (`/points/{lat},{lon}`).
3. Fetches active alerts (`/alerts/active?area={state}`) and 7-day forecast (`/gridpoints/{office}/{x},{y}/forecast`).
4. Normalises each item into a document dict with stable dedup key:
   - Alerts: `id` = NWS alert ID (`urn:oid:…` string from the API).
   - Forecasts: `id` = `SHA-256(location + period_name + narrative_text)`.
5. Computes `content_hash = SHA-256(narrative_text)` so amended alerts are detected without re-embedding unchanged text.

A built-in token-bucket rate-limiter (1 req/s) prevents hitting NWS's soft rate limit.

#### 2. `app.py` — Flask REST API with `/weather/sync` and `/weather/search`

Both endpoints required by the assignment are implemented, plus a bonus RAG chat endpoint:

| Endpoint | Description |
|---|---|
| `POST /weather/sync` | Harvests NWS data for a list of locations, upserts into `weather_documents`, clears stale embeddings |
| `POST /weather/search` | Embeds query with the same model, runs HNSW cosine search, returns ranked chunks with similarity scores |
| `POST /weather/chat` | Retrieves top-k chunks (same as search), synthesises a plain-language answer with Llama 3.3 70B (stretch goal) |
| `GET /health` | Liveness probe — never blocks on DB or model |

The encoder (`SentenceTransformer`) is instantiated once at module level and shared across workers via gunicorn's `--preload` flag, avoiding the ~2 s model load on every request.

#### 3. `lakebase.py` + `schema.sql` + `repository.py` — schema and data access

- **`lakebase.py`** — `get_lakebase_connection()` context manager using `psycopg2` + `LAKEBASE_CONNECTION_URL`.
- **`schema.sql`** — idempotent DDL (`CREATE … IF NOT EXISTS`) for both tables plus the HNSW index; applied by `apply_schema()` which is called on startup.
- **`repository.py`** — all SQL in one place: `upsert_documents`, `clear_stale_embeddings`, `find_unembedded`, `insert_embeddings`, and `search`. The search query uses pgvector's `<=>` cosine-distance operator; the `ORDER BY` clause repeats the bare expression (not an alias) so the HNSW index is used.

Schema:
```sql
-- Raw NWS text, one row per alert or forecast period
weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT, source_type TEXT, event TEXT,
    narrative_text TEXT, content_hash TEXT,
    issued_at TIMESTAMPTZ, synced_at TIMESTAMPTZ, payload JSONB
)

-- One row per text chunk; FK cascades on delete
weather_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT, chunk_text TEXT,
    embedding VECTOR(384), content_hash TEXT,
    model_name TEXT, created_at TIMESTAMPTZ
)

CREATE INDEX weather_embeddings_hnsw
ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
```

#### 4. `notebooks/refresh_weather_index.py` — psycopg2-based embedding job

A Databricks Notebook (deployed as a scheduled Job via Asset Bundles) that in a single pass:

1. Re-syncs NWS data for a fixed set of locations.
2. Calls `repository.clear_stale_embeddings()` to delete rows whose `content_hash` no longer matches the parent document (amended alerts).
3. Reads unembedded documents with `repository.find_unembedded()`.
4. Chunks each `narrative_text` with a sliding window (800 chars, 100-char overlap, snapped to sentence/word boundaries).
5. Embeds chunks in batches of 64 with `all-MiniLM-L6-v2` (384-dim, CPU).
6. Writes rows to `weather_embeddings` via `psycopg2` `execute_values` — no Spark JDBC.

The connection string (`LAKEBASE_CONNECTION_URL`) is read from the `database/lakebase-url` Databricks secret — the same scope used by the app.

---

### Stretch goals completed

| Goal | Implementation |
|---|---|
| LLM-generated summary (RAG) | `POST /weather/chat` — retrieves top-k chunks, synthesises answer with Llama 3.3 70B via Databricks Foundation Models OpenAI-compatible endpoint |
| Dedup / upsert on stable ID | `repository.upsert_documents()` uses `ON CONFLICT (id) DO UPDATE`; `content_hash` detects amendments without re-embedding unchanged text |
| Scheduled Databricks Job | `resources/refresh_weather_index_job.yml` deploys via Asset Bundles; scheduled every 30 min (paused by default — unpause in the Jobs UI) |
| Filter by `source_type` | Both `/weather/search` and `/weather/chat` accept an optional `"source_type": "alert"|"forecast"` body field |
| HNSW index | Created in `schema.sql`; `ORDER BY e.embedding <=> %s::vector` expression is repeated verbatim so the planner uses the index |

---

### Known limitations and future improvements

**Similarity threshold filtering** — The current retrieval returns the top-k results by cosine distance regardless of how relevant they are. A query about "hurricane risk in Texas" can surface a Chicago forecast with a low similarity score simply because it is the closest match in a sparse index. A minimum-similarity cutoff (e.g. discard results below 0.3) or a cross-encoder reranker would eliminate these irrelevant results.

**Edge cases not yet handled:**
- NWS occasionally returns `null` for `detailedForecast` on some grid cells; these documents are currently stored with empty `narrative_text` and produce a zero-length embedding.
- Very long combined alert descriptions (>4 KB) are chunked but the overlap logic does not account for paragraph boundaries that span multiple NWS fields.
- The geocoder returns the first match for a city name; ambiguous names (e.g. "Springfield") resolve to whichever city Open-Meteo ranks first.

**Multi-state alert deduplication** — NWS may issue the same alert across multiple states under different IDs. There is no cross-location dedup, so the same physical event can appear multiple times in search results.

**Token refresh for the Lakebase connection** — The `LAKEBASE_CONNECTION_URL` is stored as a static secret. Lakebase OAuth tokens expire after 1 hour. In production, a token-refresh loop (as shown in the Lakebase connectivity docs) should replace the static URL.
