# WeatherLens AI

Harvests NWS weather text, embeds it with a sentence transformer, stores vectors in Databricks Lakebase (pgvector), and serves semantic search + AI-powered chat via a Flask REST API.

**Live app:** https://weather-lens-ai-7474648001102301.aws.databricksapps.com

![WeatherLens AI — search results and AI summary](docs/screenshot-search.png)

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Data sources](#data-sources)
3. [Schema decisions](#schema-decisions)
4. [End-to-end walkthrough](#end-to-end-walkthrough)
5. [Local development](#local-development)
6. [Deployment](#deployment)
7. [CI/CD](#cicd)
8. [Deliverables](#deliverables)

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

| Endpoint | What it returns |
|---|---|
| `GET /alerts/active?area={state}` | Active weather alerts (Flood Watch, Tornado Warning, Heat Advisory, …) for a US state |
| `GET /gridpoints/{office}/{x},{y}/forecast` | 7-day period-by-period forecast for a lat/lon grid cell |

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

Filter to active alerts only:

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

`DATABRICKS_TOKEN` is **not** needed in the deployed app. Databricks Apps auto-injects `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`; the LLM client exchanges these for a short-lived M2M OAuth token on each request. `DATABRICKS_TOKEN` is only required for local development.

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

**Override locations or secret scope via variables:**
```bash
databricks bundle deploy --var="locations=Boston, MA;Portland, OR" --var="secret_scope=myapp"
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

The `prod` target sets a fixed workspace root path (`/Workspace/Shared/.bundle/weather-lens-ai/prod`) so the deploy path is stable regardless of which user or service principal runs the workflow — no hard-coded usernames in CI.

---

## Deliverables

### Live demo

Search query: *"illinois weather risks"* — 5 semantically matched results with AI summary generated by Llama 3.3 70B:

![WeatherLens AI — search results and AI summary](docs/screenshot-search.png)

---

### Files

| File | Deliverable |
|---|---|
| `weather_client.py` | NWS harvesting client — geocodes cities, fetches alerts + forecasts, normalises to document schema, rate-limits at 1 req/s |
| `app.py` | Flask API — `/weather/sync`, `/weather/search`, `/weather/chat`, `/health` |
| `lakebase.py` + `schema.sql` + `repository.py` | Connection helper, idempotent DDL, all SQL (upsert, search, stale-embedding cleanup) |
| `embeddings.py` | Chunker (800-char window, 100-char overlap) + `Encoder` wrapper around `all-MiniLM-L6-v2` |
| `notebooks/refresh_weather_index.py` | Databricks Notebook job — syncs NWS → clears stale embeddings → embeds new chunks via `psycopg2` `execute_values` (no Spark JDBC) |
| `resources/*.yml` + `databricks.yml` | Asset Bundle — deploys app + scheduled job, wires `database/lakebase-url` secret, exposes variables for locations/scope/key |
| `.github/workflows/` | CI (test + validate) and CD (deploy to prod on merge to main) |

### Data source

**NWS API (`api.weather.gov`)** — no API key, generous rate limits, rich free-text narrative fields ideal for embedding:

| Endpoint | Text embedded |
|---|---|
| `GET /alerts/active?area={state}` | `description` + `instruction` |
| `GET /gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

Geocoding (city → lat/lon → NWS grid) via Open-Meteo's free API.

### Schema decisions

- **Two tables** keep the HNSW index narrow and NWS metadata queryable without touching vectors. `ON DELETE CASCADE` cleanly removes orphaned chunks.
- **`content_hash`** on both tables detects amended alerts: a mismatch deletes stale embedding rows so they are re-embedded on the next job run.
- **HNSW over IVFFlat** — no training phase, inserts incrementally, consistent quality for a continuously updated pipeline.
- **800-char chunks, 100-char overlap**, snapped to sentence/word boundaries. `ORDER BY` repeats the raw `<=>` expression (not an alias) so the planner uses the index.
- **`all-MiniLM-L6-v2`** (384-dim, CPU) — good semantic similarity for short English paragraphs; `device="cpu"` avoids Apple MPS instability in server processes.

### Stretch goals completed

| Goal | How |
|---|---|
| LLM-generated summary | `POST /weather/chat` — top-k chunks → Llama 3.3 70B via Foundation Models OpenAI-compatible endpoint |
| Dedup / upsert | `ON CONFLICT (id) DO UPDATE`; `content_hash` skips re-embedding unchanged text |
| Scheduled job | DABs job every 30 min, email on failure |
| Filter by `source_type` | Both search and chat accept `"source_type": "alert"` or `"forecast"` |
| HNSW index | `vector_cosine_ops`, query expression repeated verbatim so planner uses the index |

### Known limitations and what I'd improve

- **Similarity threshold** — retrieval returns top-k regardless of score; a minimum cutoff (~0.3) or cross-encoder reranker would eliminate unrelated forecasts surfacing in results.
- **Null forecasts** — NWS occasionally returns `null` for `detailedForecast`; these produce empty embeddings and pollute search results.
- **Ambiguous geocoding** — "Springfield" resolves to whichever city Open-Meteo ranks first; no disambiguation.
- **Multi-state alert duplication** — the same physical event issued across states under different IDs appears multiple times in results.
- **Lakebase token expiry** — the connection URL is a static secret; tokens expire after 1 hour and should be refreshed automatically in production.
