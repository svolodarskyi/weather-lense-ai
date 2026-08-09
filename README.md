# WeatherLens AI

Harvests NWS weather text, embeds it with a sentence transformer, stores vectors in Databricks Lakebase (pgvector), and serves semantic search + AI-powered chat via a Flask REST API.

**Live app:** https://weather-lens-ai-7474648001102301.aws.databricksapps.com

![WeatherLens AI — search results and AI summary](docs/screenshot-search.png)

---

## Architecture

```
NWS API (api.weather.gov)
        │
        ▼
  weather_client.py       ← harvest alerts + forecasts, geocode cities
        │
        ▼
  weather_documents        ← raw NWS text in Lakebase (Postgres)
        │
        ▼
  refresh_weather_index    ← chunk → embed → write VECTOR(384) rows
  (Databricks Job, 30 min) ← deployed via Asset Bundles
        │
        ▼
  weather_embeddings       ← pgvector table with HNSW index
        │
        ▼
  Flask API (app.py)       ← /weather/sync  /weather/search  /weather/chat
        │
        ▼
  Llama 3.3 70B            ← RAG answers via Databricks Foundation Models
```

---

## Data source

**NWS API (`api.weather.gov`)** — no API key, generous rate limits, rich narrative text ideal for embedding:

| Endpoint | Text embedded |
|---|---|
| `GET /alerts/active?area={state}` | `description` + `instruction` |
| `GET /gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

City names are geocoded via Open-Meteo (free, no key) → lat/lon → NWS grid.

---

## Schema

```sql
-- Raw NWS text, one row per alert or forecast period
weather_documents (
    id TEXT PRIMARY KEY,           -- stable NWS alert ID or SHA-256 hash
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

**Key decisions:**
- Two tables keep the HNSW index narrow (embeddings only) and let NWS metadata be queried without touching vectors.
- `content_hash` on both tables detects amended alerts: a hash mismatch deletes stale embedding rows so they are re-embedded on the next job run.
- HNSW over IVFFlat: no training phase, inserts incrementally — better for a continuously updated pipeline.
- Chunking: 800-char window, 100-char overlap, snapped to sentence/word boundaries. `ORDER BY` repeats the raw `<=>` expression (not an alias) so the planner uses the index.
- Model: `all-MiniLM-L6-v2` (384-dim, CPU) — good semantic similarity for short English paragraphs.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
export LAKEBASE_CONNECTION_URL="postgresql://user:token@host/databricks_postgres?sslmode=require"
export NWS_USER_AGENT="WeatherLensAI/1.0 yourname@example.com"
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<pat>"
```

```bash
# Initialise schema (one-time)
python -c "from lakebase import apply_schema; apply_schema()"

# Run server
python app.py

# Run tests (all I/O mocked — no DB or NWS connection needed)
pytest tests/ -v
```

---

## Deployment

Managed by a Databricks Asset Bundle (`databricks.yml`). Two targets: `dev` (default, user-scoped path) and `prod` (fixed path at `/Workspace/Shared/.bundle/…`, used by CI/CD).

**One-time prerequisites:**
```bash
# Store Lakebase connection URL as a secret
databricks secrets create-scope database --profile DEFAULT   # skip if exists
databricks secrets put-secret database lakebase-url \
  --string-value "postgresql://..." --profile DEFAULT

# Initialise schema
LAKEBASE_CONNECTION_URL="<url>" python -c "from lakebase import apply_schema; apply_schema()"
```

**Deploy:**
```bash
git clone https://github.com/svolodarskyi/weather-lense-ai.git && cd weather-lense-ai

databricks bundle deploy --profile DEFAULT

databricks apps deploy weather-lens-ai \
  --source-code-path /Workspace/Users/$(databricks current-user me --profile DEFAULT -o json \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['user_name'])")/.bundle/weather-lens-ai/dev/files \
  --profile DEFAULT
```

The app's service principal gets `READ` access to `database/lakebase-url` automatically via the bundle resource definition. No `DATABRICKS_TOKEN` needed in the deployed app — it uses M2M OAuth (`DATABRICKS_CLIENT_ID`/`SECRET` auto-injected by Databricks Apps).

**Trigger the index job:**
```bash
databricks bundle run refresh_weather_index --profile DEFAULT
```

The job runs every 30 min (paused by default — unpause in the Jobs UI or set `pause_status: UNPAUSED` in `resources/refresh_weather_index_job.yml`).

**Override locations or secret scope via variables:**
```bash
databricks bundle deploy --var="locations=Boston, MA;Portland, OR" --var="secret_scope=myapp"
```

---

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | PR to `main`, push to non-main branch | `pytest` + `databricks bundle validate` |
| `cd.yml` | Push/merge to `main` | `pytest` → `bundle deploy --target prod` → `apps deploy` |

**Required GitHub secrets** (Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `DATABRICKS_HOST` | `https://dbc-432266ff-3fab.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Databricks PAT (or swap for `DATABRICKS_CLIENT_ID`/`SECRET`) |

---

## Deliverables

### Live demo

Search: *"illinois weather risks"* — 5 results with AI summary (Llama 3.3 70B):

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

---

### Stretch goals completed

| Goal | How |
|---|---|
| LLM-generated summary | `POST /weather/chat` — top-k chunks → Llama 3.3 70B via Foundation Models OpenAI-compatible endpoint |
| Dedup / upsert | `ON CONFLICT (id) DO UPDATE`; `content_hash` skips re-embedding unchanged text |
| Scheduled job | DABs job every 30 min, email on failure |
| Filter by `source_type` | Both search and chat accept `"source_type": "alert"` or `"forecast"` |
| HNSW index | `vector_cosine_ops`, query expression repeated verbatim so planner uses the index |

---

### Known limitations and what I'd improve

- **Similarity threshold** — retrieval returns top-k regardless of score; a minimum cutoff (~0.3) or cross-encoder reranker would eliminate unrelated forecasts surfacing in results.
- **Null forecasts** — NWS occasionally returns `null` for `detailedForecast`; these produce empty embeddings and pollute search results.
- **Ambiguous geocoding** — "Springfield" resolves to whichever city Open-Meteo ranks first; no disambiguation.
- **Multi-state alert duplication** — the same physical event issued across states under different IDs appears multiple times in results.
- **Lakebase token expiry** — the connection URL is a static secret; tokens expire after 1 hour and should be refreshed automatically in production.
