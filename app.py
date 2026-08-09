"""
WeatherLens AI — Flask REST API.

Endpoints:
  GET  /health         — liveness probe; never blocks on model or DB load
  POST /weather/sync   — harvest NWS text for a list of locations, upsert to Lakebase
  POST /weather/search — semantic search over stored embeddings

Both WeatherClient and Encoder are instantiated once at module level:
  - WeatherClient owns a requests.Session and a rate-limiter; per-request
    creation would reset the pacer and lose connection reuse.
  - Encoder loads ~80 MB of model weights; per-request creation would
    re-read them on every search call.
"""

import os
from flask import Flask, jsonify, render_template, request
from lakebase import get_lakebase_connection
from weather_client import WeatherClient
from embeddings import Encoder, MODEL_NAME
import repository

app = Flask(__name__)

client  = WeatherClient(user_agent=os.getenv("NWS_USER_AGENT", ""))
encoder = Encoder(MODEL_NAME)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/weather/sync")
def weather_sync():
    body = request.get_json(silent=True) or {}

    locations = body.get("locations")
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "locations must be a non-empty list of strings"}), 400

    # Clamp limit to a sane default; ignore non-integer values from callers.
    raw_limit = body.get("limit", 50)
    limit = raw_limit if isinstance(raw_limit, int) and raw_limit > 0 else 50

    docs = client.fetch(locations, limit=limit)

    with get_lakebase_connection() as conn:
        repository.upsert_documents(conn, docs)
        repository.clear_stale_embeddings(conn)

    return jsonify({"synced": len(docs), "errors": client.errors})


@app.post("/weather/search")
def weather_search():
    body = request.get_json(silent=True) or {}

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query must be a non-empty string"}), 400

    top_k = body.get("top_k", 5)
    if not isinstance(top_k, int):
        top_k = 5
    top_k = max(1, min(20, top_k))

    # Optional filter; absent or null means search across both source types.
    source_type = body.get("source_type") or None

    q = query.strip()
    [q_vec] = encoder.encode([q])

    with get_lakebase_connection() as conn:
        results = repository.search(conn, q_vec, top_k, source_type=source_type)

    return jsonify({"query": q, "count": len(results), "results": results})
