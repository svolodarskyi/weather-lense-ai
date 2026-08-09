"""
WeatherLens AI — Flask REST API.

Endpoints (Sprint 3):
  GET  /health         — liveness probe; never blocks on model or DB load
  POST /weather/sync   — harvest NWS text for a list of locations, upsert to Lakebase

Endpoints (Sprint 5):
  POST /weather/search — semantic search over stored embeddings

The WeatherClient is instantiated once at module level so its requests.Session
and rate-limiting state persist across calls. A per-request client would reset
the pacer on every request and lose the benefit of connection reuse.
"""

import os
from flask import Flask, jsonify, request
from lakebase import get_lakebase_connection
from weather_client import WeatherClient
import repository

app = Flask(__name__)

client = WeatherClient(user_agent=os.getenv("NWS_USER_AGENT", ""))


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
