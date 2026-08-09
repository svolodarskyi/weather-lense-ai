"""
Sprint 3 tests — app.py (/health and /weather/sync).

WeatherClient, Lakebase connection, and repository functions are mocked.
No real HTTP or database calls are made.
"""

import os
import sys

# Must be set before app is imported — the module-level WeatherClient reads it.
os.environ.setdefault("NWS_USER_AGENT", "(WeatherLens AI test, test@test.com)")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import MagicMock, patch

import app as app_module
from weather_client import WeatherDoc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(**overrides) -> WeatherDoc:
    defaults = dict(
        id="urn:oid:2.49.0.1.840.0.test",
        location="Chicago, IL",
        latitude=41.8781,
        longitude=-87.6298,
        source_type="alert",
        event="Flood Watch",
        headline="Flood Watch until Monday",
        narrative_text="Heavy rain is expected. Move to higher ground.",
        issued_at="2026-08-08T00:00:00Z",
        effective_at="2026-08-08T06:00:00Z",
        expires_at="2026-08-09T18:00:00Z",
        severity="Moderate",
        payload={"type": "Feature"},
    )
    defaults.update(overrides)
    return WeatherDoc(**defaults)


def _make_lakebase_ctx():
    """Return a MagicMock that acts as a context manager yielding a connection."""
    conn = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, conn


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_status_ok(self, client):
        data = resp = client.get("/health")
        assert resp.get_json() == {"status": "ok"}

    def test_does_not_hit_database(self, client):
        with patch.object(app_module, "get_lakebase_connection") as mock_conn:
            client.get("/health")
        mock_conn.assert_not_called()


# ---------------------------------------------------------------------------
# POST /weather/sync — validation
# ---------------------------------------------------------------------------

class TestWeatherSyncValidation:
    def test_missing_locations_key_returns_400(self, client):
        resp = client.post("/weather/sync",
                           data=json.dumps({}), content_type="application/json")
        assert resp.status_code == 400

    def test_empty_list_returns_400(self, client):
        resp = client.post("/weather/sync",
                           json={"locations": []})
        assert resp.status_code == 400

    def test_string_locations_returns_400(self, client):
        resp = client.post("/weather/sync",
                           json={"locations": "Chicago, IL"})
        assert resp.status_code == 400

    def test_null_locations_returns_400(self, client):
        resp = client.post("/weather/sync",
                           json={"locations": None})
        assert resp.status_code == 400

    def test_missing_body_returns_400(self, client):
        resp = client.post("/weather/sync")
        assert resp.status_code == 400

    def test_error_response_contains_error_key(self, client):
        resp = client.post("/weather/sync", json={})
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# POST /weather/sync — happy path
# ---------------------------------------------------------------------------

class TestWeatherSyncHappyPath:
    def _sync(self, client, locations=None, extra=None):
        """Run a sync with mocked client, Lakebase, and repository."""
        docs = [_make_doc(id=f"id-{i}") for i in range(len(locations or ["Chicago, IL"]))]
        mock_client = MagicMock()
        mock_client.fetch.return_value = docs
        mock_client.errors = []

        ctx, conn = _make_lakebase_ctx()

        body = {"locations": locations or ["Chicago, IL"]}
        if extra:
            body.update(extra)

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=len(docs)), \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0):
            resp = client.post("/weather/sync", json=body)

        return resp, mock_client, docs

    def test_returns_200(self, client):
        resp, _, _ = self._sync(client)
        assert resp.status_code == 200

    def test_synced_equals_doc_count(self, client):
        locations = ["Chicago, IL", "Miami, FL"]
        resp, _, docs = self._sync(client, locations=locations)
        assert resp.get_json()["synced"] == len(docs)

    def test_errors_list_in_response(self, client):
        resp, _, _ = self._sync(client)
        data = resp.get_json()
        assert "errors" in data
        assert isinstance(data["errors"], list)

    def test_client_errors_forwarded(self, client):
        mock_client = MagicMock()
        mock_client.fetch.return_value = []
        mock_client.errors = ["Bad location: Nowhere, XX"]
        ctx, _ = _make_lakebase_ctx()

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=0), \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0):
            resp = client.post("/weather/sync", json={"locations": ["Nowhere, XX"]})

        data = resp.get_json()
        assert data["errors"] == ["Bad location: Nowhere, XX"]

    def test_calls_fetch_with_locations(self, client):
        locations = ["Chicago, IL", "Austin, TX"]
        _, mock_client, _ = self._sync(client, locations=locations)
        mock_client.fetch.assert_called_once()
        call_args = mock_client.fetch.call_args
        assert call_args[0][0] == locations

    def test_default_limit_is_50(self, client):
        _, mock_client, _ = self._sync(client, locations=["Chicago, IL"])
        _, kwargs = mock_client.fetch.call_args
        positional = mock_client.fetch.call_args[0]
        # limit is passed as second positional arg or keyword
        limit = positional[1] if len(positional) > 1 else kwargs.get("limit", 50)
        assert limit == 50

    def test_custom_limit_forwarded_to_fetch(self, client):
        mock_client = MagicMock()
        mock_client.fetch.return_value = []
        mock_client.errors = []
        ctx, _ = _make_lakebase_ctx()

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=0), \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0):
            client.post("/weather/sync", json={"locations": ["Chicago, IL"], "limit": 10})

        positional = mock_client.fetch.call_args[0]
        kwargs = mock_client.fetch.call_args[1]
        limit = positional[1] if len(positional) > 1 else kwargs.get("limit")
        assert limit == 10

    def test_calls_upsert_documents(self, client):
        mock_client = MagicMock()
        mock_client.fetch.return_value = [_make_doc()]
        mock_client.errors = []
        ctx, _ = _make_lakebase_ctx()

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=1) as mock_up, \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0):
            client.post("/weather/sync", json={"locations": ["Chicago, IL"]})

        mock_up.assert_called_once()

    def test_calls_clear_stale_embeddings(self, client):
        mock_client = MagicMock()
        mock_client.fetch.return_value = [_make_doc()]
        mock_client.errors = []
        ctx, _ = _make_lakebase_ctx()

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=1), \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0) as mock_clear:
            client.post("/weather/sync", json={"locations": ["Chicago, IL"]})

        mock_clear.assert_called_once()

    def test_zero_docs_returns_synced_zero(self, client):
        mock_client = MagicMock()
        mock_client.fetch.return_value = []
        mock_client.errors = []
        ctx, _ = _make_lakebase_ctx()

        with patch.object(app_module, "client", mock_client), \
             patch.object(app_module, "get_lakebase_connection", return_value=ctx), \
             patch.object(app_module.repository, "upsert_documents", return_value=0), \
             patch.object(app_module.repository, "clear_stale_embeddings", return_value=0):
            resp = client.post("/weather/sync", json={"locations": ["Nowhere, AK"]})

        assert resp.get_json()["synced"] == 0
