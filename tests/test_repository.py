"""
Sprint 3 tests — repository.py (write side).

psycopg2 is mocked; no real database required.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weather_client import WeatherDoc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


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
        payload={"type": "Feature", "properties": {}},
    )
    defaults.update(overrides)
    return WeatherDoc(**defaults)


# ---------------------------------------------------------------------------
# upsert_documents
# ---------------------------------------------------------------------------

class TestUpsertDocuments:
    def test_empty_list_returns_zero_without_db_call(self):
        conn, _ = _make_conn()
        from repository import upsert_documents
        result = upsert_documents(conn, [])
        assert result == 0
        conn.cursor.assert_not_called()

    def test_calls_execute_values_once(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        mock_ev.assert_called_once()

    def test_sql_inserts_into_weather_documents(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        sql = mock_ev.call_args[0][1]
        assert "INSERT INTO weather_documents" in sql

    def test_sql_has_on_conflict_do_update(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        sql = mock_ev.call_args[0][1]
        assert "ON CONFLICT (id)" in sql
        assert "DO UPDATE SET" in sql

    def test_sql_updates_narrative_text_and_content_hash(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        sql = mock_ev.call_args[0][1]
        assert "narrative_text" in sql
        assert "content_hash" in sql

    def test_sql_updates_headline_and_timestamps(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        sql = mock_ev.call_args[0][1]
        assert "headline" in sql
        assert "issued_at" in sql
        assert "effective_at" in sql
        assert "expires_at" in sql

    def test_sql_sets_synced_at_to_now(self):
        conn, cursor = _make_conn()
        with patch("repository.execute_values") as mock_ev:
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        sql = mock_ev.call_args[0][1]
        assert "synced_at" in sql
        assert "NOW()" in sql

    def test_returns_len_docs(self):
        conn, cursor = _make_conn()
        docs = [_make_doc(id=f"id-{i}") for i in range(5)]
        with patch("repository.execute_values"):
            from repository import upsert_documents
            result = upsert_documents(conn, docs)
        assert result == 5

    def test_payload_wrapped_as_json_adapter(self):
        """Payload must be wrapped in psycopg2.extras.Json, not passed as raw dict.

        psycopg2 cannot serialise a plain Python dict to JSONB; passing it raw
        raises `can't adapt type 'dict'`. The Json wrapper tells the adapter layer
        to serialise with json.dumps on the way to the wire.
        """
        conn, cursor = _make_conn()
        captured = {}
        def capture(cur, sql, rows, **kwargs):
            captured["rows"] = rows
        with patch("repository.execute_values", side_effect=capture):
            from repository import upsert_documents
            upsert_documents(conn, [_make_doc()])
        payload_arg = captured["rows"][0][12]   # payload is the 13th column (index 12)
        assert type(payload_arg).__name__ == "Json"

    def test_rows_match_doc_count(self):
        conn, cursor = _make_conn()
        docs = [_make_doc(id=f"id-{i}") for i in range(3)]
        captured = {}
        def capture(cur, sql, rows, **kwargs):
            captured["rows"] = rows
        with patch("repository.execute_values", side_effect=capture):
            from repository import upsert_documents
            upsert_documents(conn, docs)
        assert len(captured["rows"]) == 3

    def test_row_contains_doc_id_and_location(self):
        conn, cursor = _make_conn()
        doc = _make_doc(id="special-id", location="Miami, FL")
        captured = {}
        def capture(cur, sql, rows, **kwargs):
            captured["rows"] = rows
        with patch("repository.execute_values", side_effect=capture):
            from repository import upsert_documents
            upsert_documents(conn, [doc])
        row = captured["rows"][0]
        assert row[0] == "special-id"
        assert row[1] == "Miami, FL"


# ---------------------------------------------------------------------------
# clear_stale_embeddings
# ---------------------------------------------------------------------------

class TestClearStaleEmbeddings:
    def test_calls_execute(self):
        conn, cursor = _make_conn()
        from repository import clear_stale_embeddings
        clear_stale_embeddings(conn)
        cursor.execute.assert_called_once()

    def test_sql_targets_weather_embeddings(self):
        conn, cursor = _make_conn()
        from repository import clear_stale_embeddings
        clear_stale_embeddings(conn)
        sql = cursor.execute.call_args[0][0]
        assert "weather_embeddings" in sql

    def test_sql_joins_weather_documents(self):
        conn, cursor = _make_conn()
        from repository import clear_stale_embeddings
        clear_stale_embeddings(conn)
        sql = cursor.execute.call_args[0][0]
        assert "weather_documents" in sql

    def test_sql_checks_content_hash_mismatch(self):
        conn, cursor = _make_conn()
        from repository import clear_stale_embeddings
        clear_stale_embeddings(conn)
        sql = cursor.execute.call_args[0][0]
        assert "content_hash" in sql
        assert "!=" in sql or "<>" in sql

    def test_sql_matches_on_document_id(self):
        conn, cursor = _make_conn()
        from repository import clear_stale_embeddings
        clear_stale_embeddings(conn)
        sql = cursor.execute.call_args[0][0]
        assert "document_id" in sql

    def test_returns_rowcount(self):
        conn, cursor = _make_conn()
        cursor.rowcount = 7
        from repository import clear_stale_embeddings
        result = clear_stale_embeddings(conn)
        assert result == 7
