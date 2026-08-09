"""
Sprint 3 + Sprint 5 tests — repository.py (write side and read side).

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


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def _make_search_row(doc_id="doc-1", location="Chicago, IL", event="Flood Watch",
                     source_type="alert", chunk_text="Heavy rain.", distance=0.15):
    """Return a tuple matching the SELECT column order in _SEARCH_SQL."""
    return (doc_id, location, event, source_type, chunk_text, doc_id, distance)


class TestSearch:
    _vec = [0.1] * 384

    def _run(self, rows, top_k=5, source_type=None):
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = rows
        from repository import search
        result = search(conn, self._vec, top_k, source_type=source_type)
        return result, cursor

    def test_empty_table_returns_empty_list(self):
        result, _ = self._run([])
        assert result == []

    def test_returns_result_for_each_unique_doc(self):
        rows = [_make_search_row("d1"), _make_search_row("d2")]
        result, _ = self._run(rows, top_k=5)
        assert len(result) == 2

    def test_deduplicates_by_document_id(self):
        # Two rows for the same document — only one should appear in results.
        rows = [
            _make_search_row("d1", distance=0.1),
            _make_search_row("d1", distance=0.3),
        ]
        result, _ = self._run(rows, top_k=5)
        assert len(result) == 1

    def test_keeps_lowest_distance_on_dedup(self):
        # DB already returns rows ordered by distance; first occurrence wins.
        rows = [
            _make_search_row("d1", distance=0.1),   # lower distance → kept
            _make_search_row("d1", distance=0.3),
        ]
        result, _ = self._run(rows, top_k=5)
        assert abs(result[0]["similarity"] - (1.0 - 0.1)) < 1e-5

    def test_similarity_equals_one_minus_distance(self):
        rows = [_make_search_row("d1", distance=0.25)]
        result, _ = self._run(rows, top_k=5)
        assert abs(result[0]["similarity"] - 0.75) < 1e-5

    def test_top_k_caps_result_count(self):
        rows = [_make_search_row(f"d{i}", distance=0.1 * i) for i in range(10)]
        result, _ = self._run(rows, top_k=3)
        assert len(result) == 3

    def test_limit_is_top_k_times_three(self):
        _, cursor = self._run([], top_k=4)
        sql, params = cursor.execute.call_args[0]
        # Last param in params tuple is the LIMIT value = top_k * 3 = 12
        assert params[-1] == 12

    def test_sql_uses_bare_cosine_distance_operator(self):
        _, cursor = self._run([], top_k=1)
        sql = cursor.execute.call_args[0][0]
        assert "<=>" in sql

    def test_sql_order_by_repeats_expression_not_alias(self):
        # ORDER BY must use the expression, not the alias "distance",
        # so the HNSW index can be used.
        _, cursor = self._run([], top_k=1)
        sql = cursor.execute.call_args[0][0]
        order_by_section = sql[sql.upper().index("ORDER BY"):]
        assert "<=>" in order_by_section

    def test_sql_does_not_use_one_minus_form(self):
        # "1 - (...) DESC" cannot use the HNSW index.
        _, cursor = self._run([], top_k=1)
        sql = cursor.execute.call_args[0][0]
        assert "1 -" not in sql and "1-" not in sql

    def test_result_dict_has_required_keys(self):
        rows = [_make_search_row("d1")]
        result, _ = self._run(rows, top_k=5)
        assert set(result[0].keys()) == {"location", "event", "source_type", "chunk_text", "similarity"}

    def test_source_type_filter_adds_where_clause(self):
        _, cursor = self._run([], top_k=5, source_type="alert")
        sql = cursor.execute.call_args[0][0]
        assert "source_type" in sql.lower() and "WHERE" in sql.upper()

    def test_source_type_filter_passed_as_param(self):
        _, cursor = self._run([], top_k=5, source_type="forecast")
        params = cursor.execute.call_args[0][1]
        assert "forecast" in params

    def test_no_filter_query_does_not_contain_where(self):
        _, cursor = self._run([], top_k=5, source_type=None)
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql.upper()

    def test_embedding_passed_as_vector_string(self):
        _, cursor = self._run([], top_k=1)
        params = cursor.execute.call_args[0][1]
        # Embedding string should look like a Python list repr
        vec_str = params[0]
        assert vec_str.startswith("[") and vec_str.endswith("]")
