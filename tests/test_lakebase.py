"""
Sprint 2 tests — lakebase.py

All psycopg2 calls are mocked; no real database required.
Live tests (requiring actual Lakebase) are marked @pytest.mark.live.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path so "import lakebase" resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lakebase import LakebaseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn():
    """Return (conn_mock, cursor_mock) wired for context-manager use."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# ---------------------------------------------------------------------------
# get_lakebase_connection()
# ---------------------------------------------------------------------------

class TestGetLakbaseConnection:
    def test_commits_on_clean_exit(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "postgresql://u:p@host/db")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn):
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        conn.close.assert_called_once()

    def test_rolls_back_on_exception(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "postgresql://u:p@host/db")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn):
            from lakebase import get_lakebase_connection
            with pytest.raises(RuntimeError):
                with get_lakebase_connection():
                    raise RuntimeError("simulated failure")
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()
        conn.close.assert_called_once()

    def test_always_closes_even_after_rollback(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "postgresql://u:p@host/db")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn):
            from lakebase import get_lakebase_connection
            with pytest.raises(ValueError):
                with get_lakebase_connection():
                    raise ValueError("boom")
        conn.close.assert_called_once()

    def test_strips_leading_trailing_whitespace_from_url(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "  postgresql://u:p@host/db  ")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn) as mock_connect:
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        dsn = mock_connect.call_args[0][0]
        assert " " not in dsn

    def test_strips_embedded_newline_from_url(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "postgresql://u:p@host/db\n")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn) as mock_connect:
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        dsn = mock_connect.call_args[0][0]
        assert "\n" not in dsn

    def test_falls_back_to_pg_env_vars(self, monkeypatch):
        monkeypatch.delenv("LAKEBASE_CONNECTION_URL", raising=False)
        monkeypatch.setenv("PGHOST", "ep-test.cloud.databricks.com")
        monkeypatch.setenv("PGUSER", "user@example.com")
        monkeypatch.setenv("PGPASSWORD", "oauth-token-abc")
        monkeypatch.setenv("PGDATABASE", "databricks_postgres")
        monkeypatch.setenv("PGPORT", "5432")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn) as mock_connect:
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        dsn = mock_connect.call_args[0][0]
        assert "ep-test.cloud.databricks.com" in dsn
        assert "sslmode=require" in dsn

    def test_pg_vars_include_sslmode_require(self, monkeypatch):
        monkeypatch.delenv("LAKEBASE_CONNECTION_URL", raising=False)
        monkeypatch.setenv("PGHOST", "myhost")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn) as mock_connect:
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        assert "sslmode=require" in mock_connect.call_args[0][0]

    def test_raises_lakebase_error_when_no_config(self, monkeypatch):
        monkeypatch.delenv("LAKEBASE_CONNECTION_URL", raising=False)
        monkeypatch.delenv("PGHOST", raising=False)
        from lakebase import get_lakebase_connection
        with pytest.raises(LakebaseError, match="LAKEBASE_CONNECTION_URL"):
            with get_lakebase_connection():
                pass

    def test_blank_url_falls_through_to_pg_vars(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "   ")
        monkeypatch.setenv("PGHOST", "fallback-host")
        monkeypatch.setenv("PGUSER", "u")
        monkeypatch.setenv("PGPASSWORD", "p")
        conn, _ = _make_conn()
        with patch("psycopg2.connect", return_value=conn) as mock_connect:
            from lakebase import get_lakebase_connection
            with get_lakebase_connection():
                pass
        assert "fallback-host" in mock_connect.call_args[0][0]


# ---------------------------------------------------------------------------
# apply_schema()
# ---------------------------------------------------------------------------

class TestApplySchema:
    def _run(self, monkeypatch):
        monkeypatch.setenv("LAKEBASE_CONNECTION_URL", "postgresql://u:p@host/db")
        conn, cursor = _make_conn()
        with patch("psycopg2.connect", return_value=conn):
            from lakebase import apply_schema
            apply_schema()
        return conn, cursor

    def test_executes_four_statements(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        assert cursor.execute.call_count == 4

    def test_creates_vector_extension(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("CREATE EXTENSION" in s and "vector" in s for s in sqls)

    def test_creates_weather_documents_table(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("weather_documents" in s for s in sqls)

    def test_creates_weather_embeddings_table(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("weather_embeddings" in s and "VECTOR(384)" in s for s in sqls)

    def test_creates_hnsw_index(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("hnsw" in s and "vector_cosine_ops" in s for s in sqls)

    def test_all_statements_are_idempotent(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        for s in sqls:
            assert "IF NOT EXISTS" in s, f"Not idempotent: {s[:80]!r}"

    def test_extension_is_first_statement(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        first = cursor.execute.call_args_list[0][0][0]
        assert "EXTENSION" in first

    def test_embeddings_table_after_documents_table(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        docs_idx = next(i for i, s in enumerate(sqls) if "weather_documents" in s)
        emb_idx  = next(i for i, s in enumerate(sqls) if "weather_embeddings" in s and "VECTOR" in s)
        assert docs_idx < emb_idx

    def test_commits_after_all_ddl(self, monkeypatch):
        conn, cursor = self._run(monkeypatch)
        conn.commit.assert_called_once()
        assert cursor.execute.call_count == 4

    def test_documents_table_has_check_on_source_type(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        docs_sql = next(s for s in sqls if "weather_documents" in s)
        assert "alert" in docs_sql and "forecast" in docs_sql

    def test_embeddings_fk_has_on_delete_cascade(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        emb_sql = next(s for s in sqls if "weather_embeddings" in s and "VECTOR" in s)
        assert "ON DELETE CASCADE" in emb_sql

    def test_embeddings_table_has_unique_constraint(self, monkeypatch):
        _, cursor = self._run(monkeypatch)
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        emb_sql = next(s for s in sqls if "weather_embeddings" in s and "VECTOR" in s)
        assert "UNIQUE" in emb_sql and "chunk_index" in emb_sql


# ---------------------------------------------------------------------------
# Live smoke test (requires real Lakebase — deselect with -m "not live")
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_apply_schema_live():
    """Round-trip: connect to Lakebase, apply schema, verify tables exist."""
    from lakebase import apply_schema, get_lakebase_connection
    apply_schema()
    with get_lakebase_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN "
                "('weather_documents', 'weather_embeddings')"
            )
            names = {row[0] for row in cur.fetchall()}
    assert "weather_documents" in names
    assert "weather_embeddings" in names
