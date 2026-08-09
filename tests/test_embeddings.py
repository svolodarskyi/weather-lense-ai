"""
Sprint 4 tests — embeddings.py and the embedding job (ingest_weather_embeddings.py).

sentence_transformers is injected via sys.modules so these tests run without
the real model installed (no 800 MB download required).  Live tests that
actually encode text are marked @pytest.mark.live.
"""

import importlib.util
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings import chunk_text, MODEL_NAME, DIMS, CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Fixture: fake sentence_transformers module
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_st(monkeypatch):
    """Inject a mock sentence_transformers into sys.modules.

    Returns a factory: `fake_st(n_texts)` → MagicMock Encoder instance whose
    .encode() returns an (n_texts × 384) zero array.
    """
    mock_module = MagicMock()

    def _make_instance(n=1):
        inst = MagicMock()
        inst.encode.side_effect = lambda texts, **_kw: np.zeros((len(texts), 384))
        mock_module.SentenceTransformer.return_value = inst
        return inst

    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_module)
    return _make_instance


# ---------------------------------------------------------------------------
# Helper to load the notebook as a module
# ---------------------------------------------------------------------------

def _load_notebook():
    path = os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "ingest_weather_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("ingest_weather_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# chunk_text — pure Python, no mocking needed
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Flood watch for Cook County. Heavy rain expected."
        chunks = chunk_text(text, chunk_size=800)
        assert chunks == [text]

    def test_text_exactly_chunk_size_is_one_chunk(self):
        text = "A" * 800
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) == 1

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n  ") == []

    def test_long_text_produces_multiple_chunks(self):
        text = "Weather alert. " * 100   # ~1500 chars
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) >= 2

    def test_no_mid_word_splits(self):
        # All words are 12 chars; no chunk should contain a partial word.
        word = "FLOODWATCH  "   # 12 chars including trailing spaces
        text = word * 100
        chunks = chunk_text(text, chunk_size=800)
        for chunk in chunks:
            for part in chunk.split():
                # Each token must be a complete repetition of "FLOODWATCH"
                assert part == "FLOODWATCH", f"Partial word in chunk: {part!r}"

    def test_paragraph_break_preferred_over_word_break(self):
        # Place a paragraph break at 50% mark; a word break exists closer to end.
        # The chunker should take the paragraph break.
        para_break_pos = 410  # > half (400)
        text = "A" * para_break_pos + "\n\n" + "B" * 800
        chunks = chunk_text(text, chunk_size=800)
        # First chunk must end before the paragraph break characters
        assert "B" not in chunks[0]

    def test_sentence_break_preferred_over_word_break(self):
        # Sentence break at 450; word break at 780.
        text = "A" * 450 + ". " + "B" * 400 + " " + "C" * 200
        chunks = chunk_text(text, chunk_size=800)
        first = chunks[0]
        # Sentence break ends the first chunk; the word at 780 is NOT the break.
        assert first.endswith(".") or first.endswith("A")

    def test_50pct_threshold_prevents_early_break(self):
        # Period at 200 chars (25% of 800) must NOT be the break point.
        text = "A" * 200 + ". " + "B" * 800
        chunks = chunk_text(text, chunk_size=800)
        # First chunk must be longer than 202 chars (the early break would give ~202)
        assert len(chunks[0]) > 400

    def test_consecutive_chunks_overlap(self):
        # Chunk N+1 should start before chunk N ends (by roughly `overlap` chars).
        text = "word " * 400   # ~2000 chars
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        assert len(chunks) >= 2
        # Find where chunk 0 content ends and check chunk 1 starts before that.
        # Simplified: first word of chunk[1] should appear in the tail of chunk[0].
        first_word_of_chunk1 = chunks[1].split()[0]
        assert first_word_of_chunk1 in chunks[0]

    def test_all_text_covered(self):
        # No text should be skipped entirely.
        unique_words = [f"WORD{i:04d}" for i in range(200)]
        text = " ".join(unique_words)
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        combined = " ".join(chunks)
        for word in unique_words:
            assert word in combined, f"{word} missing from all chunks"

    def test_single_very_long_word_force_breaks(self):
        # A single word longer than chunk_size gets force-broken.
        text = "X" * 2000
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) >= 2
        total = sum(len(c) for c in chunks)
        assert total >= 1900  # minimal loss from overlap trimming


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_unknown_model_raises_value_error(self, fake_st):
        fake_st()
        from embeddings import Encoder
        with pytest.raises(ValueError, match="Unknown model"):
            Encoder("gpt-4-turbo-not-an-embedding-model")

    def test_known_model_instantiates_without_error(self, fake_st):
        fake_st()
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        assert enc is not None

    def test_model_name_property(self, fake_st):
        fake_st()
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        assert enc.model_name == MODEL_NAME

    def test_dims_property(self, fake_st):
        fake_st()
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        assert enc.dims == 384

    def test_encode_returns_list_of_lists(self, fake_st):
        fake_st(2)
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        result = enc.encode(["text one", "text two"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_encode_vector_has_correct_dims(self, fake_st):
        fake_st(1)
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        result = enc.encode(["some weather text"])
        assert len(result[0]) == 384

    def test_encode_passes_normalize_embeddings(self, fake_st):
        inst = fake_st(1)
        from embeddings import Encoder
        enc = Encoder(MODEL_NAME)
        enc.encode(["text"])
        _, kwargs = inst.encode.call_args
        assert kwargs.get("normalize_embeddings") is True


# ---------------------------------------------------------------------------
# Embedding job (ingest_weather_embeddings.py)
# ---------------------------------------------------------------------------

def _make_conn_mock():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def _make_conn_ctx(conn):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestFindUnembedded:
    def test_executes_anti_join_query(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        cursor.fetchall.return_value = []
        nb.find_unembedded(conn, MODEL_NAME)
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "LEFT JOIN" in sql
        assert "e.id IS NULL" in sql

    def test_passes_model_name_as_param(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        cursor.fetchall.return_value = []
        nb.find_unembedded(conn, MODEL_NAME)
        params = cursor.execute.call_args[0][1]
        assert MODEL_NAME in params

    def test_returns_fetchall_result(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        expected = [("id1", "text1", "hash1"), ("id2", "text2", "hash2")]
        cursor.fetchall.return_value = expected
        result = nb.find_unembedded(conn, MODEL_NAME)
        assert result == expected


class TestWriteDocument:
    def _make_encoder(self, n_chunks):
        enc = MagicMock()
        enc.model_name = MODEL_NAME
        enc.encode.return_value = [[0.1] * 384] * n_chunks
        return enc

    def test_returns_zero_for_empty_text(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, _ = _make_conn_mock()
        enc = self._make_encoder(0)
        result = nb.write_document(conn, enc, "doc1", "", "hash1")
        assert result == 0

    def test_returns_chunk_count(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        enc = self._make_encoder(3)
        text = "Weather alert. " * 80   # ~1200 chars → 2-3 chunks
        with patch.object(nb, "execute_values"):
            result = nb.write_document(conn, enc, "doc1", text, "hash1")
        assert result > 0

    def test_deletes_before_inserting(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        enc = self._make_encoder(2)
        with patch.object(nb, "execute_values") as mock_ev:
            nb.write_document(conn, enc, "doc1", "word " * 200, "hash1")
        # DELETE must be called before execute_values
        delete_call_idx = next(
            i for i, c in enumerate(cursor.execute.call_args_list)
            if "DELETE" in c[0][0]
        )
        assert delete_call_idx == 0

    def test_uses_vector_cast_template(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        enc = self._make_encoder(2)
        with patch.object(nb, "execute_values") as mock_ev:
            nb.write_document(conn, enc, "doc1", "word " * 200, "hash1")
        template = mock_ev.call_args[1].get("template") or mock_ev.call_args[0][3]
        assert "::vector" in template

    def test_row_id_is_doc_chunk_composite(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        enc = self._make_encoder(2)
        captured = {}
        def capture(cur, sql, rows, **kw):
            captured["rows"] = rows
        with patch.object(nb, "execute_values", side_effect=capture):
            nb.write_document(conn, enc, "my-doc-id", "word " * 200, "hash1")
        first_row = captured["rows"][0]
        assert first_row[0] == "my-doc-id::0"
        assert first_row[1] == "my-doc-id"
        assert first_row[2] == 0   # chunk_index

    def test_content_hash_stored_on_row(self, fake_st):
        fake_st()
        nb = _load_notebook()
        conn, cursor = _make_conn_mock()
        enc = self._make_encoder(1)
        captured = {}
        def capture(cur, sql, rows, **kw):
            captured["rows"] = rows
        with patch.object(nb, "execute_values", side_effect=capture):
            nb.write_document(conn, enc, "doc1", "short text for one chunk", "DEADBEEF")
        row = captured["rows"][0]
        assert "DEADBEEF" in row   # content_hash is in the row tuple


class TestRunJob:
    def _setup(self, fake_st, docs):
        fake_st()
        nb = _load_notebook()
        enc = MagicMock()
        enc.model_name = MODEL_NAME

        conn, cursor = _make_conn_mock()
        cursor.fetchall.return_value = docs

        conn_ctx = _make_conn_ctx(conn)
        return nb, enc, conn, cursor, conn_ctx

    def test_calls_apply_schema(self, fake_st):
        nb, enc, conn, cursor, ctx = self._setup(fake_st, [])
        with patch.object(nb, "apply_schema") as mock_apply, \
             patch.object(nb, "get_lakebase_connection", return_value=ctx), \
             patch.object(nb, "write_document", return_value=0):
            nb.run(enc)
        mock_apply.assert_called_once()

    def test_returns_zero_zero_when_no_docs(self, fake_st):
        nb, enc, conn, cursor, ctx = self._setup(fake_st, [])
        with patch.object(nb, "apply_schema"), \
             patch.object(nb, "get_lakebase_connection", return_value=ctx), \
             patch.object(nb, "write_document", return_value=0):
            result = nb.run(enc)
        assert result == (0, 0)

    def test_calls_write_document_for_each_doc(self, fake_st):
        docs = [
            ("id1", "text one", "hash1"),
            ("id2", "text two", "hash2"),
        ]
        nb, enc, conn, cursor, ctx = self._setup(fake_st, docs)
        with patch.object(nb, "apply_schema"), \
             patch.object(nb, "get_lakebase_connection", return_value=ctx), \
             patch.object(nb, "write_document", return_value=2) as mock_write:
            nb.run(enc)
        assert mock_write.call_count == 2

    def test_returns_correct_totals(self, fake_st):
        docs = [("id1", "text", "h1"), ("id2", "text", "h2"), ("id3", "text", "h3")]
        nb, enc, conn, cursor, ctx = self._setup(fake_st, docs)
        with patch.object(nb, "apply_schema"), \
             patch.object(nb, "get_lakebase_connection", return_value=ctx), \
             patch.object(nb, "write_document", return_value=3):
            n_docs, n_chunks = nb.run(enc)
        assert n_docs == 3
        assert n_chunks == 9  # 3 docs × 3 chunks each


# ---------------------------------------------------------------------------
# Live smoke test (deselect with -m "not live")
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_embedding_job_live():
    """Run the full job against real Lakebase and verify embeddings exist."""
    from notebooks.ingest_weather_embeddings import run
    from lakebase import get_lakebase_connection
    from embeddings import Encoder

    enc = Encoder(MODEL_NAME)
    n_docs, n_chunks = run(enc)
    assert n_docs >= 0   # may be 0 if already up to date

    with get_lakebase_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM weather_embeddings")
            count = cur.fetchone()[0]
    assert count > 0
