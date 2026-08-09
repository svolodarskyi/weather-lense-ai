"""
Sprint 8 tests — llm.py (build_context, chat).

The OpenAI client and Databricks credentials are mocked.
No real LLM calls are made.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm


_SOURCES = [
    {
        "location": "Chicago, IL",
        "event": "Flood Watch",
        "source_type": "alert",
        "chunk_text": "Heavy rain expected through Monday.",
        "similarity": 0.88,
    },
    {
        "location": "Miami, FL",
        "event": "Saturday",
        "source_type": "forecast",
        "chunk_text": "Partly cloudy with a chance of thunderstorms.",
        "similarity": 0.72,
    },
]


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_empty_sources_returns_empty_string(self):
        assert llm.build_context([]) == ""

    def test_contains_location(self):
        ctx = llm.build_context(_SOURCES)
        assert "Chicago, IL" in ctx
        assert "Miami, FL" in ctx

    def test_contains_event(self):
        ctx = llm.build_context(_SOURCES)
        assert "Flood Watch" in ctx
        assert "Saturday" in ctx

    def test_contains_chunk_text(self):
        ctx = llm.build_context(_SOURCES)
        assert "Heavy rain expected" in ctx
        assert "Partly cloudy" in ctx

    def test_contains_source_type_uppercased(self):
        ctx = llm.build_context(_SOURCES)
        assert "ALERT" in ctx
        assert "FORECAST" in ctx

    def test_numbered_entries(self):
        ctx = llm.build_context(_SOURCES)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_single_source(self):
        ctx = llm.build_context([_SOURCES[0]])
        assert "[1]" in ctx
        assert "[2]" not in ctx


# ---------------------------------------------------------------------------
# chat — client config guard
# ---------------------------------------------------------------------------

class TestChatConfigGuard:
    def test_raises_without_host(self):
        with patch.dict(os.environ, {"DATABRICKS_HOST": "", "DATABRICKS_TOKEN": "tok"}):
            with pytest.raises(ValueError, match="DATABRICKS_HOST"):
                llm.chat("question", _SOURCES)

    def test_raises_without_token(self):
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://host", "DATABRICKS_TOKEN": ""}):
            with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
                llm.chat("question", _SOURCES)


# ---------------------------------------------------------------------------
# chat — LLM call
# ---------------------------------------------------------------------------

class TestChat:
    def _mock_client(self, answer="Rain likely."):
        mock_choice = MagicMock()
        mock_choice.message.content = answer
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    def test_returns_answer_string(self):
        mc = self._mock_client("Rain likely.")
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            result = llm.chat("Will it rain?", _SOURCES)
        assert result == "Rain likely."

    def test_calls_correct_model(self):
        mc = self._mock_client()
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            llm.chat("question", _SOURCES)
        kwargs = mc.chat.completions.create.call_args[1]
        assert kwargs["model"] == llm.MODEL

    def test_question_in_user_message(self):
        mc = self._mock_client()
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            llm.chat("Will it flood?", _SOURCES)
        messages = mc.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Will it flood?" in user_msg["content"]

    def test_context_in_user_message(self):
        mc = self._mock_client()
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            llm.chat("question", _SOURCES)
        messages = mc.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Heavy rain expected" in user_msg["content"]

    def test_system_prompt_present(self):
        mc = self._mock_client()
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            llm.chat("question", _SOURCES)
        messages = mc.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert len(system_msg["content"]) > 0

    def test_answer_stripped(self):
        mc = self._mock_client("  Rain likely.  ")
        with patch.dict(os.environ, {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}), \
             patch("llm.OpenAI", return_value=mc):
            result = llm.chat("question", _SOURCES)
        assert result == "Rain likely."
