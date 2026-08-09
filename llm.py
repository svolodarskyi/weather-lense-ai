"""
WeatherLens AI — LLM client for RAG chat.

Uses the Databricks Foundation Models OpenAI-compatible endpoint.
Model: databricks-meta-llama-3-3-70b-instruct

Connection env vars (same pattern as lakebase.py):
  DATABRICKS_HOST   — workspace URL, e.g. https://dbc-xxx.cloud.databricks.com
  DATABRICKS_TOKEN  — personal access token or service principal secret

In a deployed Databricks App both vars are injected automatically.
"""

import os
from openai import OpenAI

MODEL = "databricks-meta-llama-3-3-70b-instruct"

_SYSTEM_PROMPT = """\
You are WeatherLens AI, a weather intelligence assistant.
Answer the user's question using ONLY the NWS weather context provided below.
Be concise (3-5 sentences). If the context does not contain enough information
to answer confidently, say so — do not invent weather data.
Do not mention "context" or "chunks" in your answer; respond naturally."""

_CONTEXT_TEMPLATE = """\
--- Weather context ---
{context}
--- End context ---

Question: {question}"""


def _build_client() -> OpenAI:
    host  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise ValueError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set for the chat endpoint."
        )
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


def build_context(search_results: list[dict]) -> str:
    """Format repository.search() results into a numbered context block."""
    parts = []
    for i, r in enumerate(search_results, 1):
        parts.append(
            f"[{i}] {r['source_type'].upper()} — {r['location']} — {r['event']}\n"
            f"{r['chunk_text']}"
        )
    return "\n\n".join(parts)


def chat(question: str, search_results: list[dict]) -> str:
    """Send question + retrieved NWS context to Llama, return the answer text."""
    client  = _build_client()
    context = build_context(search_results)
    user_msg = _CONTEXT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
