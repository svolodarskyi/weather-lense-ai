"""
WeatherLens AI — LLM client for RAG chat.

Uses the Databricks Foundation Models OpenAI-compatible endpoint.
Model: databricks-meta-llama-3-3-70b-instruct

Auth: DATABRICKS_TOKEN if set (local dev); otherwise the SDK credential chain
(M2M OAuth via DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET, auto-injected
by Databricks Apps).
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


def _get_m2m_token(host: str) -> str:
    """Exchange DATABRICKS_CLIENT_ID/SECRET for an M2M OAuth Bearer token."""
    import base64
    import json
    import urllib.parse
    import urllib.request

    client_id = os.getenv("DATABRICKS_CLIENT_ID", "")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        return ""
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode(
        {"grant_type": "client_credentials", "scope": "all-apis"}
    ).encode()
    req = urllib.request.Request(
        f"{host}/oidc/v1/token",
        data=body,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["access_token"]


def _build_client() -> OpenAI:
    host  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")

    # Ensure host carries a scheme (DATABRICKS_HOST in Apps may omit https://)
    if host and not host.startswith(("http://", "https://")):
        host = f"https://{host}"

    # Fall back to SDK config for host when env var is absent
    if not host:
        from databricks.sdk.config import Config
        host = (Config().host or "").rstrip("/")

    # No static PAT → exchange service principal credentials for an OAuth token
    if not token:
        token = _get_m2m_token(host)

    if not host or not token:
        raise ValueError(
            "Could not resolve Databricks credentials. "
            "Set DATABRICKS_TOKEN, or deploy as a Databricks App with "
            "DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET injected."
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
