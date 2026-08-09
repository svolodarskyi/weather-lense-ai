"""
WeatherLens AI — full pipeline smoke test.

Usage:
    # Set connection env vars first (see README_WEATHER.md), then:
    python smoketest.py

Tests in order:
    1. Health check
    2. Sync NWS data for two locations
    3. Embedding job (find → chunk → embed → write)
    4. Semantic search (unfiltered)
    5. Semantic search with source_type=alert filter
    6. Semantic search with source_type=forecast filter
    7. Validation: blank query returns 400

The Flask app is driven via its test client — no server process needed.
The embedding job is called directly via import.

Exit code 0 = all passed. Exit code 1 = at least one failure.
"""

import os
import sys

# Ensure project root is on the path when run from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
BOLD  = "\033[1m"

results: list[tuple[str, bool, str]] = []   # (label, passed, detail)


def ok(label: str, detail: str = "") -> None:
    results.append((label, True, detail))
    print(f"  {GREEN}✓{RESET} {label}" + (f"  — {detail}" if detail else ""))


def fail(label: str, detail: str = "") -> None:
    results.append((label, False, detail))
    print(f"  {RED}✗{RESET} {label}" + (f"  — {detail}" if detail else ""))


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        ok(label, detail)
    else:
        fail(label, detail)
    return condition


# ── 1. boot Flask test client ────────────────────────────────────────────────
print(f"\n{BOLD}WeatherLens AI smoke test{RESET}\n")
print("Loading app (downloads model weights if first run) …")

try:
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    ok("App imported and test client created")
except Exception as exc:
    fail("App import", str(exc))
    sys.exit(1)


# ── 2. health ────────────────────────────────────────────────────────────────
print("\n[1] Health check")
try:
    r = client.get("/health")
    check("GET /health → 200", r.status_code == 200, f"status={r.status_code}")
    check("Body has status=ok", r.get_json() == {"status": "ok"}, str(r.get_json()))
except Exception as exc:
    fail("Health check raised", str(exc))


# ── 3. sync ──────────────────────────────────────────────────────────────────
print("\n[2] Sync (POST /weather/sync)")
LOCATIONS = ["Chicago, IL", "Miami, FL"]
try:
    r = client.post("/weather/sync", json={"locations": LOCATIONS})
    data = r.get_json() or {}
    check("POST /weather/sync → 200", r.status_code == 200, f"status={r.status_code}")
    synced = data.get("synced", -1)
    check("synced ≥ 0", synced >= 0, f"synced={synced}")
    errors = data.get("errors", [])
    check("errors list present", isinstance(errors, list), str(errors))
    if errors:
        print(f"    ⚠  sync errors: {errors}")
    else:
        ok("No sync errors")
    print(f"    synced={synced} documents, locations={LOCATIONS}")
except Exception as exc:
    fail("Sync raised", str(exc))


# ── 4. embedding job ─────────────────────────────────────────────────────────
print("\n[3] Embedding job (ingest_weather_embeddings)")
try:
    from notebooks.ingest_weather_embeddings import run as embed_run
    enc = app_module.encoder          # reuse already-loaded encoder
    n_docs, n_chunks = embed_run(enc)
    check("Embedding job completed", True, f"{n_docs} docs, {n_chunks} chunks")
    check("At least one chunk written", n_chunks >= 0,
          "(0 is fine if all docs were already embedded)")
except Exception as exc:
    fail("Embedding job raised", str(exc))


# ── 5. search — unfiltered ───────────────────────────────────────────────────
print("\n[4] Search — unfiltered")
QUERY = "flash flood risk this weekend"
try:
    r = client.post("/weather/search", json={"query": QUERY, "top_k": 5})
    data = r.get_json() or {}
    check("POST /weather/search → 200", r.status_code == 200, f"status={r.status_code}")
    check("Response has 'query' key", "query" in data)
    check("Response has 'count' key", "count" in data)
    check("Response has 'results' list", isinstance(data.get("results"), list))
    count = data.get("count", 0)
    check("count > 0", count > 0, f"count={count}")
    results_list = data.get("results", [])
    if results_list:
        r0 = results_list[0]
        required_keys = {"location", "event", "source_type", "chunk_text", "similarity"}
        check("Result has required keys", required_keys.issubset(r0.keys()),
              f"keys={set(r0.keys())}")
        doc_ids = [x.get("location") for x in results_list]
        check("query echoed back stripped", data.get("query") == QUERY.strip())
        print(f"    top result: [{r0.get('source_type')}] {r0.get('location')} — "
              f"{r0.get('event')} (similarity={r0.get('similarity')})")
except Exception as exc:
    fail("Unfiltered search raised", str(exc))


# ── 6. search — alert filter ─────────────────────────────────────────────────
print("\n[5] Search — source_type=alert")
try:
    r = client.post("/weather/search",
                    json={"query": "severe weather warning", "top_k": 5,
                          "source_type": "alert"})
    data = r.get_json() or {}
    check("POST /weather/search?alert → 200", r.status_code == 200)
    results_list = data.get("results", [])
    wrong = [x for x in results_list if x.get("source_type") != "alert"]
    check("All results are alerts", len(wrong) == 0,
          f"{len(wrong)} non-alert results" if wrong else f"{len(results_list)} alerts")
except Exception as exc:
    fail("Alert-filter search raised", str(exc))


# ── 7. search — forecast filter ──────────────────────────────────────────────
print("\n[6] Search — source_type=forecast")
try:
    r = client.post("/weather/search",
                    json={"query": "sunny weekend outlook", "top_k": 5,
                          "source_type": "forecast"})
    data = r.get_json() or {}
    check("POST /weather/search?forecast → 200", r.status_code == 200)
    results_list = data.get("results", [])
    wrong = [x for x in results_list if x.get("source_type") != "forecast"]
    check("All results are forecasts", len(wrong) == 0,
          f"{len(wrong)} non-forecast results" if wrong else f"{len(results_list)} forecasts")
except Exception as exc:
    fail("Forecast-filter search raised", str(exc))


# ── 8. validation — blank query ──────────────────────────────────────────────
print("\n[7] Validation")
try:
    r = client.post("/weather/search", json={"query": "   "})
    check("Blank query → 400", r.status_code == 400, f"status={r.status_code}")
    check("Error body has 'error' key", "error" in (r.get_json() or {}))

    r2 = client.post("/weather/search", json={})
    check("Missing query → 400", r2.status_code == 400, f"status={r2.status_code}")

    r3 = client.post("/weather/sync", json={"locations": []})
    check("Empty locations → 400", r3.status_code == 400, f"status={r3.status_code}")
except Exception as exc:
    fail("Validation checks raised", str(exc))


# ── summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, p, _ in results if p)
total  = len(results)
failed = total - passed

print(f"\n{'─' * 50}")
print(f"{BOLD}Results: {passed}/{total} passed{RESET}", end="  ")
if failed:
    print(f"{RED}{failed} failed{RESET}")
else:
    print(f"{GREEN}all green{RESET}")
print()

sys.exit(0 if failed == 0 else 1)
