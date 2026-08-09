"""
WeatherLens AI — National Weather Service harvest client.

Pulls two document types from api.weather.gov:
  - Active alerts  (warnings, watches, advisories) via /alerts/active
  - Location forecasts (7-day narrative) via /gridpoints/{office}/{x},{y}/forecast

City/state names are resolved to coordinates via Open-Meteo (no key needed).
Both document types land in a shared dict schema, filterable by source_type.
"""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

NWS_URL  = os.getenv("NWS_BASE_URL", "https://api.weather.gov")
GEO_URL  = os.getenv("GEO_URL", "https://geocoding-api.open-meteo.com/v1/search")
NWS_AGENT = os.getenv("NWS_USER_AGENT", "")
MAX_RPS  = float(os.getenv("NWS_MAX_RPS", "4"))

ALERT    = "alert"
FORECAST = "forecast"

# NWS returns HTTP 301 for /points when coordinates have more than 4 decimal
# places. Rounding avoids the redirect with negligible precision loss — the
# NWS forecast grid is 2.5 km, far coarser than a few centimetres.
_COORD_DP = 4
_COORD_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$")

_STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME",
    "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR", "GUAM": "GU", "AMERICAN SAMOA": "AS",
    "VIRGIN ISLANDS": "VI", "U.S. VIRGIN ISLANDS": "VI",
    "NORTHERN MARIANA ISLANDS": "MP",
}
_STATE_CODES = set(_STATE_NAMES.values())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HarvestError(Exception):
    """Base class for client failures."""

class BadLocation(HarvestError):
    """Location string could not be parsed or geocoded."""

class GeocodeFailed(HarvestError):
    """Open-Meteo returned no match for this city + state."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Location:
    """A parsed location before any network call."""
    raw: str
    kind: str           # "coords" or "city"
    lat: float | None = None
    lon: float | None = None
    city: str | None = None
    state: str | None = None

    @property
    def label(self) -> str:
        if self.kind == "coords":
            return f"{self.lat},{self.lon}"
        return f"{self.city}, {self.state}"


@dataclass
class NWSGrid:
    """An NWS forecast office and grid square for a resolved location."""
    label: str
    lat: float
    lon: float
    state: str
    office: str
    gx: int
    gy: int


@dataclass
class WeatherDoc:
    """A normalised weather narrative ready to store and embed."""
    id: str
    location: str
    latitude: float | None
    longitude: float | None
    source_type: str
    event: str | None
    headline: str | None
    narrative_text: str
    issued_at: str | None
    effective_at: str | None
    expires_at: str | None
    severity: str | None
    payload: dict
    content_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _text_hash(self.narrative_text)

    def as_row(self) -> dict:
        """Return a column-keyed dict matching the weather_documents schema."""
        return {
            "id": self.id,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_type": self.source_type,
            "event": self.event,
            "headline": self.headline,
            "narrative_text": self.narrative_text,
            "issued_at": self.issued_at,
            "effective_at": self.effective_at,
            "expires_at": self.expires_at,
            "severity": self.severity,
            "payload": self.payload,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """SHA-256 fingerprint of the text that will be embedded.

    Stored on both weather_documents and weather_embeddings so the embedding
    job can detect in-place amendments — NWS re-issues alerts under the same
    id with updated wording without changing the id.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def _to_state_code(name: str) -> str | None:
    """Accept 'IL' or 'Illinois', return 'IL'. Returns None if unrecognised."""
    upper = name.strip().upper()
    return upper if upper in _STATE_CODES else _STATE_NAMES.get(upper)


def _parse(raw: str) -> Location:
    """Parse a location string into a Location.

    Accepts 'lat,lon' coordinate pairs and 'City, ST' or 'City, State'.
    Rejects a bare city name — it matches places in 30+ states and guessing
    one silently returns confident weather for the wrong location.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise BadLocation(f"Location must be a non-empty string, got {raw!r}")

    text = raw.strip()

    m = _COORD_RE.match(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not -90 <= lat <= 90:
            raise BadLocation(f"Latitude {lat} is out of range (-90 to 90)")
        if not -180 <= lon <= 180:
            raise BadLocation(f"Longitude {lon} is out of range (-180 to 180)")
        return Location(raw=text, kind="coords",
                        lat=round(lat, _COORD_DP), lon=round(lon, _COORD_DP))

    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 2 and parts[0]:
        code = _to_state_code(parts[1])
        if code:
            return Location(raw=text, kind="city", city=parts[0], state=code)

    raise BadLocation(f"Cannot parse {raw!r} — expected 'City, ST' or 'lat,lon'")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WeatherClient:
    """Fetches and normalises weather documents from api.weather.gov."""

    def __init__(self, user_agent: str | None = None, rps: float = MAX_RPS) -> None:
        agent = (user_agent or NWS_AGENT or "").strip()
        if not agent:
            raise ValueError(
                "NWS requires a User-Agent header identifying the app and a "
                "contact address — e.g. '(WeatherLens AI, you@example.com)'.\n"
                "Set NWS_USER_AGENT or pass user_agent= to WeatherClient."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": agent,
            "Accept": "application/geo+json",
        })
        self._min_gap = 1.0 / rps if rps > 0 else 0.0
        self._last_req = 0.0
        self.errors: list[str] = []

    # -- HTTP ---------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> Any:
        """Rate-limited GET with up to 3 attempts on transient failures.

        4xx errors are raised immediately — retrying a bad request cannot
        change the outcome. 5xx and network failures get exponential backoff.
        """
        gap = self._min_gap - (time.monotonic() - self._last_req)
        if gap > 0:
            time.sleep(gap)

        for attempt in range(1, 4):
            self._last_req = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()

            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    # Surface the NWS problem-detail message when available.
                    try:
                        body = exc.response.json()
                        extra = body.get("detail") or body.get("title") or ""
                        msg = f"{exc} — {extra}" if extra else str(exc)
                    except Exception:
                        msg = str(exc)
                    raise requests.HTTPError(msg, response=exc.response) from None
                if attempt == 3:
                    raise

            except requests.RequestException:
                if attempt == 3:
                    raise

            delay = 0.5 * (2 ** (attempt - 1))
            log.warning("Request failed (attempt %d/3), retrying in %.1fs: %s", attempt, delay, url)
            time.sleep(delay)

    # -- Resolution ---------------------------------------------------------

    def _geocode(self, city: str, state: str) -> tuple[float, float]:
        """Resolve a city + state to lat/lon via Open-Meteo."""
        data = self._get(GEO_URL, params={
            "name": city, "count": 10,
            "countryCode": "US", "language": "en", "format": "json",
        })
        for result in data.get("results") or []:
            if result.get("country_code", "").upper() != "US":
                continue
            if _to_state_code(result.get("admin1") or "") != state:
                continue
            return (
                round(float(result["latitude"]), _COORD_DP),
                round(float(result["longitude"]), _COORD_DP),
            )
        raise GeocodeFailed(f"No geocoding result for {city!r} in {state!r}")

    def _grid(self, location: str) -> NWSGrid:
        """Resolve a location string to an NWS office + grid square."""
        parsed = _parse(location)

        if parsed.kind == "coords":
            lat, lon = parsed.lat, parsed.lon
        else:
            lat, lon = self._geocode(parsed.city, parsed.state)

        data = self._get(f"{NWS_URL}/points/{lat},{lon}")
        props = data["properties"]
        rel = (props.get("relativeLocation") or {}).get("properties") or {}

        city  = rel.get("city") or (parsed.city if parsed.kind == "city" else None)
        # Read state back from NWS, not from the input. A point just across a
        # state line belongs to the state NWS assigns it.
        state = (rel.get("state") or parsed.state or "").upper()
        label = f"{city}, {state}" if city and state else f"{lat},{lon}"

        return NWSGrid(
            label=label, lat=lat, lon=lon, state=state,
            office=props["gridId"],
            gx=int(props["gridX"]),
            gy=int(props["gridY"]),
        )

    # -- Fetching -----------------------------------------------------------

    def _alerts(self, grid: NWSGrid) -> list[dict]:
        # No `limit` param — /alerts/active rejects it with HTTP 400 and the
        # error body has no 'features' key, so code that assumes the key would
        # silently report zero alerts for a state that has dozens.
        data = self._get(f"{NWS_URL}/alerts/active", params={"area": grid.state})
        return data.get("features") or []

    def _forecast(self, grid: NWSGrid) -> dict:
        return self._get(
            f"{NWS_URL}/gridpoints/{grid.office}/{grid.gx},{grid.gy}/forecast"
        )

    # -- Normalisation ------------------------------------------------------

    def _to_alert_docs(self, grid: NWSGrid, features: list) -> list[WeatherDoc]:
        docs = []
        for feature in features:
            props = feature.get("properties") or {}

            desc  = (props.get("description") or "").strip()
            instr = (props.get("instruction") or "").strip()
            # Both fields answer different questions: 'description' covers
            # what/where/when, 'instruction' covers what to do about it.
            # Joining them means a query like "should I evacuate" can match
            # the instruction even when the description doesn't mention it.
            narrative = "\n\n".join(p for p in [desc, instr] if p)
            if not narrative:
                continue

            alert_id = props.get("id") or feature.get("id")
            if not alert_id:
                continue

            sent = props.get("sent")
            docs.append(WeatherDoc(
                id=str(alert_id),  # NWS URN is globally unique — no hashing needed
                location=props.get("areaDesc") or grid.label,
                latitude=grid.lat, longitude=grid.lon,
                source_type=ALERT,
                event=props.get("event"),
                headline=props.get("headline") or props.get("event"),
                narrative_text=narrative,
                issued_at=sent,
                effective_at=props.get("effective") or props.get("onset") or sent,
                expires_at=props.get("expires") or props.get("ends"),
                severity=props.get("severity"),
                payload=feature,
            ))
        return docs

    def _to_forecast_docs(self, grid: NWSGrid, data: dict) -> list[WeatherDoc]:
        props = (data or {}).get("properties") or {}
        generated = props.get("generatedAt") or props.get("updateTime")
        docs = []
        for period in props.get("periods") or []:
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative:
                continue

            start, end = period.get("startTime"), period.get("endTime")
            # ID is keyed on the time window (startTime + endTime), not on the
            # period name. NWS renames periods as the day advances — "This
            # Afternoon" becomes "Today" in a later issuance. A name-based ID
            # would store the same forecast period twice under different keys.
            window_id = hashlib.sha256(
                f"{grid.label}|{start}|{end}".encode()
            ).hexdigest()[:16]

            docs.append(WeatherDoc(
                id=f"forecast:{grid.office}/{grid.gx},{grid.gy}:{window_id}",
                location=grid.label,
                latitude=grid.lat, longitude=grid.lon,
                source_type=FORECAST,
                event=period.get("name"),
                headline=period.get("shortForecast"),
                narrative_text=narrative,
                issued_at=generated,
                effective_at=start,
                expires_at=end,
                severity=None,
                payload=period,
            ))
        return docs

    # -- Public API ---------------------------------------------------------

    def fetch(self, locations: list[str], limit: int = 50) -> list[WeatherDoc]:
        """Fetch alerts and forecasts for a list of locations.

        Documents are deduplicated by id — neighbouring locations share a
        forecast office and often return the same statewide alerts. One bad
        location is recorded in self.errors and does not abort the rest.
        """
        self.errors = []
        seen: set[str] = set()
        results: list[WeatherDoc] = []

        for loc in locations:
            try:
                grid  = self._grid(loc)
                batch = (
                    self._to_alert_docs(grid, self._alerts(grid))
                    + self._to_forecast_docs(grid, self._forecast(grid))
                )
                for doc in batch:
                    if doc.id not in seen:
                        seen.add(doc.id)
                        results.append(doc)
                        if limit and len(results) >= limit:
                            return results

            except Exception as exc:  # noqa: BLE001
                msg = f"{loc}: {exc}"
                log.warning("Skipping location — %s", msg)
                self.errors.append(msg)

        return results
