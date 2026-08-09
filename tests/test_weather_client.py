"""
Tests for weather_client.py — Sprint 1.

External boundaries mocked: requests.Session (HTTP), time.sleep (pacing).
Everything else runs for real: parsing, normalization, ID derivation,
content_hash computation, deduplication.
"""

import hashlib
import time
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from weather_client import (
    ALERT,
    FORECAST,
    BadLocation,
    GeocodeFailed,
    Location,
    NWSGrid,
    WeatherClient,
    WeatherDoc,
    _parse,
    _text_hash,
    _to_state_code,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mock_session):
    """WeatherClient wired to a mock HTTP session and frozen time."""
    return WeatherClient(
        user_agent="(WeatherLens AI, test@example.com)",
        rps=0,  # disable pacing in unit tests
    )


@pytest.fixture
def mock_session():
    """Patch requests.Session so no real HTTP is made."""
    with patch("weather_client.requests.Session") as cls:
        session = MagicMock()
        cls.return_value = session
        yield session


@pytest.fixture
def alert_feature():
    """Minimal NWS alert feature matching the real API shape."""
    return {
        "id": "urn:oid:2.49.0.1.840.0.abc123",
        "properties": {
            "id": "urn:oid:2.49.0.1.840.0.abc123",
            "event": "Flash Flood Warning",
            "headline": "Flash Flood Warning issued for Cook County",
            "description": "Heavy rainfall will cause flooding.",
            "instruction": "Move to higher ground immediately.",
            "areaDesc": "Cook County, IL",
            "sent": "2026-08-08T14:00:00-05:00",
            "effective": "2026-08-08T14:00:00-05:00",
            "expires": "2026-08-08T20:00:00-05:00",
            "severity": "Extreme",
        },
    }


@pytest.fixture
def forecast_response():
    """Minimal NWS gridpoint forecast response."""
    return {
        "properties": {
            "generatedAt": "2026-08-08T12:00:00Z",
            "periods": [
                {
                    "number": 1,
                    "name": "Tonight",
                    "startTime": "2026-08-08T18:00:00-05:00",
                    "endTime": "2026-08-09T06:00:00-05:00",
                    "shortForecast": "Mostly Cloudy",
                    "detailedForecast": "Mostly cloudy, with a low around 68.",
                },
                {
                    "number": 2,
                    "name": "Sunday",
                    "startTime": "2026-08-09T06:00:00-05:00",
                    "endTime": "2026-08-09T18:00:00-05:00",
                    "shortForecast": "Sunny",
                    "detailedForecast": "Sunny, with a high near 85.",
                },
            ],
        }
    }


@pytest.fixture
def grid():
    return NWSGrid(
        label="Chicago, IL",
        lat=41.8781,
        lon=-87.6298,
        state="IL",
        office="LOT",
        gx=65,
        gy=72,
    )


# ---------------------------------------------------------------------------
# _to_state_code
# ---------------------------------------------------------------------------

class TestToStateCode:
    def test_abbreviation(self):
        assert _to_state_code("IL") == "IL"

    def test_abbreviation_lowercase(self):
        assert _to_state_code("il") == "IL"

    def test_full_name(self):
        assert _to_state_code("Illinois") == "IL"

    def test_full_name_uppercase(self):
        assert _to_state_code("ILLINOIS") == "IL"

    def test_full_name_with_whitespace(self):
        assert _to_state_code("  California  ") == "CA"

    def test_multiword_state(self):
        assert _to_state_code("New York") == "NY"

    def test_territory(self):
        assert _to_state_code("Puerto Rico") == "PR"

    def test_unknown_returns_none(self):
        assert _to_state_code("Narnia") is None

    def test_empty_returns_none(self):
        assert _to_state_code("") is None


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------

class TestParse:
    def test_city_state_abbreviation(self):
        loc = _parse("Chicago, IL")
        assert loc.kind == "city"
        assert loc.city == "Chicago"
        assert loc.state == "IL"

    def test_city_full_state_name(self):
        loc = _parse("Austin, Texas")
        assert loc.kind == "city"
        assert loc.state == "TX"

    def test_coordinate_pair(self):
        loc = _parse("41.8781,-87.6298")
        assert loc.kind == "coords"
        assert loc.lat == 41.8781
        assert loc.lon == -87.6298

    def test_coordinates_rounded_to_4dp(self):
        loc = _parse("41.878123456,-87.629812345")
        assert loc.lat == 41.8781
        assert loc.lon == -87.6298

    def test_coordinates_with_spaces(self):
        loc = _parse("  41.8781 , -87.6298  ")
        assert loc.kind == "coords"

    def test_negative_coordinates(self):
        loc = _parse("-33.8688,151.2093")
        assert loc.lat == -33.8688
        assert loc.lon == 151.2093

    def test_label_coords(self):
        loc = _parse("41.8781,-87.6298")
        assert loc.label == "41.8781,-87.6298"

    def test_label_city(self):
        loc = _parse("Chicago, IL")
        assert loc.label == "Chicago, IL"

    def test_bare_city_raises(self):
        with pytest.raises(BadLocation, match="Cannot parse"):
            _parse("Springfield")

    def test_empty_raises(self):
        with pytest.raises(BadLocation):
            _parse("")

    def test_non_string_raises(self):
        with pytest.raises(BadLocation):
            _parse({"lat": 41.8})  # type: ignore

    def test_unknown_state_raises(self):
        with pytest.raises(BadLocation, match="Cannot parse"):
            _parse("Chicago, ZZ")

    def test_latitude_out_of_range(self):
        with pytest.raises(BadLocation, match="Latitude"):
            _parse("91.0,-87.6298")

    def test_longitude_out_of_range(self):
        with pytest.raises(BadLocation, match="Longitude"):
            _parse("41.8781,-181.0")


# ---------------------------------------------------------------------------
# _text_hash
# ---------------------------------------------------------------------------

class TestTextHash:
    def test_is_sha256(self):
        text = "Heavy rain expected."
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert _text_hash(text) == expected

    def test_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_texts_differ(self):
        assert _text_hash("rain") != _text_hash("snow")

    def test_empty_string(self):
        assert len(_text_hash("")) == 64  # sha256 hex is always 64 chars


# ---------------------------------------------------------------------------
# WeatherDoc
# ---------------------------------------------------------------------------

class TestWeatherDoc:
    def _make(self, **overrides):
        defaults = dict(
            id="test-id",
            location="Chicago, IL",
            latitude=41.8781,
            longitude=-87.6298,
            source_type=ALERT,
            event="Flood Warning",
            headline="Flood Warning issued",
            narrative_text="Heavy rain expected in the area.",
            issued_at="2026-08-08T14:00:00Z",
            effective_at="2026-08-08T14:00:00Z",
            expires_at="2026-08-08T20:00:00Z",
            severity="Moderate",
            payload={},
        )
        defaults.update(overrides)
        return WeatherDoc(**defaults)

    def test_content_hash_auto_computed(self):
        doc = self._make()
        assert doc.content_hash == _text_hash("Heavy rain expected in the area.")

    def test_explicit_hash_respected(self):
        doc = self._make(content_hash="custom-hash")
        assert doc.content_hash == "custom-hash"

    def test_as_row_has_all_keys(self):
        doc = self._make()
        row = doc.as_row()
        expected = {
            "id", "location", "latitude", "longitude", "source_type",
            "event", "headline", "narrative_text", "issued_at",
            "effective_at", "expires_at", "severity", "payload", "content_hash",
        }
        assert set(row.keys()) == expected

    def test_as_row_values_match(self):
        doc = self._make()
        row = doc.as_row()
        assert row["id"] == "test-id"
        assert row["source_type"] == ALERT
        assert row["content_hash"] == doc.content_hash


# ---------------------------------------------------------------------------
# WeatherClient — construction
# ---------------------------------------------------------------------------

class TestWeatherClientInit:
    def test_missing_user_agent_raises(self):
        with patch("weather_client.NWS_AGENT", ""):
            with pytest.raises(ValueError, match="User-Agent"):
                WeatherClient()

    def test_blank_user_agent_raises(self):
        with pytest.raises(ValueError, match="User-Agent"):
            WeatherClient(user_agent="   ")

    def test_valid_user_agent_accepted(self):
        c = WeatherClient(user_agent="(WeatherLens AI, test@example.com)")
        assert c is not None


# ---------------------------------------------------------------------------
# WeatherClient — _get (HTTP layer)
# ---------------------------------------------------------------------------

class TestGet:
    def _resp(self, json_data, status=200):
        r = MagicMock()
        r.json.return_value = json_data
        r.status_code = status
        if status >= 400:
            r.raise_for_status.side_effect = requests.HTTPError(
                response=MagicMock(status_code=status, json=MagicMock(return_value=json_data))
            )
        else:
            r.raise_for_status.return_value = None
        return r

    def test_returns_json_on_200(self, client, mock_session):
        mock_session.get.return_value = self._resp({"status": "ok"})
        result = client._get("http://example.com/test")
        assert result == {"status": "ok"}

    def test_passes_params(self, client, mock_session):
        mock_session.get.return_value = self._resp({})
        client._get("http://example.com/test", params={"area": "IL"})
        mock_session.get.assert_called_once_with(
            "http://example.com/test", params={"area": "IL"}, timeout=30
        )

    def test_retries_on_500(self, client, mock_session):
        err_resp = MagicMock(status_code=500)
        err = requests.HTTPError(response=err_resp)
        err_resp.raise_for_status = MagicMock(side_effect=err)

        fail = MagicMock()
        fail.raise_for_status.side_effect = requests.HTTPError(response=err_resp)

        ok = self._resp({"ok": True})
        mock_session.get.side_effect = [fail, fail, ok]

        with patch("weather_client.time.sleep"):
            result = client._get("http://example.com")
        assert mock_session.get.call_count == 3
        assert result == {"ok": True}

    def test_no_retry_on_400(self, client, mock_session):
        err_resp = MagicMock(
            status_code=400,
            json=MagicMock(return_value={"detail": "bad param", "title": "Bad Request"}),
        )
        exc = requests.HTTPError(response=err_resp)
        fail = MagicMock()
        fail.raise_for_status.side_effect = exc

        mock_session.get.return_value = fail
        with pytest.raises(requests.HTTPError, match="bad param"):
            client._get("http://example.com")

        assert mock_session.get.call_count == 1  # no retry

    def test_problem_detail_folded_into_message(self, client, mock_session):
        err_resp = MagicMock(
            status_code=422,
            json=MagicMock(return_value={"detail": "limit not allowed here"}),
        )
        fail = MagicMock()
        fail.raise_for_status.side_effect = requests.HTTPError(response=err_resp)
        mock_session.get.return_value = fail

        with pytest.raises(requests.HTTPError, match="limit not allowed here"):
            client._get("http://example.com")

    def test_raises_after_3_failures(self, client, mock_session):
        fail = MagicMock()
        fail.raise_for_status.side_effect = requests.ConnectionError("timeout")
        mock_session.get.side_effect = requests.ConnectionError("timeout")

        with patch("weather_client.time.sleep"):
            with pytest.raises(requests.ConnectionError):
                client._get("http://example.com")

        assert mock_session.get.call_count == 3


# ---------------------------------------------------------------------------
# WeatherClient — _alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_fetches_correct_url(self, client, mock_session, grid):
        mock_session.get.return_value = MagicMock(
            raise_for_status=MagicMock(), json=MagicMock(return_value={"features": []})
        )
        client._alerts(grid)
        url = mock_session.get.call_args[0][0]
        assert url.endswith("/alerts/active")

    def test_passes_state_param(self, client, mock_session, grid):
        mock_session.get.return_value = MagicMock(
            raise_for_status=MagicMock(), json=MagicMock(return_value={"features": []})
        )
        client._alerts(grid)
        params = mock_session.get.call_args[1]["params"]
        assert params == {"area": "IL"}

    def test_no_limit_param_sent(self, client, mock_session, grid):
        mock_session.get.return_value = MagicMock(
            raise_for_status=MagicMock(), json=MagicMock(return_value={"features": []})
        )
        client._alerts(grid)
        params = mock_session.get.call_args[1]["params"]
        assert "limit" not in params

    def test_empty_features_returns_empty_list(self, client, mock_session, grid):
        mock_session.get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"features": []}),
        )
        result = client._alerts(grid)
        assert result == []

    def test_missing_features_key_returns_empty_list(self, client, mock_session, grid):
        # NWS /alerts/active returns no 'features' key when it rejects the request.
        # Treating this as "no alerts" rather than crashing is the correct behaviour.
        mock_session.get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"title": "Bad Request"}),
        )
        result = client._alerts(grid)
        assert result == []


# ---------------------------------------------------------------------------
# WeatherClient — _to_alert_docs
# ---------------------------------------------------------------------------

class TestToAlertDocs:
    def test_returns_weather_doc(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert len(docs) == 1
        assert isinstance(docs[0], WeatherDoc)

    def test_id_is_nws_urn(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert docs[0].id == "urn:oid:2.49.0.1.840.0.abc123"

    def test_narrative_joins_description_and_instruction(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert "Heavy rainfall will cause flooding." in docs[0].narrative_text
        assert "Move to higher ground immediately." in docs[0].narrative_text
        assert "\n\n" in docs[0].narrative_text

    def test_skips_feature_with_no_narrative(self, client, grid):
        feature = {
            "id": "urn:oid:test",
            "properties": {"id": "urn:oid:test", "description": "", "instruction": None},
        }
        docs = client._to_alert_docs(grid, [feature])
        assert docs == []

    def test_uses_area_desc_as_location(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert docs[0].location == "Cook County, IL"

    def test_falls_back_to_grid_label_when_no_area_desc(self, client, grid):
        feature = {
            "id": "urn:oid:test",
            "properties": {
                "id": "urn:oid:test",
                "description": "Some text.",
                "instruction": None,
                "areaDesc": None,
            },
        }
        docs = client._to_alert_docs(grid, [feature])
        assert docs[0].location == grid.label

    def test_source_type_is_alert(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert docs[0].source_type == ALERT

    def test_content_hash_is_set(self, client, grid, alert_feature):
        docs = client._to_alert_docs(grid, [alert_feature])
        assert docs[0].content_hash == _text_hash(docs[0].narrative_text)

    def test_only_description_no_instruction(self, client, grid):
        feature = {
            "id": "urn:oid:test",
            "properties": {
                "id": "urn:oid:test",
                "description": "Wind advisory in effect.",
                "instruction": None,
            },
        }
        docs = client._to_alert_docs(grid, [feature])
        assert docs[0].narrative_text == "Wind advisory in effect."

    def test_skips_feature_without_id(self, client, grid):
        feature = {"properties": {"description": "Some text.", "instruction": None}}
        docs = client._to_alert_docs(grid, [feature])
        assert docs == []


# ---------------------------------------------------------------------------
# WeatherClient — _to_forecast_docs
# ---------------------------------------------------------------------------

class TestToForecastDocs:
    def test_returns_weather_docs(self, client, grid, forecast_response):
        docs = client._to_forecast_docs(grid, forecast_response)
        assert len(docs) == 2
        assert all(isinstance(d, WeatherDoc) for d in docs)

    def test_source_type_is_forecast(self, client, grid, forecast_response):
        docs = client._to_forecast_docs(grid, forecast_response)
        assert all(d.source_type == FORECAST for d in docs)

    def test_id_uses_time_window_not_period_name(self, client, grid, forecast_response):
        docs = client._to_forecast_docs(grid, forecast_response)
        # ID must include office and grid, keyed on time window
        assert docs[0].id.startswith("forecast:LOT/65,72:")
        # Must NOT contain the period name "Tonight"
        assert "Tonight" not in docs[0].id

    def test_same_window_produces_same_id(self, client, grid, forecast_response):
        # Simulate NWS renaming "Tonight" to "This Evening" — id must not change.
        import copy
        renamed = copy.deepcopy(forecast_response)
        renamed["properties"]["periods"][0]["name"] = "This Evening"
        docs_original = client._to_forecast_docs(grid, forecast_response)
        docs_renamed  = client._to_forecast_docs(grid, renamed)
        assert docs_original[0].id == docs_renamed[0].id

    def test_skips_period_with_no_narrative(self, client, grid):
        response = {
            "properties": {
                "generatedAt": "2026-08-08T12:00:00Z",
                "periods": [{"name": "Tonight", "startTime": "T1", "endTime": "T2",
                              "shortForecast": "Cloudy", "detailedForecast": ""}],
            }
        }
        docs = client._to_forecast_docs(grid, response)
        assert docs == []

    def test_content_hash_matches_narrative(self, client, grid, forecast_response):
        docs = client._to_forecast_docs(grid, forecast_response)
        for doc in docs:
            assert doc.content_hash == _text_hash(doc.narrative_text)

    def test_empty_response(self, client, grid):
        assert client._to_forecast_docs(grid, {}) == []


# ---------------------------------------------------------------------------
# WeatherClient — fetch (integration of parts, all HTTP mocked)
# ---------------------------------------------------------------------------

class TestFetch:
    def _setup(self, client, mock_session, alerts, forecast):
        """Wire mock responses: geocode → points → alerts → forecast."""
        responses = [
            # Open-Meteo geocode
            {"results": [{"latitude": 41.8781, "longitude": -87.6298,
                          "country_code": "US", "admin1": "Illinois"}]},
            # NWS /points
            {"properties": {"gridId": "LOT", "gridX": 65, "gridY": 72,
                            "relativeLocation": {"properties": {"city": "Chicago", "state": "IL"}}}},
            # /alerts/active
            {"features": alerts},
            # /gridpoints forecast
            forecast,
        ]
        mock_session.get.side_effect = [
            MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value=r))
            for r in responses
        ]

    def test_returns_list_of_weather_docs(self, client, mock_session,
                                           alert_feature, forecast_response):
        self._setup(client, mock_session, [alert_feature], forecast_response)
        docs = client.fetch(["Chicago, IL"])
        assert len(docs) > 0
        assert all(isinstance(d, WeatherDoc) for d in docs)

    def test_deduplicates_by_id(self, client, mock_session,
                                 alert_feature, forecast_response):
        # Same alert from two locations — only one should appear.
        self._setup(client, mock_session, [alert_feature, alert_feature], forecast_response)
        docs = client.fetch(["Chicago, IL"])
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_bad_location_recorded_not_raised(self, client, mock_session,
                                               alert_feature, forecast_response):
        # First location fails, second succeeds.
        mock_session.get.side_effect = [
            MagicMock(raise_for_status=MagicMock(
                side_effect=requests.ConnectionError("timeout"))),
            # Then valid responses for the second location
            MagicMock(raise_for_status=MagicMock(),
                      json=MagicMock(return_value={
                          "results": [{"latitude": 41.8, "longitude": -87.6,
                                       "country_code": "US", "admin1": "Illinois"}]})),
            MagicMock(raise_for_status=MagicMock(),
                      json=MagicMock(return_value={
                          "properties": {"gridId": "LOT", "gridX": 65, "gridY": 72,
                                         "relativeLocation": {"properties": {
                                             "city": "Chicago", "state": "IL"}}}})),
            MagicMock(raise_for_status=MagicMock(),
                      json=MagicMock(return_value={"features": [alert_feature]})),
            MagicMock(raise_for_status=MagicMock(),
                      json=MagicMock(return_value=forecast_response)),
        ]
        docs = client.fetch(["BAD LOCATION 999", "Chicago, IL"])
        assert len(client.errors) == 1
        assert "BAD LOCATION 999" in client.errors[0]
        assert len(docs) > 0

    def test_respects_limit(self, client, mock_session, alert_feature, forecast_response):
        self._setup(client, mock_session, [alert_feature], forecast_response)
        docs = client.fetch(["Chicago, IL"], limit=2)
        assert len(docs) <= 2

    def test_errors_reset_between_calls(self, client, mock_session,
                                         alert_feature, forecast_response):
        mock_session.get.side_effect = requests.ConnectionError("down")
        with patch("weather_client.time.sleep"):
            client.fetch(["Bad, ZZ"])
        assert len(client.errors) == 1

        self._setup(client, mock_session, [alert_feature], forecast_response)
        client.fetch(["Chicago, IL"])
        assert client.errors == []
