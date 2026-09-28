"""
Tests for the get_coach_report MCP tool with a mocked Intervals.icu API (no network).
"""

import asyncio
import json
import os
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("API_KEY", "test")
os.environ.setdefault("ATHLETE_ID", "i1")

from intervals_mcp_server.server import get_coach_report  # pylint: disable=wrong-import-position  # noqa: E402
from intervals_mcp_server.tools import coach_report as tool_module  # pylint: disable=wrong-import-position  # noqa: E402
from tests.coach_fixtures import END, realistic_scenario  # pylint: disable=wrong-import-position  # noqa: E402


def _fake_api(monkeypatch, overrides=None):
    """Route requests by URL suffix to fixture data and record them."""
    activities, wellness, athlete = realistic_scenario()
    responses = {"athlete": athlete, "activities": activities, "wellness": wellness}
    responses.update(overrides or {})
    calls = []

    async def fake_request(url, api_key=None, params=None, method="GET", data=None):  # pylint: disable=unused-argument
        calls.append((url, params))
        if url.endswith("/activities"):
            return responses["activities"]
        if url.endswith("/wellness"):
            return responses["wellness"]
        return responses["athlete"]

    monkeypatch.setattr(tool_module, "make_intervals_request", fake_request)
    return calls


def test_get_coach_report_returns_compact_json(monkeypatch):
    calls = _fake_api(monkeypatch)
    result = asyncio.run(get_coach_report(days=28, end_date=END.isoformat()))
    report = json.loads(result)
    assert report["schema_version"] == "1.0"
    assert report["period"]["end"] == END.isoformat()
    assert len(result.encode("utf-8")) < 4096
    assert " " not in result.split('"code"')[0]  # compact separators
    params = {url.rsplit("/", 1)[-1]: p for url, p in calls}
    assert params["activities"] == {"oldest": "2026-08-30", "newest": "2026-09-28"}
    assert params["wellness"] == {"oldest": "2026-07-30", "newest": "2026-09-27"}
    assert all("i1" in url for url, _ in calls)


def test_get_coach_report_short_window_still_fetches_fixed_windows(monkeypatch):
    calls = _fake_api(monkeypatch)
    report = json.loads(asyncio.run(get_coach_report(days=7, end_date=END.isoformat())))
    assert report["period"]["days"] == 7
    assert len(report["volume"]) == 1
    params = {url.rsplit("/", 1)[-1]: p for url, p in calls}
    assert params["activities"]["oldest"] == "2026-08-30"


def test_get_coach_report_default_end_uses_athlete_timezone(monkeypatch):
    _fake_api(monkeypatch)
    monkeypatch.setattr(tool_module, "_athlete_today", lambda athlete: END)
    report = json.loads(asyncio.run(get_coach_report()))
    assert report["period"]["end"] == END.isoformat()
    assert report["period"]["load_end"] == END.isoformat()  # END already has an activity
    assert report["period"]["days"] == 28


def test_get_coach_report_morning_without_activity_ends_load_yesterday(monkeypatch):
    _fake_api(monkeypatch)
    today = END + timedelta(days=1)  # Monday, no activity recorded yet
    monkeypatch.setattr(tool_module, "_athlete_today", lambda athlete: today)
    report = json.loads(asyncio.run(get_coach_report()))
    assert report["period"]["end"] == today.isoformat()
    assert report["period"]["load_end"] == END.isoformat()
    assert report["period"]["start"] == "2026-08-31"
    assert [week["week"] for week in report["volume"]][-1] == "2026-W39"


def test_get_coach_report_explicit_end_date_is_not_shifted(monkeypatch):
    _fake_api(monkeypatch)
    today = END + timedelta(days=1)
    report = json.loads(asyncio.run(get_coach_report(end_date=today.isoformat())))
    assert report["period"]["load_end"] == today.isoformat()


def test_athlete_today_timezone_handling():
    assert isinstance(tool_module._athlete_today({"timezone": "Europe/Berlin"}), date)  # pylint: disable=protected-access
    assert tool_module._athlete_today({"timezone": "Not/AZone"}) == date.today()  # pylint: disable=protected-access
    assert tool_module._athlete_today(None) == date.today()  # pylint: disable=protected-access


def test_get_coach_report_validates_input(monkeypatch):
    calls = _fake_api(monkeypatch)
    assert asyncio.run(get_coach_report(days=3)).startswith("Error: days must be between 7 and 90")
    assert asyncio.run(get_coach_report(days=91)).startswith("Error: days must be between 7 and 90")
    assert asyncio.run(get_coach_report(end_date="27.09.2026")).startswith("Error: Invalid date format")
    assert not calls


def test_get_coach_report_passes_api_errors_through(monkeypatch):
    _fake_api(monkeypatch, {"activities": {"error": True, "message": "401 Unauthorized"}})
    result = asyncio.run(get_coach_report(end_date=END.isoformat()))
    assert result == "Error fetching activities: 401 Unauthorized"

    _fake_api(monkeypatch, {"wellness": {"error": True, "message": "boom"}})
    assert asyncio.run(get_coach_report(end_date=END.isoformat())) == "Error fetching wellness data: boom"

    _fake_api(monkeypatch, {"athlete": {"error": True, "message": "403 Forbidden"}})
    assert asyncio.run(get_coach_report()) == "Error fetching athlete: 403 Forbidden"


def test_get_coach_report_with_empty_data(monkeypatch):
    _fake_api(monkeypatch, {"activities": [], "wellness": [], "athlete": {}})
    report = json.loads(asyncio.run(get_coach_report(end_date=END.isoformat())))
    assert report["load"]["acwr"] is None
    assert report["recovery"]["hrv"]["status"] == "insufficient_data"
