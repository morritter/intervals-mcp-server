"""
Coach report MCP tool for Intervals.icu.

This module only fetches data. All numbers are computed by the pure
``intervals_mcp_server.coach_metrics`` module.
"""

import asyncio
import json
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from intervals_mcp_server.api.client import make_intervals_request
from intervals_mcp_server.coach_metrics import DEFAULT_CONFIG, build_coach_report
from intervals_mcp_server.config import get_config
from intervals_mcp_server.utils.validation import validate_date

# Import mcp instance from shared module for tool registration
from intervals_mcp_server.mcp_instance import mcp  # noqa: F401

config = get_config()

MIN_DAYS = 7
MAX_DAYS = 90


def _athlete_today(athlete: Any) -> date:
    """Today in the athlete's time zone; the server itself may run in UTC."""
    timezone_name = athlete.get("timezone") if isinstance(athlete, dict) else None
    if isinstance(timezone_name, str) and timezone_name:
        try:
            return datetime.now(ZoneInfo(timezone_name)).date()
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return date.today()


def _error_message(result: Any) -> str | None:
    """Message of an error dict returned by make_intervals_request, else None."""
    if isinstance(result, dict) and result.get("error"):
        return str(result.get("message"))
    return None


async def _fetch_activities(athlete_id: str, end: date, days: int) -> Any:
    oldest = end - timedelta(days=max(days, DEFAULT_CONFIG.chronic_days) - 1)
    # newest is padded by one day; activities after the end date are dropped later.
    params = {"oldest": oldest.isoformat(), "newest": (end + timedelta(days=1)).isoformat()}
    return await make_intervals_request(url=f"/athlete/{athlete_id}/activities", params=params)


async def _fetch_wellness(athlete_id: str, end: date, days: int) -> Any:
    oldest = end - timedelta(days=max(days, DEFAULT_CONFIG.baseline_days) - 1)
    params = {"oldest": oldest.isoformat(), "newest": end.isoformat()}
    return await make_intervals_request(url=f"/athlete/{athlete_id}/wellness", params=params)


@mcp.tool()
async def get_coach_report(days: int = 28, end_date: str | None = None) -> str:  # pylint: disable=too-many-return-statements
    """Compact, pre-computed training report (JSON) for coaching questions.

    Use this FIRST for overview questions: how training is going, load, fatigue and
    form, recovery (HRV, resting HR, sleep), weekly volume per sport, intensity
    distribution, durability, FTP vs. eFTP, weekly reviews and planning the next
    days. All numbers are computed deterministically on the server; interpret them,
    do not recompute them from raw activities. Use get_activities,
    get_activity_details or get_activity_intervals only to inspect single sessions.

    Args:
        days: Report window in days (7-90, default 28). ACWR/monotony (7 and 28 days)
            and the HRV/RHR baselines (60 days) always use their fixed windows.
        end_date: Last day of the report, YYYY-MM-DD (default: today in the
            athlete's time zone).

    Output keys (schema_version 1.0); null always means "not enough data", never 0:
    - flags: rule-based hints {code, sev: alarm|warning|info, val, thr}, most
      severe first. Start here.
    - load: ctl, atl, tsb, ramp at the end date (src: api, or
      recomputed_without_planned when Intervals.icu already counted planned
      workouts); load_7d, load_28d, acwr, monotony (Foster, 7 days incl. rest
      days), strain, primary_sport, primary_monotony, effective_monotony
      (multi-sport corrected, used for flags), deload.
    - recovery.hrv: 7-day mean of ln(rMSSD) vs. 60-day band (mean +/- 0.5 SD),
      status below|within|above|insufficient_data, days_below (streak).
      recovery.rhr: 7-day vs. 60-day mean, delta (bpm), days_high (streak >= +5).
      recovery.sleep: avg_h, avg_7d_h, nights_short (< 7 h), recent_h (last 3
      nights, oldest first).
    - volume: per ISO week and sport family (Rad = Ride + VirtualRide, Lauf,
      Schwimmen, Kraft, Sonstige): h, load, n, km; rest_days, hard_days,
      weekly monotony.
    - intensity: 3-zone shares pct [low, moderate, high] for all sports and for
      Rad (power for cycling, HR otherwise; mapping in intensity.map), Treff
      polarization index pi (pi_note explains edge cases), class cls, and drift
      of the last 7 vs. 28 days.
    - capability: durability = median aerobic decoupling (%) of steady rides/runs
      >= 60 min after a quality filter, with excluded count and reasons;
      ef = efficiency factor 7 vs. 28 days; eftp = cycling eFTP now vs. 28/56
      days ago.
    - top_sessions: up to 5 notable sessions (highest load, plus the highest IF).
    - thresholds: FTP, indoor FTP, LTHR. coverage: available data (days with
      HRV/RHR/sleep, sessions with power/HR zones, RPE/feel).
    """
    if not MIN_DAYS <= days <= MAX_DAYS:
        return f"Error: days must be between {MIN_DAYS} and {MAX_DAYS}."
    end: date | None = None
    if end_date:
        try:
            end = date.fromisoformat(validate_date(end_date))
        except ValueError as exc:
            return f"Error: {exc}"

    athlete_id = config.athlete_id
    if not athlete_id:
        return "Error: No ATHLETE_ID found in environment variables."

    if end is None:
        athlete = await make_intervals_request(url=f"/athlete/{athlete_id}")
        if (message := _error_message(athlete)) is not None:
            return f"Error fetching athlete: {message}"
        end = _athlete_today(athlete)
        activities, wellness = await asyncio.gather(
            _fetch_activities(athlete_id, end, days), _fetch_wellness(athlete_id, end, days)
        )
    else:
        athlete, activities, wellness = await asyncio.gather(
            make_intervals_request(url=f"/athlete/{athlete_id}"),
            _fetch_activities(athlete_id, end, days),
            _fetch_wellness(athlete_id, end, days),
        )
        if (message := _error_message(athlete)) is not None:
            return f"Error fetching athlete: {message}"

    if (message := _error_message(activities)) is not None:
        return f"Error fetching activities: {message}"
    if (message := _error_message(wellness)) is not None:
        return f"Error fetching wellness data: {message}"

    report = build_coach_report(activities, wellness, athlete, end, days)
    return json.dumps(report, separators=(",", ":"), ensure_ascii=False)
