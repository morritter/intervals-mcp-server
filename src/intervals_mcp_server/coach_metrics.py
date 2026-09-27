"""
Deterministic training metrics for the coach report.

This module is pure: no network access, no clock, no logging. The MCP tool in
``tools/coach_report.py`` fetches raw Intervals.icu data and passes it in together
with an explicit end date. Every number an LLM coach needs is computed here so the
model only has to interpret, not calculate.

All thresholds and mappings live in :class:`CoachConfig`.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

SCHEMA_VERSION = "1.0"

Activity = dict[str, Any]
WellnessRecord = dict[str, Any]

RAD = "Rad"
LAUF = "Lauf"
SCHWIMMEN = "Schwimmen"
KRAFT = "Kraft"
SONSTIGE = "Sonstige"

# Intervals.icu activity type -> sport family used throughout the report.
SPORT_FAMILIES: dict[str, str] = {
    "Ride": RAD,
    "VirtualRide": RAD,
    "GravelRide": RAD,
    "MountainBikeRide": RAD,
    "EBikeRide": RAD,
    "Run": LAUF,
    "TrailRun": LAUF,
    "VirtualRun": LAUF,
    "Swim": SCHWIMMEN,
    "OpenWaterSwim": SCHWIMMEN,
    "WeightTraining": KRAFT,
}


@dataclass(frozen=True)
class CoachConfig:  # pylint: disable=too-many-instance-attributes
    """Central place for every threshold, window and mapping used by the report."""

    # Fixed analysis windows (days, inclusive of the end date).
    acute_days: int = 7
    chronic_days: int = 28
    baseline_days: int = 60

    # Load
    acwr_low_warn: float = 0.8
    acwr_high_warn: float = 1.3
    acwr_low_alarm: float = 0.75
    acwr_high_alarm: float = 1.35
    monotony_warn: float = 2.0
    monotony_alarm: float = 2.5
    primary_sport_min_active_days: int = 3
    deload_ratio: float = 0.8
    week_monotony_min_days: int = 5
    week_monotony_min_active_days: int = 3

    # Thresholds and data quality
    ftp_eftp_max_dev_pct: float = 2.0
    sport_inactive_days: int = 7
    tracked_sports: tuple[str, ...] = (RAD, LAUF, SCHWIMMEN, KRAFT)
    subjective_missing_share: float = 0.8

    # Recovery
    hrv_valid_min: float = 10.0
    hrv_valid_max: float = 250.0
    hrv_band_k: float = 0.5
    hrv_min_points_acute: int = 4
    hrv_min_points_baseline: int = 21
    hrv_alarm_days: int = 3
    rhr_high_delta_bpm: float = 5.0
    rhr_warn_days: int = 2
    rhr_alarm_days: int = 3
    rhr_min_points_acute: int = 4
    rhr_min_points_baseline: int = 21
    sleep_short_hours: float = 7.0
    sleep_recent_nights: int = 3
    sleep_short_min_nights: int = 2

    # Top sessions
    top_n: int = 5


DEFAULT_CONFIG = CoachConfig()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Return ``value`` as float if it is a finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _r(value: float | None, digits: int = 1) -> float | int | None:
    """Round for output; 0 digits yields an int, None stays None."""
    if value is None or not math.isfinite(value):
        return None
    if digits == 0:
        return int(round(value))
    return round(value, digits)


def parse_day(value: Any) -> date | None:
    """Parse the date part of an ISO date/datetime string."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def days_back(end: date, count: int) -> list[date]:
    """Return ``count`` consecutive days ending at ``end`` (oldest first)."""
    return [end - timedelta(days=offset) for offset in range(count - 1, -1, -1)]


def sport_family(activity_type: Any) -> str:
    """Map an Intervals.icu activity type to a report sport family."""
    if not isinstance(activity_type, str):
        return SONSTIGE
    return SPORT_FAMILIES.get(activity_type, SONSTIGE)


def activity_day(activity: Activity) -> date | None:
    """Local calendar day of an activity."""
    return parse_day(activity.get("start_date_local"))


def activity_load(activity: Activity) -> float:
    """Training load of an activity; missing or negative loads count as 0."""
    load = _num(activity.get("icu_training_load"))
    return max(load, 0.0) if load is not None else 0.0


def activities_between(activities: list[Activity], start: date, end: date) -> list[Activity]:
    """Activities whose local day lies within [start, end]."""
    selected = []
    for activity in activities:
        day = activity_day(activity)
        if day is not None and start <= day <= end:
            selected.append(activity)
    return selected


def daily_loads(
    activities: list[Activity], end: date, count: int, family: str | None = None
) -> list[float]:
    """Daily training load for ``count`` days ending at ``end``, zero-filled."""
    per_day: dict[date, float] = defaultdict(float)
    for activity in activities:
        if family is not None and sport_family(activity.get("type")) != family:
            continue
        day = activity_day(activity)
        if day is not None:
            per_day[day] += activity_load(activity)
    return [per_day.get(day, 0.0) for day in days_back(end, count)]


def monotony(daily: list[float]) -> float | None:
    """Foster monotony: mean / sample SD of daily load (0-days included).

    Returns None when fewer than two days are given or the SD is 0 (every day
    identical, including a period without any training), because the ratio is
    undefined there.
    """
    if len(daily) < 2:
        return None
    sd = statistics.stdev(daily)
    if sd == 0:
        return None
    return statistics.mean(daily) / sd


def wellness_by_day(wellness: Any) -> dict[date, WellnessRecord]:
    """Index wellness records (list or date-keyed dict from the API) by day."""
    records: list[tuple[Any, Any]] = []
    if isinstance(wellness, dict):
        records = list(wellness.items())
    elif isinstance(wellness, list):
        records = [(item.get("id") if isinstance(item, dict) else None, item) for item in wellness]
    indexed: dict[date, WellnessRecord] = {}
    for key, record in records:
        if not isinstance(record, dict):
            continue
        day = parse_day(record.get("id")) or parse_day(key)
        if day is not None:
            indexed[day] = record
    return indexed


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_metrics(
    activities: list[Activity], end: date, config: CoachConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Acute/chronic load, ACWR, Foster monotony and strain, primary-sport monotony, deload.

    * ACWR = load of the last 7 days / (load of the last 28 days / 4); None if the
      28-day load is 0.
    * Monotony over the last 7 daily loads including rest days (None if undefined,
      see :func:`monotony`); strain = 7-day load x monotony.
    * Primary sport = sport family with the highest 7-day load. Its own monotony is
      only computed with at least ``primary_sport_min_active_days`` active days.
      With more than one sport family, ``effective_monotony`` uses the primary
      sport value, because a steady cross-training floor (e.g. strength) inflates
      total monotony without adding overuse risk (Section 11 approach).
    * Deload: 7-day load <= ``deload_ratio`` x 28-day weekly average.
    """
    acute = daily_loads(activities, end, config.acute_days)
    chronic = daily_loads(activities, end, config.chronic_days)
    load_acute = sum(acute)
    load_chronic = sum(chronic)
    weeks_chronic = config.chronic_days / 7

    acwr = load_acute / (load_chronic / weeks_chronic) if load_chronic > 0 else None
    total_monotony = monotony(acute)
    monotony_note = None
    if total_monotony is None:
        monotony_note = "no_load" if load_acute == 0 else "no_variation"
    strain = load_acute * total_monotony if total_monotony is not None else None

    acute_start = end - timedelta(days=config.acute_days - 1)
    families = sorted(
        {
            sport_family(a.get("type"))
            for a in activities_between(activities, acute_start, end)
            if activity_load(a) > 0
        }
    )
    family_loads = {
        family: daily_loads(activities, end, config.acute_days, family) for family in families
    }
    primary = None
    primary_monotony = None
    if family_loads:
        primary = max(families, key=lambda family: sum(family_loads[family]))
        active_days = sum(1 for load in family_loads[primary] if load > 0)
        if active_days >= config.primary_sport_min_active_days:
            primary_monotony = monotony(family_loads[primary])
    multi_sport = len(family_loads) > 1
    effective = (
        primary_monotony if multi_sport and primary_monotony is not None else total_monotony
    )

    deload = load_chronic > 0 and load_acute <= config.deload_ratio * load_chronic / weeks_chronic

    return {
        "load_7d": _r(load_acute, 0),
        "load_28d": _r(load_chronic, 0),
        "acwr": _r(acwr, 2),
        "monotony": _r(total_monotony, 2),
        "monotony_note": monotony_note,
        "strain": _r(strain, 0),
        "primary_sport": primary,
        "primary_monotony": _r(primary_monotony, 2),
        "effective_monotony": _r(effective, 2),
        "deload": deload,
    }


def fitness_status(
    wellness: dict[date, WellnessRecord], activities: list[Activity], end: date
) -> dict[str, Any]:
    """CTL, ATL, TSB (= CTL - ATL) and ramp rate at the end date.

    Uses the Intervals.icu values of the end date, or the last earlier day that has
    them (reported as ``as_of``). If the end date's ``ctlLoad`` is higher than the
    load actually completed that day, Intervals.icu has counted planned but not yet
    done workouts. CTL/ATL are then recomputed from the previous day with the
    Intervals.icu exponential model (time constants 42 and 7 days) and only the
    completed load; ramp rate becomes CTL(end) - CTL(end - 7 days).
    """
    record_day = None
    for day in days_back(end, 8)[::-1]:
        record = wellness.get(day, {})
        if _num(record.get("ctl")) is not None and _num(record.get("atl")) is not None:
            record_day = day
            break
    if record_day is None:
        return {"ctl": None, "atl": None, "tsb": None, "ramp": None, "src": None}

    record = wellness[record_day]
    ctl = _num(record.get("ctl"))
    atl = _num(record.get("atl"))
    ramp = _num(record.get("rampRate"))
    source = "api"

    if record_day == end:
        planned_load = _num(record.get("ctlLoad"))
        done_load = daily_loads(activities, end, 1)[0]
        previous = wellness.get(end - timedelta(days=1), {})
        prev_ctl = _num(previous.get("ctl"))
        prev_atl = _num(previous.get("atl"))
        if (
            planned_load is not None
            and planned_load > done_load + 1
            and prev_ctl is not None
            and prev_atl is not None
        ):
            ctl = prev_ctl + (done_load - prev_ctl) * (1 - math.exp(-1 / 42))
            atl = prev_atl + (done_load - prev_atl) * (1 - math.exp(-1 / 7))
            week_ago_ctl = _num(wellness.get(end - timedelta(days=7), {}).get("ctl"))
            ramp = ctl - week_ago_ctl if week_ago_ctl is not None else None
            source = "recomputed_without_planned"

    if ctl is None or atl is None:  # unreachable: the search above requires both
        return {"ctl": None, "atl": None, "tsb": None, "ramp": None, "src": None}
    result: dict[str, Any] = {
        "ctl": _r(ctl, 1),
        "atl": _r(atl, 1),
        "tsb": _r(ctl - atl, 1),
        "ramp": _r(ramp, 1),
        "src": source,
    }
    if record_day != end:
        result["as_of"] = record_day.isoformat()
    return result


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def iso_week_label(day: date) -> str:
    """ISO week label such as ``2026-W53`` (ISO year, not calendar year)."""
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def weekly_volume(
    activities: list[Activity], end: date, days: int, config: CoachConfig = DEFAULT_CONFIG
) -> list[dict[str, Any]]:
    """Per ISO week (Mon-Sun) and sport family: hours, load, sessions, km.

    Only days inside the report window count, so the first/last week can be partial
    (``days`` < 7). Also returns rest days (daily load 0) and the week's Foster
    monotony, which is only reported for weeks with enough days and active days
    (short weeks produce meaningless values).
    """
    window = days_back(end, days)
    by_day: dict[date, list[Activity]] = defaultdict(list)
    for activity in activities_between(activities, window[0], end):
        day = activity_day(activity)
        if day is not None:
            by_day[day].append(activity)

    weeks: dict[str, list[date]] = {}
    for day in window:
        weeks.setdefault(iso_week_label(day), []).append(day)

    rows = []
    for label, week_days in weeks.items():
        sports: dict[str, dict[str, float]] = defaultdict(
            lambda: {"secs": 0.0, "load": 0.0, "n": 0.0, "meters": 0.0}
        )
        daily = []
        for day in week_days:
            day_load = 0.0
            for activity in by_day.get(day, []):
                totals = sports[sport_family(activity.get("type"))]
                totals["secs"] += _num(activity.get("moving_time")) or 0.0
                totals["load"] += activity_load(activity)
                totals["n"] += 1
                totals["meters"] += _num(activity.get("distance")) or 0.0
                day_load += activity_load(activity)
            daily.append(day_load)

        active_days = sum(1 for load in daily if load > 0)
        week_monotony = None
        if (
            len(daily) >= config.week_monotony_min_days
            and active_days >= config.week_monotony_min_active_days
        ):
            week_monotony = monotony(daily)

        sport_rows: dict[str, dict[str, Any]] = {}
        for family in sorted(sports):
            totals = sports[family]
            row: dict[str, Any] = {
                "h": _r(totals["secs"] / 3600, 1),
                "load": _r(totals["load"], 0),
                "n": int(totals["n"]),
            }
            if totals["meters"] > 0:
                row["km"] = _r(totals["meters"] / 1000, 1)
            sport_rows[family] = row

        rows.append(
            {
                "week": label,
                "days": len(week_days),
                "sports": sport_rows,
                "total": {
                    "h": _r(sum(t["secs"] for t in sports.values()) / 3600, 1),
                    "load": _r(sum(daily), 0),
                    "n": int(sum(t["n"] for t in sports.values())),
                },
                "rest_days": sum(1 for load in daily if load == 0),
                "monotony": _r(week_monotony, 2),
            }
        )
    return rows
